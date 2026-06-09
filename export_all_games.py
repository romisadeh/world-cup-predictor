"""
One-time export of all WC 2026 game probabilities to CSV.

Fetches live data from Polymarket (moneyline + exact scores) for every upcoming
match and merges it with bookmaker consensus from odds_consensus.csv.

Output: data/all_games_export.csv

Usage:
    python export_all_games.py
"""

import os
import time
import pandas as pd

from consensus_predict import (
    fetch_polymarket_2026,
    fetch_fifwc_slug_index,
    fetch_polymarket_match,
    fetch_polymarket_exact_scores,
    build_consensus,
    top_scorelines,
    estimate_expected_goals,
    scoreline_matrix,
)

DATA_DIR = "data"


def main():
    odds_file = f"{DATA_DIR}/odds_consensus.csv"
    if not os.path.exists(odds_file):
        print("Missing data/odds_consensus.csv — run python 1_fetch_data.py first.")
        return

    odds = pd.read_csv(odds_file)
    odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True)
    odds = odds.sort_values("commence_time")

    print("Fetching Polymarket tournament-winner market...")
    poly_winner = fetch_polymarket_2026()
    print(f"  {len(poly_winner)} teams found in winner market")

    print(f"Fetching slug index ({len(fetch_fifwc_slug_index())} matches)...")

    rows = []
    total = len(odds)
    for i, (_, row) in enumerate(odds.iterrows(), 1):
        home, away = row["home_team"], row["away_team"]
        kickoff = row["commence_time"].strftime("%Y-%m-%d %H:%M UTC")
        print(f"[{i}/{total}] {home} vs {away} ...", end=" ", flush=True)

        poly_match = fetch_polymarket_match(home, away)
        poly_exact = fetch_polymarket_exact_scores(home, away)

        cons_h, cons_d, cons_a, sources = build_consensus(
            row, poly_match=poly_match, poly_winner=poly_winner,
            w_bookmakers=0.7, w_polymarket=0.3,
        )

        # Scoreline top 5
        if poly_exact:
            top5 = top_scorelines(poly_exact, n=5)
            scoreline_source = "polymarket"
        else:
            lam_h, lam_a = estimate_expected_goals(cons_h, cons_d, cons_a)
            top5 = top_scorelines(scoreline_matrix(lam_h, lam_a), n=5)
            scoreline_source = "poisson"

        # Polymarket moneyline (normalised)
        if poly_match:
            pm_total = sum(poly_match)
            pm_h = round(poly_match[0] / pm_total * 100, 1)
            pm_d = round(poly_match[1] / pm_total * 100, 1)
            pm_a = round(poly_match[2] / pm_total * 100, 1)
        else:
            pm_h = pm_d = pm_a = None

        out = {
            "kickoff": kickoff,
            "home_team": home,
            "away_team": away,
            "n_bookmakers": int(row["n_bookmakers"]),
            # Bookmaker consensus
            "book_home_%": round(row["avg_home_prob"] * 100, 1),
            "book_draw_%": round(row["avg_draw_prob"] * 100, 1),
            "book_away_%": round(row["avg_away_prob"] * 100, 1),
            # Polymarket moneyline
            "poly_home_%": pm_h,
            "poly_draw_%": pm_d,
            "poly_away_%": pm_a,
            "poly_moneyline_source": "match" if poly_match else (
                "winner_fallback" if "polymarket_winner" in sources else "none"
            ),
            # Final consensus (70/30 blend)
            "consensus_home_%": round(cons_h * 100, 1),
            "consensus_draw_%": round(cons_d * 100, 1),
            "consensus_away_%": round(cons_a * 100, 1),
            "consensus_sources": ", ".join(sources),
            # Scorelines
            "scoreline_source": scoreline_source,
            # Scores use "H - A" format (spaces around dash) so Excel doesn't misread as a date
            "score_1st": top5[0][0].replace("-", " - ") if len(top5) > 0 else None,
            "score_1st_%": top5[0][1] if len(top5) > 0 else None,
            "score_2nd": top5[1][0].replace("-", " - ") if len(top5) > 1 else None,
            "score_2nd_%": top5[1][1] if len(top5) > 1 else None,
            "score_3rd": top5[2][0].replace("-", " - ") if len(top5) > 2 else None,
            "score_3rd_%": top5[2][1] if len(top5) > 2 else None,
            "score_4th": top5[3][0].replace("-", " - ") if len(top5) > 3 else None,
            "score_4th_%": top5[3][1] if len(top5) > 3 else None,
            "score_5th": top5[4][0].replace("-", " - ") if len(top5) > 4 else None,
            "score_5th_%": top5[4][1] if len(top5) > 4 else None,
        }
        rows.append(out)

        poly_tag = "match" if poly_match else ("winner" if poly_winner else "none")
        scores_tag = f"{top5[0][0]} ({top5[0][1]}%)" if top5 else "n/a"
        print(f"poly={poly_tag} | top score: {scores_tag}")

        time.sleep(0.3)  # gentle rate limiting

    df = pd.DataFrame(rows)
    out_path = f"{DATA_DIR}/all_games_export.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} games to {out_path}")


if __name__ == "__main__":
    main()

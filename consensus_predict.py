"""
World Cup 2026 - Live Consensus Predictor
===========================================
Combines live probabilities from 55 bookmakers (The Odds API) + Polymarket
into a single consensus forecast for each 2026 World Cup match.

Also estimates SCORELINE probabilities (e.g. 2-1, 1-0) using a Poisson model
derived from the consensus win/draw/loss probabilities.

** No historical data. No 2022. Just the live wisdom of the market. **

Run after: python 1_fetch_data.py   (to get fresh bookmaker odds)
Usage:
    python consensus_predict.py            # next match
    python consensus_predict.py --next 5   # next 5 matches
    python consensus_predict.py --all      # all matches
"""

import os
import sys
import json
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.stats import poisson

DATA_DIR = "data"
GAMMA_API = "https://gamma-api.polymarket.com"


# ===========================================================================
# 1. POLYMARKET - fetch live 2026 World Cup markets
# ===========================================================================

def fetch_polymarket_2026():
    """
    Fetch active 2026 World Cup markets from the Gamma API (no auth needed).
    Returns dict: {team_name: win_probability} from the "World Cup Winner" market.
    """
    print("Fetching Polymarket (2026 World Cup)...", end=" ", flush=True)

    probs = {}
    slugs = ["world-cup-winner", "2026-world-cup-winner", "fifa-world-cup-2026-winner"]

    event = None
    for slug in slugs:
        try:
            resp = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=20)
            if resp.status_code == 200:
                events = resp.json()
                if events:
                    event = events[0] if isinstance(events, list) else events
                    break
        except Exception:
            continue

    try:
        if event:
            for market in event.get("markets", []):
                team = market.get("groupItemTitle") or market.get("question", "")
                prices = market.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    try:
                        prices = json.loads(prices)
                    except Exception:
                        prices = []
                if prices and team:
                    probs[team.strip()] = float(prices[0])
        print(f"OK ({len(probs)} teams)")
    except Exception as e:
        print(f"failed ({e})")
        print("   Continuing with bookmaker odds only")

    return probs


def fetch_polymarket_match(home, away):
    """
    Try to find a Polymarket market specific to this match (home vs away).
    Returns (p_home, p_draw, p_away) or None if not found.
    """
    try:
        resp = requests.get(
            f"{GAMMA_API}/public-search",
            params={"q": f"{home} {away} World Cup", "limit_per_type": 5},
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        for event in data.get("events", []):
            title = event.get("title", "").lower()
            if home.lower() in title and away.lower() in title:
                outcomes = {}
                for m in event.get("markets", []):
                    name = (m.get("groupItemTitle") or "").lower()
                    prices = m.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        prices = json.loads(prices) if prices else []
                    if prices:
                        outcomes[name] = float(prices[0])

                p_home = outcomes.get(home.lower())
                p_away = outcomes.get(away.lower())
                p_draw = outcomes.get("draw") or outcomes.get("tie")
                if p_home and p_away:
                    return (p_home, p_draw or 0, p_away)
    except Exception:
        pass
    return None


# ===========================================================================
# 2. CONSENSUS - merge sources into one forecast
# ===========================================================================

def build_consensus(odds_row, poly_match=None, poly_winner=None,
                    w_bookmakers=0.7, w_polymarket=0.3):
    """
    Merge all probability sources into a weighted consensus forecast.

    Sources:
      1. Bookmaker consensus (already aggregated from 55 providers) - high weight
      2. Polymarket for the specific match (if available)

    Default weights: 70% bookmakers, 30% Polymarket.
    Returns (p_home, p_draw, p_away, list_of_sources_used).
    """
    book_h = odds_row["avg_home_prob"]
    book_d = odds_row["avg_draw_prob"]
    book_a = odds_row["avg_away_prob"]

    sources = [("bookmakers", book_h, book_d, book_a, w_bookmakers)]

    if poly_match:
        ph, pd_, pa = poly_match
        total = ph + pd_ + pa
        if total > 0:
            sources.append(("polymarket_match", ph/total, pd_/total, pa/total, w_polymarket))

    total_w = sum(s[4] for s in sources)
    cons_h = sum(s[1] * s[4] for s in sources) / total_w
    cons_d = sum(s[2] * s[4] for s in sources) / total_w
    cons_a = sum(s[3] * s[4] for s in sources) / total_w

    t = cons_h + cons_d + cons_a
    return cons_h/t, cons_d/t, cons_a/t, [s[0] for s in sources]


# ===========================================================================
# 3. POISSON - scoreline probabilities
# ===========================================================================

def estimate_expected_goals(p_home, p_draw, p_away, max_goals=10):
    """
    Reverse-engineer each team's expected goals (xG) from the win/draw/loss
    probabilities. Vectorized grid search: finds the (lambda_home, lambda_away)
    pair whose Poisson outcome distribution best matches the consensus.
    Runs in well under a second.
    """
    lams = np.arange(0.2, 3.5, 0.05)
    goals = np.arange(max_goals + 1)
    # pmf_table[i, g] = P(g goals | lambda = lams[i])
    pmf_table = poisson.pmf(goals[None, :], lams[:, None])

    # index helpers on the (home_goals x away_goals) joint matrix
    il = np.tril_indices(max_goals + 1, k=-1)   # home > away  -> home win
    iu = np.triu_indices(max_goals + 1, k=1)    # home < away  -> away win
    diag = np.arange(max_goals + 1)

    best = None
    best_err = float("inf")
    for i, lam_h in enumerate(lams):
        ph_vec = pmf_table[i]
        for j, lam_a in enumerate(lams):
            joint = np.outer(ph_vec, pmf_table[j])  # joint[h, a]
            home_win = joint[il].sum()
            draw = joint[diag, diag].sum()
            away_win = joint[iu].sum()
            err = (home_win - p_home) ** 2 + (draw - p_draw) ** 2 + (away_win - p_away) ** 2
            if err < best_err:
                best_err = err
                best = (lam_h, lam_a)
    return best


def scoreline_matrix(lam_h, lam_a, max_goals=6):
    """
    Build a matrix of scoreline probabilities using two independent Poisson
    distributions. Returns dict {(home_goals, away_goals): probability}.
    """
    matrix = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[(h, a)] = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
    total = sum(matrix.values())
    return {k: v / total for k, v in matrix.items()}


def top_scorelines(matrix, n=5):
    """Return the n most likely scorelines as a list of (score_str, pct)."""
    ranked = sorted(matrix.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(f"{h}-{a}", round(p * 100, 1)) for (h, a), p in ranked]


# ===========================================================================
# 4. DISPLAY
# ===========================================================================

def _bar(pct, width=20):
    filled = int(round(pct / 100 * width))
    return "#" * filled + "-" * (width - filled)


def get_matches(n=1, show_all=False):
    odds_file = f"{DATA_DIR}/odds_consensus.csv"
    if not os.path.exists(odds_file):
        print("Missing odds_consensus.csv - run first: python 1_fetch_data.py")
        sys.exit(1)

    odds = pd.read_csv(odds_file)
    odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True)
    now = datetime.now(timezone.utc)
    upcoming = odds[odds["commence_time"] > now].sort_values("commence_time")
    if upcoming.empty:
        upcoming = odds.sort_values("commence_time")
    return upcoming if show_all else upcoming.head(n)


def predict(n=1, show_all=False, w_book=0.7, w_poly=0.3):
    print("=" * 56)
    print("World Cup 2026 - Live Consensus Predictor")
    print("=" * 56)

    poly_winner = fetch_polymarket_2026()

    matches = get_matches(n, show_all)
    print(f"{len(matches)} matches | weights: {int(w_book*100)}% bookmakers + "
          f"{int(w_poly*100)}% Polymarket\n")

    results = []
    for _, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        kickoff = m["commence_time"]

        poly_match = fetch_polymarket_match(home, away)

        cons_h, cons_d, cons_a, used_sources = build_consensus(
            m, poly_match=poly_match, poly_winner=poly_winner,
            w_bookmakers=w_book, w_polymarket=w_poly,
        )
        pH, pD, pA = cons_h * 100, cons_d * 100, cons_a * 100
        winner = [f"{home} win", "Draw", f"{away} win"][int(np.argmax([pH, pD, pA]))]

        lam_h, lam_a = estimate_expected_goals(cons_h, cons_d, cons_a)
        matrix = scoreline_matrix(lam_h, lam_a)
        top5 = top_scorelines(matrix, n=5)
        most_likely_score = top5[0]

        print("+" + "-" * 54)
        print(f"|  {home}  vs  {away}")
        print(f"|  {kickoff.strftime('%d/%m/%Y %H:%M')} UTC")
        print(f"|  Sources: {', '.join(used_sources)} ({int(m['n_bookmakers'])} bookmakers)")
        print("+" + "-" * 54)
        print(f"|  {home:<18} {_bar(pH)} {pH:5.1f}%")
        print(f"|  {'Draw':<18} {_bar(pD)} {pD:5.1f}%")
        print(f"|  {away:<18} {_bar(pA)} {pA:5.1f}%")
        print("+" + "-" * 54)
        print(f"|  Outcome prediction: {winner}")
        print(f"|  Expected goals: {home} {lam_h:.2f} - {lam_a:.2f} {away}")
        print(f"|  Most likely score: {most_likely_score[0]} ({most_likely_score[1]}%)")
        print(f"|  Top scorelines:")
        for score, pct in top5:
            print(f"|     {score:<6} {_bar(pct, 15)} {pct:4.1f}%")
        print("+" + "-" * 54)
        print()

        results.append({
            "home_team": home, "away_team": away,
            "commence_time": kickoff,
            "consensus_home": round(pH, 1),
            "consensus_draw": round(pD, 1),
            "consensus_away": round(pA, 1),
            "prediction": winner,
            "xg_home": round(lam_h, 2),
            "xg_away": round(lam_a, 2),
            "most_likely_score": most_likely_score[0],
            "most_likely_score_pct": most_likely_score[1],
            "sources": ",".join(used_sources),
        })

    if results:
        out = pd.DataFrame(results)
        out.to_csv(f"{DATA_DIR}/consensus_predictions.csv", index=False)
        print(f"Saved: {DATA_DIR}/consensus_predictions.csv")

    return results


# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--next", type=int, default=1, help="number of upcoming matches")
    parser.add_argument("--all", action="store_true", help="all matches")
    parser.add_argument("--w-book", type=float, default=0.7, help="bookmaker weight (0-1)")
    parser.add_argument("--w-poly", type=float, default=0.3, help="Polymarket weight (0-1)")
    args = parser.parse_args()

    predict(n=args.next, show_all=args.all, w_book=args.w_book, w_poly=args.w_poly)

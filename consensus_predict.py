"""
World Cup 2026 - Live Consensus Predictor
===========================================
מאחד הסתברויות חיות מ-55 בוקמייקרים (The Odds API) + Polymarket
לתחזית קונצנזוס אחת לכל משחק של מונדיאל 2026.

** ללא נתונים היסטוריים. ללא 2022. רק חוכמת השוק של עכשיו. **

הרץ אחרי: python 1_fetch_data.py  (כדי שיהיו אודס בוקמייקרים טריים)
שימוש:
    python consensus_predict.py            # המשחק הקרוב ביותר
    python consensus_predict.py --next 5   # 5 המשחקים הקרובים
    python consensus_predict.py --all      # כל המשחקים
"""

import os
import re
import sys
import json
import argparse
import functools
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
from datetime import datetime, timezone

DATA_DIR = "data"
GAMMA_API = "https://gamma-api.polymarket.com"
WC_SERIES_SLUG = "soccer-fifwc"


# ═══════════════════════════════════════════════════════════════════════════
# 0. HELPERS — name normalisation & slug index
# ═══════════════════════════════════════════════════════════════════════════

# Bookmaker/common name ↔ Polymarket name mapping (covers both directions via
# the auto-generated reverse below).
_NAME_ALIASES = {
    "czech republic":           "czechia",
    "turkey":                   "turkiye",
    "bosnia and herzegovina":   "bosnia-herzegovina",
    "bosnia":                   "bosnia-herzegovina",
    "dr congo":                 "congo dr",
    "congo":                    "congo dr",
    "democratic republic of congo": "congo dr",
    "cape verde":               "cabo verde",
    "curacao":                  "curaçao",
    "republic of ireland":      "ireland",
    "trinidad & tobago":        "trinidad and tobago",
    "united states":            "usa",
    "united states of america": "usa",
    "korea republic":           "south korea",
    "iran (islamic republic)":  "iran",
    "ir iran":                  "iran",
    "ivory coast":              "côte d'ivoire",
    "cote d'ivoire":            "côte d'ivoire",
}
_NAME_ALIASES.update({v: k for k, v in list(_NAME_ALIASES.items()) if v not in _NAME_ALIASES})


def _parse_price_list(val):
    """Parse outcomePrices which may be a JSON string or a real list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def _teams_match(csv_name: str, poly_name: str) -> bool:
    """Return True if the two team name strings refer to the same team."""
    a = csv_name.strip().lower()
    b = poly_name.strip().lower()
    if a == b:
        return True
    # static alias in either direction
    if _NAME_ALIASES.get(a, "").lower() == b:
        return True
    if _NAME_ALIASES.get(b, "").lower() == a:
        return True
    # substring
    if a in b or b in a:
        return True
    # token overlap: any significant word (>=4 chars) from a appears in b
    tokens = {w for w in a.split() if len(w) >= 4}
    return bool(tokens and any(tok in b for tok in tokens))


@functools.lru_cache(maxsize=1)
def fetch_fifwc_slug_index():
    """
    Fetches all active FIFA WC match events from the Gamma series.
    Returns a tuple of (home_poly_name, away_poly_name, slug) for each match.
    Cached for the lifetime of the process.
    """
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={"series_slug": WC_SERIES_SLUG, "active": True, "limit": 150},
            timeout=20,
        )
        if resp.status_code != 200:
            return ()
        result = []
        for ev in resp.json():
            slug = ev.get("slug", "")
            title = ev.get("title", "")
            # Only plain match events (skip exact-score / halftime sub-events)
            if " vs. " not in title or "exact-score" in slug or "halftime" in slug:
                continue
            parts = title.split(" vs. ", 1)
            if len(parts) == 2:
                result.append((parts[0].strip(), parts[1].strip(), slug))
        return tuple(result)
    except Exception:
        return ()


def _find_match_slug(home: str, away: str):
    """Return the Polymarket event slug for a given home/away pair, or None."""
    for home_poly, away_poly, slug in fetch_fifwc_slug_index():
        if _teams_match(home, home_poly) and _teams_match(away, away_poly):
            return slug
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 1. POLYMARKET — שליפת שווקי מונדיאל 2026 חיים
# ═══════════════════════════════════════════════════════════════════════════

def fetch_polymarket_2026():
    """
    מושך שווקי מונדיאל 2026 פעילים מ-Gamma API (ללא צורך באימות).
    מחזיר dict: {team_name: win_probability} משוק "World Cup Winner".
    """
    probs = {}
    slugs = ["world-cup-winner", "2026-world-cup-winner", "fifa-world-cup-2026-winner"]

    event = None
    for slug in slugs:
        try:
            resp = requests.get(
                f"{GAMMA_API}/events",
                params={"slug": slug},
                timeout=20,
            )
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
                prices = _parse_price_list(market.get("outcomePrices", "[]"))
                if prices and team:
                    probs[team.strip()] = float(prices[0])

    except Exception:
        pass

    return probs


def _parse_moneyline_event(event, home, away):
    """
    Extract (p_home, p_draw, p_away) from a moneyline event dict.
    Returns None if home or away probabilities cannot be found.
    """
    team_probs = {}
    p_draw = None

    for m in event.get("markets", []):
        git = (m.get("groupItemTitle") or "").strip()
        prices = _parse_price_list(m.get("outcomePrices"))
        if not git or not prices:
            continue
        try:
            yes_price = float(prices[0])
        except Exception:
            continue
        if "draw" in git.lower() or "tie" in git.lower():
            p_draw = yes_price
        else:
            team_probs[git] = yes_price

    def _find(team):
        for g, v in team_probs.items():
            if g.strip().lower() == team.strip().lower():
                return v
        for g, v in team_probs.items():
            if _teams_match(team, g):
                return v
        return None

    p_home = _find(home)
    p_away = _find(away)
    if p_home is not None and p_away is not None:
        return (p_home, p_draw or 0.0, p_away)
    return None


def fetch_polymarket_match(home, away):
    """
    מוצא שוק Polymarket ספציפי למשחק. מחזיר (p_home, p_draw, p_away) או None.
    Uses the slug index first (reliable); falls back to public-search.
    """
    # Primary path: slug index → direct event fetch
    slug = _find_match_slug(home, away)
    if slug:
        try:
            resp = requests.get(
                f"{GAMMA_API}/events",
                params={"slug": slug},
                timeout=15,
            )
            if resp.status_code == 200:
                events = resp.json()
                if events:
                    res = _parse_moneyline_event(events[0], home, away)
                    if res:
                        return res
        except Exception:
            pass

    # Fallback: public-search text query
    try:
        resp = requests.get(
            f"{GAMMA_API}/public-search",
            params={"q": f"{home} {away}", "limit_per_type": 10},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        for event in (resp.json().get("events") or []):
            title = (event.get("title") or "").lower()
            if home.lower() in title and away.lower() in title:
                res = _parse_moneyline_event(event, home, away)
                if res:
                    return res
    except Exception:
        pass

    return None


def fetch_polymarket_exact_scores(home, away):
    """
    Fetches Polymarket exact-score market for a given match.
    Returns {(home_goals, away_goals): probability} dict, normalized to sum=1,
    or None if no market exists for this match.
    """
    slug = _find_match_slug(home, away)
    if not slug:
        return None

    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={"slug": f"{slug}-exact-score"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        events = resp.json()
        if not events:
            return None
        event = events[0]
    except Exception:
        return None

    matrix = {}
    for m in event.get("markets", []):
        git = m.get("groupItemTitle", "")
        if not git or "any other" in git.lower():
            continue
        score_match = re.search(r"(\d+)\s*-\s*(\d+)", git)
        if not score_match:
            continue
        h_goals = int(score_match.group(1))
        a_goals = int(score_match.group(2))
        prices = _parse_price_list(m.get("outcomePrices"))
        if prices:
            try:
                matrix[(h_goals, a_goals)] = float(prices[0])
            except Exception:
                pass

    if not matrix:
        return None

    total = sum(matrix.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in matrix.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONSENSUS — איחוד מקורות לתחזית אחת
# ═══════════════════════════════════════════════════════════════════════════

def _poly_winner_to_match_probs(home, away, poly_winner, book_draw=None):
    """
    Bradley-Terry fallback: derives per-match (p_home, p_draw, p_away) from the
    tournament-winner market when no per-match Polymarket market exists.

    p(A beats B) ≈ p_winner_A / (p_winner_A + p_winner_B)
    Draw probability is anchored to the bookmaker draw (or 0.25 if unavailable).
    """
    lw = {k.strip().lower(): v for k, v in poly_winner.items()}

    def _lookup(team):
        t = team.strip().lower()
        if t in lw:
            return lw[t]
        alias = _NAME_ALIASES.get(t)
        if alias and alias.lower() in lw:
            return lw[alias.lower()]
        for k, v in lw.items():
            if t in k or k in t:
                return v
        tokens = {w for w in t.split() if len(w) >= 4}
        for k, v in lw.items():
            if any(tok in k for tok in tokens):
                return v
        return None

    p_h = _lookup(home)
    p_a = _lookup(away)
    if p_h is None or p_a is None or (p_h + p_a) == 0:
        return None

    r = p_h / (p_h + p_a)
    p_draw = max(0.10, min(0.40, book_draw if book_draw is not None else 0.25))
    return (r * (1 - p_draw), p_draw, (1 - r) * (1 - p_draw))


def build_consensus(odds_row, poly_match=None, poly_winner=None,
                    w_bookmakers=0.7, w_polymarket=0.3):
    """
    מאחד את מקורות ההסתברות לתחזית קונצנזוס משוקללת.

    מקורות:
      1. אודס בוקמייקרים (consensus כבר מ-55 ספקים) — משקל גבוה
      2. Polymarket למשחק הספציפי (אם קיים), או שוק מנצח הטורניר כ-fallback

    משקלים ניתנים לכוונון. ברירת מחדל: 70% בוקמייקרים, 30% Polymarket.
    """
    book_h = odds_row["avg_home_prob"]
    book_d = odds_row["avg_draw_prob"]
    book_a = odds_row["avg_away_prob"]

    sources = [("bookmakers", book_h, book_d, book_a, w_bookmakers)]

    poly_source = poly_match
    poly_source_name = "polymarket_match"
    if poly_source is None and poly_winner:
        home_team = str(odds_row.get("home_team", ""))
        away_team = str(odds_row.get("away_team", ""))
        poly_source = _poly_winner_to_match_probs(home_team, away_team, poly_winner, book_d)
        poly_source_name = "polymarket_winner"

    if poly_source:
        ph, pd_, pa = poly_source
        total = ph + pd_ + pa
        if total > 0:
            sources.append((poly_source_name, ph / total, pd_ / total, pa / total, w_polymarket))

    total_w = sum(s[4] for s in sources)
    if total_w <= 0:
        t = book_h + book_d + book_a
        return book_h / t, book_d / t, book_a / t, ["bookmakers"]

    cons_h = sum(s[1] * s[4] for s in sources) / total_w
    cons_d = sum(s[2] * s[4] for s in sources) / total_w
    cons_a = sum(s[3] * s[4] for s in sources) / total_w

    t = cons_h + cons_d + cons_a
    return cons_h / t, cons_d / t, cons_a / t, [s[0] for s in sources]


# ═══════════════════════════════════════════════════════════════════════════
# 2b. POISSON — הסתברויות תוצאות מדויקות (scoreline)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_expected_goals(p_home, p_draw, p_away, max_goals=10):
    """
    מחשב לאחור את ה-xG (שערים צפויים) של כל קבוצה מתוך הסתברויות ניצחון/תיקו/הפסד.
    חיפוש רשת ווקטורי: מוצא את הזוג (lambda_home, lambda_away) שהתפלגות הפואסון
    שלו הכי תואמת את הקונצנזוס. רץ בפחות משנייה.
    """
    lams = np.arange(0.2, 3.5, 0.05)
    goals = np.arange(max_goals + 1)
    pmf_table = poisson.pmf(goals[None, :], lams[:, None])

    il = np.tril_indices(max_goals + 1, k=-1)
    iu = np.triu_indices(max_goals + 1, k=1)
    diag = np.arange(max_goals + 1)

    best = None
    best_err = float("inf")
    for i, lam_h in enumerate(lams):
        ph_vec = pmf_table[i]
        for j, lam_a in enumerate(lams):
            joint = np.outer(ph_vec, pmf_table[j])
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
    בונה מטריצת הסתברויות לתוצאות, בעזרת שתי התפלגויות פואסון בלתי-תלויות.
    מחזיר dict {(home_goals, away_goals): probability}.
    """
    matrix = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[(h, a)] = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
    total = sum(matrix.values())
    return {k: v / total for k, v in matrix.items()}


def top_scorelines(matrix, n=5):
    """מחזיר את n התוצאות הסבירות ביותר כרשימת (score_str, pct)."""
    ranked = sorted(matrix.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(f"{h}-{a}", round(p * 100, 1)) for (h, a), p in ranked]


# ═══════════════════════════════════════════════════════════════════════════
# 3. תצוגה
# ═══════════════════════════════════════════════════════════════════════════

def _bar(pct, width=20):
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def get_matches(n=1, show_all=False):
    odds_file = f"{DATA_DIR}/odds_consensus.csv"
    if not os.path.exists(odds_file):
        print("⚠️  לא נמצא odds_consensus.csv — הרץ קודם: python 1_fetch_data.py")
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
    print("⚡ World Cup 2026 - Live Consensus Predictor")
    print("=" * 56)

    poly_winner = fetch_polymarket_2026()

    matches = get_matches(n, show_all)
    print(f"📅 {len(matches)} משחקים | משקל: {int(w_book*100)}% בוקמייקרים + {int(w_poly*100)}% Polymarket\n")

    results = []
    for _, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        kickoff = m["commence_time"]

        poly_match = fetch_polymarket_match(home, away)

        cons_h, cons_d, cons_a, used_sources = build_consensus(
            m, poly_match=poly_match, poly_winner=poly_winner,
            w_bookmakers=w_book, w_polymarket=w_poly,
        )

        pH, pD, pA = cons_h*100, cons_d*100, cons_a*100
        winner = [f"ניצחון {home}", "תיקו", f"ניצחון {away}"][np.argmax([pH, pD, pA])]

        print("┌" + "─" * 52)
        print(f"│  🏟️  {home}  vs  {away}")
        print(f"│  🕐 {kickoff.strftime('%d/%m/%Y %H:%M')} UTC")
        print(f"│  📡 מקורות: {', '.join(used_sources)} ({int(m['n_bookmakers'])} בוקמייקרים)")
        print("├" + "─" * 52)
        print(f"│  {home:<18} {_bar(pH)} {pH:5.1f}%")
        print(f"│  {'תיקו':<18} {_bar(pD)} {pD:5.1f}%")
        print(f"│  {away:<18} {_bar(pA)} {pA:5.1f}%")
        print("├" + "─" * 52)
        print(f"│  🎯 תחזית קונצנזוס: {winner}")
        print("└" + "─" * 52)
        print()

        results.append({
            "home_team": home, "away_team": away,
            "commence_time": kickoff,
            "consensus_home": round(pH, 1),
            "consensus_draw": round(pD, 1),
            "consensus_away": round(pA, 1),
            "prediction": winner,
            "sources": ",".join(used_sources),
        })

    if results:
        out = pd.DataFrame(results)
        out.to_csv(f"{DATA_DIR}/consensus_predictions.csv", index=False)
        print(f"💾 נשמר: {DATA_DIR}/consensus_predictions.csv")

    return results


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--next", type=int, default=1, help="כמה משחקים קרובים")
    parser.add_argument("--all", action="store_true", help="כל המשחקים")
    parser.add_argument("--w-book", type=float, default=0.7, help="משקל בוקמייקרים (0-1)")
    parser.add_argument("--w-poly", type=float, default=0.3, help="משקל Polymarket (0-1)")
    args = parser.parse_args()

    predict(n=args.next, show_all=args.all, w_book=args.w_book, w_poly=args.w_poly)

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
import sys
import argparse
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
from datetime import datetime, timezone

DATA_DIR = "data"
GAMMA_API = "https://gamma-api.polymarket.com"


# ═══════════════════════════════════════════════════════════════════════════
# 1. POLYMARKET — שליפת שווקי מונדיאל 2026 חיים
# ═══════════════════════════════════════════════════════════════════════════

def fetch_polymarket_2026():
    """
    מושך שווקי מונדיאל 2026 פעילים מ-Gamma API (ללא צורך באימות).
    מחזיר dict: {team_name: win_probability} משוק "World Cup Winner".
    """
    print("🔮 מושך Polymarket (מונדיאל 2026)...", end=" ", flush=True)

    probs = {}
    # מנסה כמה slugs אפשריים (Polymarket משנים מדי פעם)
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
                # כל market הוא "Will X win?" — שם הקבוצה + מחיר ה-Yes
                team = market.get("groupItemTitle") or market.get("question", "")
                # outcomePrices הוא לרוב string של רשימה: '["0.17","0.83"]'
                prices = market.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    import json
                    try:
                        prices = json.loads(prices)
                    except Exception:
                        prices = []
                if prices and team:
                    yes_price = float(prices[0])
                    probs[team.strip()] = yes_price

        print(f"✅ {len(probs)} קבוצות")

    except Exception as e:
        print(f"⚠️  ({e})")
        print("   ממשיך עם אודס בוקמייקרים בלבד")

    return probs


def fetch_polymarket_match(home, away):
    """
    מוצא שוק Polymarket ספציפי למשחק. מחזיר (p_home, p_draw, p_away) או None.
    מבנה Polymarket: כל תוצאה היא שוק כן/לא נפרד — שם התוצאה ב-groupItemTitle,
    וההסתברות היא המחיר הראשון (Yes). התיקו מסומן כ-"Draw (...)".
    """
    import json

    def _parse_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val.strip():
            try:
                return json.loads(val)
            except Exception:
                return []
        return []

    def _norm(s):
        return (s or "").strip().lower()

    def _parse_event(event):
        team_probs = {}
        p_draw = None
        for m in event.get("markets", []):
            git = (m.get("groupItemTitle") or "").strip()
            prices = _parse_list(m.get("outcomePrices"))
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
                if _norm(g) == _norm(team):
                    return v
            for g, v in team_probs.items():
                if _norm(team) in _norm(g) or _norm(g) in _norm(team):
                    return v
            return None

        p_home = _find(home)
        p_away = _find(away)
        if p_home is not None and p_away is not None:
            return (p_home, p_draw or 0.0, p_away)
        return None

    try:
        resp = requests.get(
            f"{GAMMA_API}/public-search",
            params={"q": f"{home} {away}", "limit_per_type": 10},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        events = resp.json().get("events", []) or []

        for event in events:
            title = _norm(event.get("title"))
            if home.lower() in title and away.lower() in title:
                res = _parse_event(event)
                if res:
                    return res
    except Exception:
        pass

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONSENSUS — איחוד מקורות לתחזית אחת
# ═══════════════════════════════════════════════════════════════════════════

def build_consensus(odds_row, poly_match=None, poly_winner=None,
                    w_bookmakers=0.7, w_polymarket=0.3):
    """
    מאחד את מקורות ההסתברות לתחזית קונצנזוס משוקללת.

    מקורות:
      1. אודס בוקמייקרים (consensus כבר מ-55 ספקים) — משקל גבוה
      2. Polymarket למשחק הספציפי (אם קיים)

    משקלים ניתנים לכוונון. ברירת מחדל: 70% בוקמייקרים, 30% Polymarket.
    """
    # מקור 1: בוקמייקרים (תמיד קיים)
    book_h = odds_row["avg_home_prob"]
    book_d = odds_row["avg_draw_prob"]
    book_a = odds_row["avg_away_prob"]

    sources = [("bookmakers", book_h, book_d, book_a, w_bookmakers)]

    # מקור 2: Polymarket ספציפי למשחק
    if poly_match:
        ph, pd_, pa = poly_match
        total = ph + pd_ + pa
        if total > 0:
            sources.append(("polymarket_match", ph / total, pd_ / total, pa / total, w_polymarket))

    # שקלול — עם הגנה מפני משקל כולל אפס
    total_w = sum(s[4] for s in sources)
    if total_w <= 0:
        # אין נתוני Polymarket וגם 0% בוקמייקרים — נופלים חזרה לבוקמייקרים
        t = book_h + book_d + book_a
        return book_h / t, book_d / t, book_a / t, ["bookmakers"]

    cons_h = sum(s[1] * s[4] for s in sources) / total_w
    cons_d = sum(s[2] * s[4] for s in sources) / total_w
    cons_a = sum(s[3] * s[4] for s in sources) / total_w

    # נרמול סופי ל-100%
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
    # pmf_table[i, g] = P(g שערים | lambda = lams[i])
    pmf_table = poisson.pmf(goals[None, :], lams[:, None])

    il = np.tril_indices(max_goals + 1, k=-1)   # home > away  -> ניצחון בית
    iu = np.triu_indices(max_goals + 1, k=1)    # home < away  -> ניצחון חוץ
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

    # מושך Polymarket פעם אחת (שוק הזוכה)
    poly_winner = fetch_polymarket_2026()

    matches = get_matches(n, show_all)
    print(f"📅 {len(matches)} משחקים | משקל: {int(w_book*100)}% בוקמייקרים + {int(w_poly*100)}% Polymarket\n")

    results = []
    for _, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        kickoff = m["commence_time"]

        # מנסה למצוא שוק Polymarket ספציפי למשחק
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

    # שמירה
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

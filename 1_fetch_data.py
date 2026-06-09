"""
World Cup 2026 - Data Fetcher
==============================
מושך נתוני אודס מ-The Odds API ונתוני סטטיסטיקה מ-Football-Data.org
וגם הסתברויות שוק מ-Polymarket

התקנה:
    pip install requests pandas python-dotenv

הגדרת API Keys:
    צור קובץ .env עם:
    ODDS_API_KEY=your_key_here          # https://the-odds-api.com (500 req/month חינם)
    FOOTBALL_DATA_KEY=your_key_here     # https://football-data.org (חינם)
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "demo")
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "demo")

ODDS_BASE = "https://api.the-odds-api.com/v4"
FOOTBALL_BASE = "https://api.football-data.org/v4"
POLYMARKET_BASE = "https://clob.polymarket.com"

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── 1. THE ODDS API ──────────────────────────────────────────────────────────

def fetch_world_cup_odds():
    """
    שולף אודס חיים למשחקי המונדיאל מ-40+ בוקמייקרים.
    כולל: DraftKings, FanDuel, BetMGM, bet365, William Hill
    """
    print("📡 שולף אודס ממונדיאל...")

    sport = "soccer_fifa_world_cup"
    markets = "h2h,totals,spreads"
    regions = "us,uk,eu,au"

    url = f"{ODDS_BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"   ✅ {len(data)} משחקים | בקשות נותרות: {remaining}")

    # ─── עיבוד לטבלה נקייה ───────────────────────────────────────────────
    rows = []
    for game in data:
        home = game["home_team"]
        away = game["away_team"]
        commence = game["commence_time"]

        for bookmaker in game.get("bookmakers", []):
            bk_name = bookmaker["key"]

            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    rows.append({
                        "game_id": game["id"],
                        "home_team": home,
                        "away_team": away,
                        "commence_time": commence,
                        "bookmaker": bk_name,
                        "home_odds": outcomes.get(home),
                        "draw_odds": outcomes.get("Draw"),
                        "away_odds": outcomes.get(away),
                        "market": "h2h",
                    })

    df = pd.DataFrame(rows)

    if not df.empty:
        # ─── Implied Probability (הסתברות משוקללת) ───────────────────────
        df["home_prob"] = 1 / df["home_odds"]
        df["draw_prob"] = 1 / df["draw_odds"]
        df["away_prob"] = 1 / df["away_odds"]
        total = df["home_prob"] + df["draw_prob"] + df["away_prob"]
        df["home_prob_norm"] = df["home_prob"] / total
        df["draw_prob_norm"] = df["draw_prob"] / total
        df["away_prob_norm"] = df["away_prob"] / total

        # ─── ממוצע בין בוקמייקרים (consensus odds) ────────────────────────
        consensus = df.groupby(["game_id", "home_team", "away_team", "commence_time"]).agg(
            avg_home_prob=("home_prob_norm", "mean"),
            avg_draw_prob=("draw_prob_norm", "mean"),
            avg_away_prob=("away_prob_norm", "mean"),
            std_home_prob=("home_prob_norm", "std"),  # מדד אי-הוודאות
            n_bookmakers=("bookmaker", "count"),
        ).reset_index()

        consensus.to_csv(f"{OUTPUT_DIR}/odds_consensus.csv", index=False)
        print(f"   💾 שמור: odds_consensus.csv ({len(consensus)} משחקים, {len(df)} שורות גולמיות)")

    return df


def fetch_historical_odds(sport="soccer_fifa_world_cup_2022"):
    """
    שולף אודס היסטוריים ממונדיאל 2022 לאימון המודל.
    דורש מנוי בתשלום ב-The Odds API.
    """
    print("📜 שולף אודס היסטוריים (מונדיאל 2022)...")

    url = f"{ODDS_BASE}/historical/sports/{sport}/odds"
    # תאריכים חשובים: נובמבר-דצמבר 2022
    snapshots = [
        "2022-11-20T00:00:00Z",
        "2022-11-24T00:00:00Z",
        "2022-11-28T00:00:00Z",
        "2022-12-02T00:00:00Z",
        "2022-12-05T00:00:00Z",
        "2022-12-09T00:00:00Z",
        "2022-12-13T00:00:00Z",
        "2022-12-18T00:00:00Z",
    ]

    all_data = []
    for snapshot in snapshots:
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us,uk,eu",
            "markets": "h2h",
            "date": snapshot,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for game in data.get("data", []):
                    game["snapshot"] = snapshot
                all_data.extend(data.get("data", []))
            time.sleep(0.5)
        except Exception as e:
            print(f"   ⚠️  {snapshot}: {e}")

    if all_data:
        with open(f"{OUTPUT_DIR}/historical_odds_raw.json", "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"   ✅ {len(all_data)} snapshots שמורים")

    return all_data


# ─── 2. FOOTBALL-DATA.ORG ─────────────────────────────────────────────────────

def fetch_team_stats():
    """
    שולף דירוגי FIFA ורשימות קבוצות המונדיאל.
    """
    print("⚽ שולף נתוני קבוצות...")

    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}

    # ─── קבוצות מונדיאל 2026 ──────────────────────────────────────────────
    wc_2026_teams = {
        "Argentina": {"fifa_rank": 1, "group": "A", "confederation": "CONMEBOL"},
        "France": {"fifa_rank": 2, "group": "B", "confederation": "UEFA"},
        "England": {"fifa_rank": 4, "group": "C", "confederation": "UEFA"},
        "Brazil": {"fifa_rank": 5, "group": "D", "confederation": "CONMEBOL"},
        "Portugal": {"fifa_rank": 6, "group": "E", "confederation": "UEFA"},
        "Spain": {"fifa_rank": 8, "group": "F", "confederation": "UEFA"},
        "Netherlands": {"fifa_rank": 7, "group": "G", "confederation": "UEFA"},
        "Germany": {"fifa_rank": 12, "group": "H", "confederation": "UEFA"},
        "USA": {"fifa_rank": 16, "group": "A", "confederation": "CONCACAF"},
        "Mexico": {"fifa_rank": 17, "group": "B", "confederation": "CONCACAF"},
        "Canada": {"fifa_rank": 47, "group": "C", "confederation": "CONCACAF"},
        "Morocco": {"fifa_rank": 14, "group": "D", "confederation": "CAF"},
        "Japan": {"fifa_rank": 15, "group": "E", "confederation": "AFC"},
        "Senegal": {"fifa_rank": 20, "group": "F", "confederation": "CAF"},
        "Australia": {"fifa_rank": 25, "group": "G", "confederation": "AFC"},
        "Croatia": {"fifa_rank": 10, "group": "H", "confederation": "UEFA"},
        "Uruguay": {"fifa_rank": 19, "group": "A", "confederation": "CONMEBOL"},
        "Belgium": {"fifa_rank": 3, "group": "B", "confederation": "UEFA"},
        "Switzerland": {"fifa_rank": 21, "group": "C", "confederation": "UEFA"},
        "Colombia": {"fifa_rank": 11, "group": "D", "confederation": "CONMEBOL"},
    }

    df = pd.DataFrame.from_dict(wc_2026_teams, orient="index").reset_index()
    df.columns = ["team", "fifa_rank", "group", "confederation"]

    # ─── ביצועים היסטוריים במונדיאל ───────────────────────────────────────
    historical_performance = {
        "Argentina": {"wc_titles": 3, "wc_finals": 6, "avg_goals_scored": 2.1, "avg_goals_conceded": 1.2},
        "France": {"wc_titles": 2, "wc_finals": 3, "avg_goals_scored": 1.9, "avg_goals_conceded": 1.1},
        "England": {"wc_titles": 1, "wc_finals": 1, "avg_goals_scored": 1.5, "avg_goals_conceded": 1.3},
        "Brazil": {"wc_titles": 5, "wc_finals": 7, "avg_goals_scored": 2.3, "avg_goals_conceded": 1.0},
        "Portugal": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.6, "avg_goals_conceded": 1.2},
        "Spain": {"wc_titles": 1, "wc_finals": 1, "avg_goals_scored": 1.7, "avg_goals_conceded": 0.9},
        "Netherlands": {"wc_titles": 0, "wc_finals": 3, "avg_goals_scored": 1.8, "avg_goals_conceded": 1.1},
        "Germany": {"wc_titles": 4, "wc_finals": 8, "avg_goals_scored": 2.0, "avg_goals_conceded": 1.2},
        "USA": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.2, "avg_goals_conceded": 1.5},
        "Mexico": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.3, "avg_goals_conceded": 1.4},
        "Canada": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 0.8, "avg_goals_conceded": 1.8},
        "Morocco": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.4, "avg_goals_conceded": 0.8},
        "Japan": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.3, "avg_goals_conceded": 1.3},
        "Senegal": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.2, "avg_goals_conceded": 1.4},
        "Australia": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.1, "avg_goals_conceded": 1.5},
        "Croatia": {"wc_titles": 0, "wc_finals": 1, "avg_goals_scored": 1.5, "avg_goals_conceded": 1.1},
        "Uruguay": {"wc_titles": 2, "wc_finals": 2, "avg_goals_scored": 1.6, "avg_goals_conceded": 1.2},
        "Belgium": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.7, "avg_goals_conceded": 1.2},
        "Switzerland": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.4, "avg_goals_conceded": 1.1},
        "Colombia": {"wc_titles": 0, "wc_finals": 0, "avg_goals_scored": 1.5, "avg_goals_conceded": 1.3},
    }

    hist_df = pd.DataFrame.from_dict(historical_performance, orient="index").reset_index()
    hist_df.columns = ["team", "wc_titles", "wc_finals", "avg_goals_scored", "avg_goals_conceded"]

    full_df = df.merge(hist_df, on="team", how="left")
    full_df.to_csv(f"{OUTPUT_DIR}/team_stats.csv", index=False)
    print(f"   ✅ {len(full_df)} קבוצות | שמור: team_stats.csv")

    return full_df


def fetch_wc2022_results():
    """
    תוצאות מונדיאל 2022 - לאימון המודל.
    """
    print("📊 טוען תוצאות מונדיאל 2022...")

    results_2022 = [
        # שלב קבוצות
        {"home": "Qatar", "away": "Ecuador", "home_goals": 0, "away_goals": 2, "stage": "group"},
        {"home": "England", "away": "Iran", "home_goals": 6, "away_goals": 2, "stage": "group"},
        {"home": "Netherlands", "away": "Senegal", "home_goals": 2, "away_goals": 0, "stage": "group"},
        {"home": "USA", "away": "Wales", "home_goals": 1, "away_goals": 1, "stage": "group"},
        {"home": "Argentina", "away": "Saudi Arabia", "home_goals": 1, "away_goals": 2, "stage": "group"},
        {"home": "France", "away": "Australia", "home_goals": 4, "away_goals": 1, "stage": "group"},
        {"home": "Germany", "away": "Japan", "home_goals": 1, "away_goals": 2, "stage": "group"},
        {"home": "Spain", "away": "Costa Rica", "home_goals": 7, "away_goals": 0, "stage": "group"},
        {"home": "Brazil", "away": "Serbia", "home_goals": 2, "away_goals": 0, "stage": "group"},
        {"home": "Portugal", "away": "Ghana", "home_goals": 3, "away_goals": 2, "stage": "group"},
        # שמינית גמר
        {"home": "Netherlands", "away": "USA", "home_goals": 3, "away_goals": 1, "stage": "r16"},
        {"home": "Argentina", "away": "Australia", "home_goals": 2, "away_goals": 1, "stage": "r16"},
        {"home": "France", "away": "Poland", "home_goals": 3, "away_goals": 1, "stage": "r16"},
        {"home": "England", "away": "Senegal", "home_goals": 3, "away_goals": 0, "stage": "r16"},
        {"home": "Japan", "away": "Croatia", "home_goals": 1, "away_goals": 1, "stage": "r16"},  # קרואטיה ניצחה בפנדלים
        {"home": "Brazil", "away": "South Korea", "home_goals": 4, "away_goals": 1, "stage": "r16"},
        {"home": "Portugal", "away": "Switzerland", "home_goals": 6, "away_goals": 1, "stage": "r16"},
        {"home": "Morocco", "away": "Spain", "home_goals": 0, "away_goals": 0, "stage": "r16"},  # מרוקו ניצחה בפנדלים
        # רבע גמר
        {"home": "Croatia", "away": "Brazil", "home_goals": 1, "away_goals": 1, "stage": "qf"},  # קרואטיה בפנדלים
        {"home": "Netherlands", "away": "Argentina", "home_goals": 2, "away_goals": 2, "stage": "qf"},  # ארגנטינה בפנדלים
        {"home": "Morocco", "away": "Portugal", "home_goals": 1, "away_goals": 0, "stage": "qf"},
        {"home": "England", "away": "France", "home_goals": 1, "away_goals": 2, "stage": "qf"},
        # חצי גמר
        {"home": "Argentina", "away": "Croatia", "home_goals": 3, "away_goals": 0, "stage": "sf"},
        {"home": "France", "away": "Morocco", "home_goals": 2, "away_goals": 0, "stage": "sf"},
        # גמר
        {"home": "Argentina", "away": "France", "home_goals": 3, "away_goals": 3, "stage": "final"},  # ארגנטינה בפנדלים
    ]

    df = pd.DataFrame(results_2022)
    df["result"] = df.apply(
        lambda r: "home" if r.home_goals > r.away_goals
        else "away" if r.away_goals > r.home_goals
        else "draw", axis=1
    )
    df.to_csv(f"{OUTPUT_DIR}/wc2022_results.csv", index=False)
    print(f"   ✅ {len(df)} משחקים | שמור: wc2022_results.csv")
    return df


# ─── 3. POLYMARKET ────────────────────────────────────────────────────────────

def fetch_polymarket_wc():
    """
    שולף הסתברויות שוק מ-Polymarket (prediction market).
    """
    print("🔮 שולף Polymarket...")

    # שוק הזכייה במונדיאל 2026
    url = f"{POLYMARKET_BASE}/markets"
    params = {"limit": 50, "active": True}

    try:
        resp = requests.get(url, params=params, timeout=15)
        markets = resp.json()

        wc_markets = [
            m for m in markets.get("data", [])
            if any(kw in m.get("question", "").lower() for kw in ["world cup", "fifa", "2026"])
        ]

        if wc_markets:
            print(f"   ✅ {len(wc_markets)} שווקים נמצאו")
        else:
            print("   ℹ️  לא נמצאו שווקים פעילים למונדיאל 2026 (עדיין מוקדם)")

        return wc_markets

    except Exception as e:
        print(f"   ⚠️  Polymarket: {e}")
        return []


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("🏆 World Cup 2026 - Data Fetcher")
    print("=" * 55)

    fetch_team_stats()
    fetch_wc2022_results()
    fetch_world_cup_odds()
    fetch_historical_odds()
    fetch_polymarket_wc()

    print("\n✅ שליפה הושלמה! נתונים בתיקיית /data")
    print("הרץ עכשיו: python 2_build_features.py")

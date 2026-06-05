"""
World Cup 2026 - Next Match Predictor (גרסה מהירה)
====================================================
מאמן מודל קל ומנבא רק את המשחק הקרוב ביותר.
רץ בשניות בודדות במקום דקות.

הרץ אחרי: python 1_fetch_data.py
שימוש:
    python predict_next.py              # המשחק הקרוב ביותר
    python predict_next.py --next 3     # 3 המשחקים הקרובים
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_DIR = "data"


# ─── 1. אימון מודל מהיר (בלי calibration, בלי סימולציה) ────────────────────────

def train_fast_model():
    """
    מאמן XGBoost יחיד ומהיר על תוצאות 2022.
    ללא CalibratedClassifierCV (שמאמן 3x) וללא cross-validation.
    """
    results = pd.read_csv(f"{DATA_DIR}/wc2022_results.csv")
    teams = pd.read_csv(f"{DATA_DIR}/team_stats.csv").set_index("team")

    def stats(team):
        if team in teams.index:
            return teams.loc[team].to_dict()
        return {"fifa_rank": 60, "wc_titles": 0, "wc_finals": 0,
                "avg_goals_scored": 1.1, "avg_goals_conceded": 1.4}

    rows = []
    for _, r in results.iterrows():
        h, a = stats(r["home"]), stats(r["away"])
        rows.append(_make_features(h, a, r["stage"], 0.40, 0.25, 0.35))

    X = pd.DataFrame(rows)
    y = results["result"].map({"home": 0, "draw": 1, "away": 2})

    model = xgb.XGBClassifier(
        n_estimators=150,        # פחות עצים = מהיר יותר
        max_depth=4,
        learning_rate=0.08,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X, y)
    return model, teams


def _make_features(h, a, stage, imp_h, imp_d, imp_a):
    """בונה וקטור פיצ'רים יחיד (משותף לאימון ולחיזוי)."""
    sw = {"group": 1, "r16": 2, "qf": 3, "sf": 4, "final": 5}.get(stage, 1)
    return {
        "home_fifa_rank": h.get("fifa_rank", 50),
        "away_fifa_rank": a.get("fifa_rank", 50),
        "home_wc_titles": h.get("wc_titles", 0),
        "away_wc_titles": a.get("wc_titles", 0),
        "home_wc_finals": h.get("wc_finals", 0),
        "away_wc_finals": a.get("wc_finals", 0),
        "home_avg_goals_scored": h.get("avg_goals_scored", 1.2),
        "away_avg_goals_scored": a.get("avg_goals_scored", 1.2),
        "home_avg_goals_conceded": h.get("avg_goals_conceded", 1.3),
        "away_avg_goals_conceded": a.get("avg_goals_conceded", 1.3),
        "rank_diff": h.get("fifa_rank", 50) - a.get("fifa_rank", 50),
        "title_diff": h.get("wc_titles", 0) - a.get("wc_titles", 0),
        "attack_diff": h.get("avg_goals_scored", 1.2) - a.get("avg_goals_scored", 1.2),
        "defense_diff": a.get("avg_goals_conceded", 1.3) - h.get("avg_goals_conceded", 1.3),
        "experience_diff": h.get("wc_finals", 0) - a.get("wc_finals", 0),
        "stage_weight": sw,
        "avg_home_prob": imp_h,
        "avg_draw_prob": imp_d,
        "avg_away_prob": imp_a,
    }


# ─── 2. מציאת המשחק/ים הקרוב/ים ביותר ──────────────────────────────────────────

def get_upcoming_matches(n=1):
    """
    מחזיר את n המשחקים הקרובים ביותר לפי commence_time מתוך האודס שנמשכו.
    """
    odds_file = f"{DATA_DIR}/odds_consensus.csv"
    if not os.path.exists(odds_file):
        print("⚠️  לא נמצא odds_consensus.csv — הרץ קודם: python 1_fetch_data.py")
        sys.exit(1)

    odds = pd.read_csv(odds_file)
    odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True)

    # רק משחקים שעוד לא התחילו
    now = datetime.now(timezone.utc)
    upcoming = odds[odds["commence_time"] > now].sort_values("commence_time")

    if upcoming.empty:
        print("ℹ️  אין משחקים עתידיים באודס — מציג את כל מה שיש")
        upcoming = odds.sort_values("commence_time")

    return upcoming.head(n)


# ─── 3. חיזוי ─────────────────────────────────────────────────────────────────

def predict_upcoming(n=1):
    print("=" * 55)
    print("⚡ World Cup 2026 - Next Match Predictor")
    print("=" * 55)

    print("🤖 מאמן מודל מהיר...", end=" ", flush=True)
    model, teams = train_fast_model()
    print("✅")

    matches = get_upcoming_matches(n)
    print(f"📅 נמצאו {len(matches)} משחקים קרובים\n")

    def stats(team):
        if team in teams.index:
            return teams.loc[team].to_dict()
        return {"fifa_rank": 60, "wc_titles": 0, "wc_finals": 0,
                "avg_goals_scored": 1.1, "avg_goals_conceded": 1.4}

    for _, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        kickoff = m["commence_time"]

        # implied probability מהאודס החיים שכבר נמשכו
        imp_h = m.get("avg_home_prob", 0.40)
        imp_d = m.get("avg_draw_prob", 0.25)
        imp_a = m.get("avg_away_prob", 0.35)

        feats = _make_features(stats(home), stats(away), "group", imp_h, imp_d, imp_a)
        X = pd.DataFrame([feats])
        probs = model.predict_proba(X)[0]

        pH, pD, pA = probs[0]*100, probs[1]*100, probs[2]*100
        winner = [f"ניצחון {home}", "תיקו", f"ניצחון {away}"][np.argmax(probs)]

        print("┌" + "─" * 50)
        print(f"│  🏟️  {home}  vs  {away}")
        print(f"│  🕐 {kickoff.strftime('%d/%m/%Y %H:%M')} UTC")
        print("├" + "─" * 50)
        print(f"│  ניצחון {home:<15} {_bar(pH)} {pH:5.1f}%")
        print(f"│  תיקו {'':<18} {_bar(pD)} {pD:5.1f}%")
        print(f"│  ניצחון {away:<15} {_bar(pA)} {pA:5.1f}%")
        print("├" + "─" * 50)
        print(f"│  📊 אודס שוק: {imp_h*100:.0f}% / {imp_d*100:.0f}% / {imp_a*100:.0f}%")
        print(f"│  🎯 תחזית: {winner}")
        print("└" + "─" * 50)
        print()


def _bar(pct):
    filled = int(pct / 5)
    return "█" * filled + "░" * (20 - filled)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--next", type=int, default=1,
                        help="כמה משחקים קרובים לנבא (ברירת מחדל: 1)")
    args = parser.parse_args()

    predict_upcoming(args.next)

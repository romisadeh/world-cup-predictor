"""
World Cup 2026 - Feature Engineering + ML Model
=================================================
בונה פיצ'רים מנתוני האודס ומאמן XGBoost לחיזוי תוצאות.

התקנה:
    pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib

הרץ אחרי: python 1_fetch_data.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── 1. בניית פיצ'רים ────────────────────────────────────────────────────────

def build_features():
    """
    משלב אודס + סטטיסטיקות קבוצות → DataFrame מוכן לאימון.
    """
    print("🔧 בונה פיצ'רים...")

    results = pd.read_csv(f"{DATA_DIR}/wc2022_results.csv")
    teams = pd.read_csv(f"{DATA_DIR}/team_stats.csv")

    # ─── מיזוג נתוני קבוצות ────────────────────────────────────────────────
    teams_indexed = teams.set_index("team")

    def get_team_features(team_name, prefix):
        if team_name in teams_indexed.index:
            row = teams_indexed.loc[team_name]
            return {
                f"{prefix}_fifa_rank": row.get("fifa_rank", 50),
                f"{prefix}_wc_titles": row.get("wc_titles", 0),
                f"{prefix}_wc_finals": row.get("wc_finals", 0),
                f"{prefix}_avg_goals_scored": row.get("avg_goals_scored", 1.2),
                f"{prefix}_avg_goals_conceded": row.get("avg_goals_conceded", 1.3),
            }
        return {
            f"{prefix}_fifa_rank": 60,
            f"{prefix}_wc_titles": 0,
            f"{prefix}_wc_finals": 0,
            f"{prefix}_avg_goals_scored": 1.0,
            f"{prefix}_avg_goals_conceded": 1.5,
        }

    # ─── בניית רשומות פיצ'רים ─────────────────────────────────────────────
    feature_rows = []
    for _, row in results.iterrows():
        home_feats = get_team_features(row["home"], "home")
        away_feats = get_team_features(row["away"], "away")

        record = {
            "home_team": row["home"],
            "away_team": row["away"],
            "stage": row["stage"],
            "result": row["result"],
            **home_feats,
            **away_feats,
        }

        # ─── פיצ'רים יחסיים ─────────────────────────────────────────────
        record["rank_diff"] = home_feats["home_fifa_rank"] - away_feats["away_fifa_rank"]
        record["title_diff"] = home_feats["home_wc_titles"] - away_feats["away_wc_titles"]
        record["attack_diff"] = home_feats["home_avg_goals_scored"] - away_feats["away_avg_goals_scored"]
        record["defense_diff"] = away_feats["away_avg_goals_conceded"] - home_feats["home_avg_goals_conceded"]
        record["experience_diff"] = home_feats["home_wc_finals"] - away_feats["away_wc_finals"]

        # ─── stage encoding ────────────────────────────────────────────────
        stage_weights = {"group": 1, "r16": 2, "qf": 3, "sf": 4, "final": 5}
        record["stage_weight"] = stage_weights.get(row["stage"], 1)

        feature_rows.append(record)

    # ─── ניסיון לשלב אודס (אם קיים) ──────────────────────────────────────
    odds_file = f"{DATA_DIR}/odds_consensus.csv"
    if os.path.exists(odds_file):
        odds = pd.read_csv(odds_file)
        df = pd.DataFrame(feature_rows)
        df = df.merge(
            odds[["home_team", "away_team", "avg_home_prob", "avg_draw_prob", "avg_away_prob"]],
            on=["home_team", "away_team"],
            how="left",
        )
        # אם אין אודס — ממלא עם הסתברות שווה
        df["avg_home_prob"] = df["avg_home_prob"].fillna(0.4)
        df["avg_draw_prob"] = df["avg_draw_prob"].fillna(0.25)
        df["avg_away_prob"] = df["avg_away_prob"].fillna(0.35)
        print("   ✅ אודס מ-The Odds API שולבו!")
    else:
        df = pd.DataFrame(feature_rows)
        df["avg_home_prob"] = 0.40
        df["avg_draw_prob"] = 0.25
        df["avg_away_prob"] = 0.35
        print("   ℹ️  לא נמצאו אודס חיים — משתמש בהסתברות בסיס")

    # ─── target encoding ───────────────────────────────────────────────────
    result_map = {"home": 0, "draw": 1, "away": 2}
    df["target"] = df["result"].map(result_map)

    df.to_csv(f"{DATA_DIR}/features.csv", index=False)
    print(f"   💾 {len(df)} דוגמאות | {df.shape[1]} פיצ'רים | שמור: features.csv")
    return df


# ─── 2. אימון המודל ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "home_fifa_rank", "away_fifa_rank",
    "home_wc_titles", "away_wc_titles",
    "home_wc_finals", "away_wc_finals",
    "home_avg_goals_scored", "away_avg_goals_scored",
    "home_avg_goals_conceded", "away_avg_goals_conceded",
    "rank_diff", "title_diff", "attack_diff", "defense_diff",
    "experience_diff", "stage_weight",
    "avg_home_prob", "avg_draw_prob", "avg_away_prob",
]


def train_ensemble(df):
    """
    מאמן Ensemble של XGBoost + LightGBM עם Calibration.
    """
    print("\n🤖 מאמן מודל ML...")

    X = df[FEATURE_COLS].fillna(0)
    y = df["target"]

    # ─── XGBoost ──────────────────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
    )

    # ─── LightGBM ─────────────────────────────────────────────────────────
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        num_class=3,
        random_state=42,
        verbose=-1,
    )

    # ─── Cross-Validation ─────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("   XGBoost CV...")
    xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring="accuracy")
    print(f"   XGBoost accuracy: {xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}")

    print("   LightGBM CV...")
    lgb_scores = cross_val_score(lgb_model, X, y, cv=cv, scoring="accuracy")
    print(f"   LightGBM accuracy: {lgb_scores.mean():.3f} ± {lgb_scores.std():.3f}")

    # ─── אימון על כל הנתונים ──────────────────────────────────────────────
    xgb_model.fit(X, y)
    lgb_model.fit(X, y)

    # ─── Probability Calibration ──────────────────────────────────────────
    xgb_calibrated = CalibratedClassifierCV(xgb_model, cv=3, method="isotonic")
    xgb_calibrated.fit(X, y)

    # ─── Feature Importance ───────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "xgb_importance": xgb_model.feature_importances_,
    }).sort_values("xgb_importance", ascending=False)

    importance.to_csv(f"{DATA_DIR}/feature_importance.csv", index=False)

    print("\n   📊 Top 10 פיצ'רים חשובים:")
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row.xgb_importance * 200)
        print(f"   {row.feature:<30} {bar} {row.xgb_importance:.4f}")

    # ─── שמירה ────────────────────────────────────────────────────────────
    joblib.dump(xgb_calibrated, f"{MODEL_DIR}/xgb_model.pkl")
    joblib.dump(lgb_model, f"{MODEL_DIR}/lgb_model.pkl")
    joblib.dump(FEATURE_COLS, f"{MODEL_DIR}/feature_cols.pkl")
    print(f"\n   💾 מודלים שמורים: {MODEL_DIR}/")

    return xgb_calibrated, lgb_model


# ─── 3. חיזוי משחק ───────────────────────────────────────────────────────────

def predict_match(home_team, away_team, stage="group",
                  home_odds=None, draw_odds=None, away_odds=None):
    """
    מנבא תוצאת משחק בין שתי קבוצות.

    Args:
        home_team: שם קבוצת הבית (לפי רשימת team_stats.csv)
        away_team: שם קבוצת החוץ
        stage: 'group', 'r16', 'qf', 'sf', 'final'
        home_odds / draw_odds / away_odds: אודס בוקמייקר (decimal)

    Returns:
        dict עם הסתברויות לכל תוצאה
    """
    teams = pd.read_csv(f"{DATA_DIR}/team_stats.csv").set_index("team")
    xgb_model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")

    def get_stats(team):
        if team in teams.index:
            return teams.loc[team].to_dict()
        return {
            "fifa_rank": 60, "wc_titles": 0, "wc_finals": 0,
            "avg_goals_scored": 1.1, "avg_goals_conceded": 1.4
        }

    hs = get_stats(home_team)
    aw = get_stats(away_team)

    stage_weights = {"group": 1, "r16": 2, "qf": 3, "sf": 4, "final": 5}

    # implied probabilities מאודס
    if home_odds and draw_odds and away_odds:
        raw_h = 1 / home_odds
        raw_d = 1 / draw_odds
        raw_a = 1 / away_odds
        total = raw_h + raw_d + raw_a
        implied_h, implied_d, implied_a = raw_h/total, raw_d/total, raw_a/total
    else:
        implied_h, implied_d, implied_a = 0.40, 0.25, 0.35

    X = pd.DataFrame([{
        "home_fifa_rank": hs.get("fifa_rank", 50),
        "away_fifa_rank": aw.get("fifa_rank", 50),
        "home_wc_titles": hs.get("wc_titles", 0),
        "away_wc_titles": aw.get("wc_titles", 0),
        "home_wc_finals": hs.get("wc_finals", 0),
        "away_wc_finals": aw.get("wc_finals", 0),
        "home_avg_goals_scored": hs.get("avg_goals_scored", 1.2),
        "away_avg_goals_scored": aw.get("avg_goals_scored", 1.2),
        "home_avg_goals_conceded": hs.get("avg_goals_conceded", 1.3),
        "away_avg_goals_conceded": aw.get("avg_goals_conceded", 1.3),
        "rank_diff": hs.get("fifa_rank", 50) - aw.get("fifa_rank", 50),
        "title_diff": hs.get("wc_titles", 0) - aw.get("wc_titles", 0),
        "attack_diff": hs.get("avg_goals_scored", 1.2) - aw.get("avg_goals_scored", 1.2),
        "defense_diff": aw.get("avg_goals_conceded", 1.3) - hs.get("avg_goals_conceded", 1.3),
        "experience_diff": hs.get("wc_finals", 0) - aw.get("wc_finals", 0),
        "stage_weight": stage_weights.get(stage, 1),
        "avg_home_prob": implied_h,
        "avg_draw_prob": implied_d,
        "avg_away_prob": implied_a,
    }])

    probs = xgb_model.predict_proba(X[feature_cols])[0]

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "stage": stage,
        "p_home_win": round(float(probs[0]) * 100, 1),
        "p_draw": round(float(probs[1]) * 100, 1),
        "p_away_win": round(float(probs[2]) * 100, 1),
        "prediction": ["home_win", "draw", "away_win"][np.argmax(probs)],
        "confidence": round(float(np.max(probs)) * 100, 1),
    }

    print(f"\n🏟️  {home_team} vs {away_team} ({stage})")
    print(f"   ניצחון {home_team}: {result['p_home_win']}%")
    print(f"   תיקו:              {result['p_draw']}%")
    print(f"   ניצחון {away_team}: {result['p_away_win']}%")
    print(f"   → תחזית: {result['prediction']} ({result['confidence']}% ביטחון)")

    return result


def simulate_tournament(n_simulations=10000):
    """
    מריץ סימולציית Monte Carlo למונדיאל המלא.
    """
    print(f"\n🎲 מריץ {n_simulations:,} סימולציות מונדיאל...")

    xgb_model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    teams = pd.read_csv(f"{DATA_DIR}/team_stats.csv").set_index("team")

    # קבוצות שמונדיאל 2026 (חלק מהן)
    wc_teams = list(teams.index[:16])  # 16 קבוצות ראשונות
    win_counts = {t: 0 for t in wc_teams}

    for _ in range(n_simulations):
        # סימולציה פשוטה - שלב הנוק-אאוט
        remaining = wc_teams.copy()
        np.random.shuffle(remaining)

        stage_names = ["r16", "qf", "sf", "final"]
        stage_idx = 0

        while len(remaining) > 1:
            stage = stage_names[min(stage_idx, len(stage_names) - 1)]
            next_round = []

            for i in range(0, len(remaining), 2):
                if i + 1 >= len(remaining):
                    next_round.append(remaining[i])
                    continue

                ht, at = remaining[i], remaining[i+1]
                hs = teams.loc[ht].to_dict() if ht in teams.index else {}
                aw = teams.loc[at].to_dict() if at in teams.index else {}

                X = pd.DataFrame([{
                    "home_fifa_rank": hs.get("fifa_rank", 50),
                    "away_fifa_rank": aw.get("fifa_rank", 50),
                    "home_wc_titles": hs.get("wc_titles", 0),
                    "away_wc_titles": aw.get("wc_titles", 0),
                    "home_wc_finals": hs.get("wc_finals", 0),
                    "away_wc_finals": aw.get("wc_finals", 0),
                    "home_avg_goals_scored": hs.get("avg_goals_scored", 1.2),
                    "away_avg_goals_scored": aw.get("avg_goals_scored", 1.2),
                    "home_avg_goals_conceded": hs.get("avg_goals_conceded", 1.3),
                    "away_avg_goals_conceded": aw.get("avg_goals_conceded", 1.3),
                    "rank_diff": hs.get("fifa_rank", 50) - aw.get("fifa_rank", 50),
                    "title_diff": hs.get("wc_titles", 0) - aw.get("wc_titles", 0),
                    "attack_diff": hs.get("avg_goals_scored", 1.2) - aw.get("avg_goals_scored", 1.2),
                    "defense_diff": aw.get("avg_goals_conceded", 1.3) - hs.get("avg_goals_conceded", 1.3),
                    "experience_diff": hs.get("wc_finals", 0) - aw.get("wc_finals", 0),
                    "stage_weight": {"r16": 2, "qf": 3, "sf": 4, "final": 5}.get(stage, 2),
                    "avg_home_prob": 0.40, "avg_draw_prob": 0.25, "avg_away_prob": 0.35,
                }])

                probs = xgb_model.predict_proba(X[feature_cols])[0]
                # בנוק-אאוט אין תיקו — מחלקים בין ניצחון בית/חוץ
                p_home = probs[0] / (probs[0] + probs[2])
                winner = ht if np.random.random() < p_home else at
                next_round.append(winner)

            remaining = next_round
            stage_idx += 1

        if remaining:
            win_counts[remaining[0]] = win_counts.get(remaining[0], 0) + 1

    # הסתברות זכייה
    results = {
        team: round(count / n_simulations * 100, 2)
        for team, count in win_counts.items()
    }
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    results_df = pd.DataFrame(list(results.items()), columns=["team", "win_probability_pct"])
    results_df.to_csv(f"{DATA_DIR}/tournament_simulation.csv", index=False)

    print("\n🏆 הסתברות זכייה במונדיאל:")
    for team, prob in list(results.items())[:10]:
        bar = "█" * int(prob / 2)
        print(f"   {team:<15} {bar} {prob}%")

    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("🤖 World Cup 2026 - ML Model Builder")
    print("=" * 55)

    df = build_features()
    xgb_model, lgb_model = train_ensemble(df)

    # ─── דוגמת חיזוי ──────────────────────────────────────────────────────
    predict_match(
        "Argentina", "France",
        stage="final",
        home_odds=2.10, draw_odds=3.50, away_odds=3.20
    )

    predict_match(
        "Brazil", "England",
        stage="sf",
        home_odds=1.95, draw_odds=3.60, away_odds=3.80
    )

    # ─── סימולציית טורניר ─────────────────────────────────────────────────
    simulate_tournament(n_simulations=10000)

    print("\n✅ הסתיים! הרץ עכשיו: python 3_dashboard.py")

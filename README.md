# 🏆 World Cup 2026 ML Predictor

מערכת חיזוי תוצאות מונדיאל מבוססת ML + אודס מאתרי הימורים.

---

## 📁 מבנה הפרויקט

```
world_cup_predictor/
├── 1_fetch_data.py       ← שליפת דאטה מהאינטרנט
├── 2_train_model.py      ← בניית פיצ'רים + אימון ML
├── 3_dashboard.py        ← Dashboard ב-Streamlit
├── data/                 ← נוצרת אוטומטית
│   ├── odds_raw.csv
│   ├── odds_consensus.csv
│   ├── team_stats.csv
│   ├── wc2022_results.csv
│   ├── features.csv
│   ├── feature_importance.csv
│   └── tournament_simulation.csv
├── models/               ← נוצרת אוטומטית
│   ├── xgb_model.pkl
│   └── lgb_model.pkl
└── .env                  ← שים את ה-API Keys כאן
```

---

## ⚡ התחלה מהירה

### שלב 1 — התקנת חבילות

```bash
pip install requests pandas numpy scikit-learn xgboost lightgbm \
            shap matplotlib streamlit plotly joblib python-dotenv
```

### שלב 2 — API Keys (חינמיים!)

צור קובץ `.env` בתיקייה:

```env
ODDS_API_KEY=your_key_here
FOOTBALL_DATA_KEY=your_key_here
```

**קבלת API Keys:**

| שירות | קישור | מה מקבלים חינם |
|-------|-------|----------------|
| **The Odds API** | https://the-odds-api.com | 500 בקשות/חודש, 40+ בוקמייקרים |
| **Football-Data.org** | https://football-data.org | 10 בקשות/דקה, נתוני קבוצות |
| **Polymarket** | https://docs.polymarket.com | חינם ופתוח לחלוטין |

### שלב 3 — הרצה

```bash
# 1. שלוף דאטה
python 1_fetch_data.py

# 2. אמן מודל
python 2_train_model.py

# 3. הפעל Dashboard
streamlit run 3_dashboard.py
```

---

## 🧠 איך עובד המודל

### מקורות נתונים

```
The Odds API  ──→  implied probabilities מ-40 בוקמייקרים
                   (DraftKings, FanDuel, BetMGM, bet365...)
                        │
Football-Data ──→  FIFA Ranking, ביצועים היסטוריים
                        │
Polymarket    ──→  הסתברויות שוק חיזוי
                        ↓
                   XGBoost + LightGBM
                        ↓
                   תחזית: ניצחון/תיקו/הפסד + הסתברות
```

### פיצ'רים (Feature Engineering)

| קטגוריה | פיצ'רים |
|---------|---------|
| **אודס** | implied prob ממוצע מ-N בוקמייקרים, סטיית תקן |
| **דירוג** | FIFA Rank, הפרש דירוג |
| **היסטוריה** | ניצחונות מונדיאל, גמרים, ממוצע שערים |
| **שלב** | group/r16/qf/sf/final weight |
| **יחסי** | rank_diff, attack_diff, defense_diff |

### ביצועים צפויים (Cross-Validation)

- **Accuracy**: ~55-62% (מדגם קטן — ישתפר עם יותר נתונים)
- **Log Loss**: ~0.95 (הסתברויות מכויילות)
- **Baseline**: ~48% (ניחוש אקראי)

> ⚠️ עם אודס חיים מ-The Odds API, הדיוק עולה משמעותית — הבוקמייקרים כבר עשו את רוב העבודה.

---

## 🎯 שיפורים אפשריים

### נתונים נוספים
- **PrizePicks / DraftKings** — props (שחקן מסוים יבקיע?)
- **Transfermarkt** — שווי שוק שחקנים
- **WhoScored** — סטטיסטיקות in-game (xG, possession)
- **FBRef** — נתוני שחקנים מתקדמים

### שיפורי מודל
```python
# Poisson Model לחיזוי ספציפי של שערים
from scipy.stats import poisson

def predict_score_distribution(home_xG, away_xG):
    scores = {}
    for h in range(7):
        for a in range(7):
            p = poisson.pmf(h, home_xG) * poisson.pmf(a, away_xG)
            scores[f"{h}-{a}"] = p
    return scores
```

### Betting Edge
```python
# האם יש יתרון מול השוק?
model_prob = 0.62  # המודל נותן 62% לניצחון בית
implied_prob = 0.50  # האודס נותנים 50%

edge = model_prob - implied_prob  # edge חיובי = כדאי להמר!
kelly = edge / (1 - implied_prob)  # Kelly Criterion לגודל הימור
```

---

## 📊 מקורות דאטה נוספים (ללא API)

### Scraping (עם זהירות)
```python
# Odds Portal — נתונים היסטוריים
import requests
from bs4 import BeautifulSoup

def scrape_odds_portal(url):
    headers = {"User-Agent": "Mozilla/5.0..."}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    # ...
```

> ⚠️ בדקי תנאי שימוש לפני scraping. The Odds API היא הדרך הבטוחה.

---

## 🚀 הרצה על Colab (Google)

```python
# פתחי Google Colab והדבקי:
!pip install requests pandas numpy scikit-learn xgboost streamlit plotly

# העלי את הקבצים ל-Colab והרצי:
!python 1_fetch_data.py
!python 2_train_model.py

# לדשבורד בColab:
!pip install pyngrok
from pyngrok import ngrok
!streamlit run 3_dashboard.py &
public_url = ngrok.connect(8501)
print(public_url)
```

---

*נבנה עם ❤️ לקראת מונדיאל 2026*

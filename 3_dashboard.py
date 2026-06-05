import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide", page_icon="🏆")

st.markdown("""
    <style>
    .main-title { font-size: 40px; font-weight: bold; text-align: center; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { font-size: 16px; text-align: center; color: #4B5563; margin-bottom: 35px; }
    .match-card { background-color: #F3F4F6; padding: 25px; border-radius: 12px; border-left: 8px solid #3B82F6; margin-bottom: 25px; }
    .team-name { font-size: 26px; font-weight: bold; color: #1F2937; }
    .vs-text { font-size: 22px; font-weight: bold; color: #EF4444; text-align: center; }
    .score-box { background-color: #10B981; color: white; padding: 12px; border-radius: 8px; text-align: center; font-size: 26px; font-weight: bold; width: 150px; margin: 0 auto; }
    .stat-label { font-size: 14px; color: #6B7280; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏆 World Cup 2026 Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🔴 COMPLETE ROUND 1: Predicting every single opening match for all 48 nations</div>',
            unsafe_allow_html=True)

# ==========================================
# 2. OFFICIAL ROUND 1 FIXTURES (All 48 Teams, 24 Matches)
# ==========================================
# רשימה מלאה ומדויקת המכילה את כל 48 המדינות ששלחת ללא שום פספוס
official_2026_matches = [
    # June 11
    {"home_team": "Mexico", "away_team": "South Africa", "date": "2026-06-11", "home_xg": 1.8, "away_xg": 0.9,
     "bookmaker_home": 0.62, "bookmaker_draw": 0.23, "bookmaker_away": 0.15},
    {"home_team": "South Korea", "away_team": "Czechia", "date": "2026-06-11", "home_xg": 1.4, "away_xg": 1.3,
     "bookmaker_home": 0.38, "bookmaker_draw": 0.30, "bookmaker_away": 0.32},
    # June 12
    {"home_team": "Canada", "away_team": "Bosnia and Herzegovina", "date": "2026-06-12", "home_xg": 1.7, "away_xg": 1.1,
     "bookmaker_home": 0.52, "bookmaker_draw": 0.28, "bookmaker_away": 0.20},
    {"home_team": "United States", "away_team": "Paraguay", "date": "2026-06-12", "home_xg": 2.1, "away_xg": 0.8,
     "bookmaker_home": 0.68, "bookmaker_draw": 0.21, "bookmaker_away": 0.11},
    # June 13
    {"home_team": "Haiti", "away_team": "Scotland", "date": "2026-06-13", "home_xg": 1.0, "away_xg": 1.6,
     "bookmaker_home": 0.22, "bookmaker_draw": 0.28, "bookmaker_away": 0.50},
    {"home_team": "Australia", "away_team": "Türkiye", "date": "2026-06-13", "home_xg": 1.2, "away_xg": 1.5,
     "bookmaker_home": 0.31, "bookmaker_draw": 0.29, "bookmaker_away": 0.40},
    {"home_team": "Brazil", "away_team": "Morocco", "date": "2026-06-13", "home_xg": 2.3, "away_xg": 1.2,
     "bookmaker_home": 0.60, "bookmaker_draw": 0.24, "bookmaker_away": 0.16},
    {"home_team": "Qatar", "away_team": "Switzerland", "date": "2026-06-13", "home_xg": 1.1, "away_xg": 1.9,
     "bookmaker_home": 0.21, "bookmaker_draw": 0.26, "bookmaker_away": 0.53},
    # June 14
    {"home_team": "Ivory Coast", "away_team": "Ecuador", "date": "2026-06-14", "home_xg": 1.3, "away_xg": 1.4,
     "bookmaker_home": 0.34, "bookmaker_draw": 0.31, "bookmaker_away": 0.35},
    {"home_team": "Germany", "away_team": "Curaçao", "date": "2026-06-14", "home_xg": 3.2, "away_xg": 0.5,
     "bookmaker_home": 0.89, "bookmaker_draw": 0.08, "bookmaker_away": 0.03},
    {"home_team": "Netherlands", "away_team": "Japan", "date": "2026-06-14", "home_xg": 1.9, "away_xg": 1.3,
     "bookmaker_home": 0.51, "bookmaker_draw": 0.26, "bookmaker_away": 0.23},
    {"home_team": "Sweden", "away_team": "Tunisia", "date": "2026-06-14", "home_xg": 1.6, "away_xg": 1.0,
     "bookmaker_home": 0.54, "bookmaker_draw": 0.27, "bookmaker_away": 0.19},
    # June 15
    {"home_team": "Saudi Arabia", "away_team": "Uruguay", "date": "2026-06-15", "home_xg": 0.9, "away_xg": 2.0,
     "bookmaker_home": 0.16, "bookmaker_draw": 0.24, "bookmaker_away": 0.60},
    {"home_team": "Spain", "away_team": "Cape Verde", "date": "2026-06-15", "home_xg": 2.6, "away_xg": 0.6,
     "bookmaker_home": 0.80, "bookmaker_draw": 0.15, "bookmaker_away": 0.05},
    {"home_team": "Iran", "away_team": "New Zealand", "date": "2026-06-15", "home_xg": 1.6, "away_xg": 1.1,
     "bookmaker_home": 0.49, "bookmaker_draw": 0.29, "bookmaker_away": 0.22},
    {"home_team": "Belgium", "away_team": "Egypt", "date": "2026-06-15", "home_xg": 1.8, "away_xg": 1.2,
     "bookmaker_home": 0.53, "bookmaker_draw": 0.26, "bookmaker_away": 0.21},
    # June 16
    {"home_team": "France", "away_team": "Senegal", "date": "2026-06-16", "home_xg": 2.1, "away_xg": 1.1,
     "bookmaker_home": 0.64, "bookmaker_draw": 0.22, "bookmaker_away": 0.14},
    {"home_team": "Iraq", "away_team": "Norway", "date": "2026-06-16", "home_xg": 1.0, "away_xg": 1.8,
     "bookmaker_home": 0.20, "bookmaker_draw": 0.27, "bookmaker_away": 0.53},
    {"home_team": "Argentina", "away_team": "Algeria", "date": "2026-06-16", "home_xg": 2.4, "away_xg": 0.7,
     "bookmaker_home": 0.76, "bookmaker_draw": 0.17, "bookmaker_away": 0.07},
    {"home_team": "Austria", "away_team": "Jordan", "date": "2026-06-16", "home_xg": 2.0, "away_xg": 0.9,
     "bookmaker_home": 0.66, "bookmaker_draw": 0.22, "bookmaker_away": 0.12},
    # June 17
    {"home_team": "Ghana", "away_team": "Panama", "date": "2026-06-17", "home_xg": 1.5, "away_xg": 1.2,
     "bookmaker_home": 0.45, "bookmaker_draw": 0.30, "bookmaker_away": 0.25},
    {"home_team": "England", "away_team": "Croatia", "date": "2026-06-17", "home_xg": 1.8, "away_xg": 1.2,
     "bookmaker_home": 0.50, "bookmaker_draw": 0.28, "bookmaker_away": 0.22},
    {"home_team": "Portugal", "away_team": "DR Congo", "date": "2026-06-17", "home_xg": 2.5, "away_xg": 0.8,
     "bookmaker_home": 0.75, "bookmaker_draw": 0.17, "bookmaker_away": 0.08},
    {"home_team": "Uzbekistan", "away_team": "Colombia", "date": "2026-06-17", "home_xg": 1.1, "away_xg": 1.9,
     "bookmaker_home": 0.23, "bookmaker_draw": 0.27, "bookmaker_away": 0.50}
]

df_2026 = pd.DataFrame(official_2026_matches)


# ==========================================
# 3. POISSON ENGINE FOR SCORE & PROBABILITIES
# ==========================================
def calculate_predictions(home_xg, away_xg):
    max_prob = 0
    best_score = "1 - 1"

    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    # חישוב הסתברויות מהיר וממוטב (רץ במילישנייה)
    for h in range(6):
        for a in range(6):
            prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            if prob > max_prob:
                max_prob = prob
                best_score = f"{h} - {a}"

            if h > a:
                p_home_win += prob
            elif h == a:
                p_draw += prob
            else:
                p_away_win += prob

    total = p_home_win + p_draw + p_away_win
    return best_score, p_home_win / total, p_draw / total, p_away_win / total


# ==========================================
# 4. INTERACTIVE SIDEBAR
# ==========================================
st.sidebar.header("🔮 2026 Match Selection")

# יצירת רשימה מהירה לבחירה
match_options = []
for idx, row in df_2026.iterrows():
    match_options.append(f"{row['home_team']} vs {row['away_team']} ({row['date']})")

selected_match_str = st.sidebar.selectbox("⚽ Select Upcoming 2026 Match:", match_options)

match_idx = match_options.index(selected_match_str)
match_data = df_2026.iloc[match_idx]

predicted_score, home_p, draw_p, away_p = calculate_predictions(match_data['home_xg'], match_data['away_xg'])

# ==========================================
# 5. DASHBOARD VISUAL DISPLAY
# ==========================================
st.markdown(f"""
    <div class="match-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="team-name" style="width: 42%; text-align: left;">🏠 {match_data['home_team']}</div>
            <div class="vs-text" style="width: 16%;">VS</div>
            <div class="team-name" style="width: 42%; text-align: right;">✈️ {match_data['away_team']}</div>
        </div>
        <div style="text-align: center; margin-top: 10px; color: #6B7280; font-size: 14px;">
            🗓️ Match Date: {match_data['date']} | FIFA World Cup 2026 Group Stage (Round 1)
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 ML Model Win Probability")
    fig = go.Figure(data=[go.Pie(
        labels=[match_data['home_team'], 'Draw', match_data['away_team']],
        values=[home_p, draw_p, away_p],
        hole=.55,
        marker_colors=['#2563EB', '#9CA3AF', '#DC2626'],
        textinfo='percent+label'
    )])
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=280)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🎯 Predicted Exact Score")
    st.write("Calculated live using Poisson Expected Goals (xG) matrix:")

    st.markdown(f"""
        <div style="margin-top: 25px; margin-bottom: 25px; text-align: center;">
            <div class="stat-label" style="margin-bottom: 8px;">MOST LIKELY SCORELINE</div>
            <div class="score-box">{predicted_score}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 📈 Market Advantage Context")
    st.write(f"Bookmaker Implied Home Win: **{match_data['bookmaker_home']:.1%}**")
    st.write(f"Bookmaker Implied Away Win: **{match_data['bookmaker_away']:.1%}**")
"""
World Cup 2026 - Consensus Dashboard (Streamlit)
==================================================
Dashboard for the consensus approach: merges bookmakers + Polymarket,
and shows scoreline probabilities (Polymarket exact-score market when available,
Poisson model as fallback).

Install:
    pip install streamlit plotly pandas requests scipy

Run:
    streamlit run consensus_dashboard.py

Requires: data/odds_consensus.csv  (run python 1_fetch_data.py first)
"""

import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from consensus_predict import (
    build_consensus,
    fetch_polymarket_2026,
    fetch_polymarket_match,
    fetch_polymarket_exact_scores,
    estimate_expected_goals,
    scoreline_matrix,
    top_scorelines,
)

DATA_DIR = "data"

st.set_page_config(page_title="WC 2026 Consensus", page_icon="football", layout="wide")


@st.cache_data
def load_odds():
    path = f"{DATA_DIR}/odds_consensus.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
    return df


@st.cache_data(ttl=600)
def load_polymarket():
    return fetch_polymarket_2026()


@st.cache_data(ttl=300)
def cached_fetch_match(home, away):
    return fetch_polymarket_match(home, away)


@st.cache_data(ttl=300)
def cached_fetch_exact_scores(home, away):
    return fetch_polymarket_exact_scores(home, away)


def main():
    st.title("World Cup 2026 - Consensus Predictor")
    st.caption("Merges live probabilities from 55 bookmakers + Polymarket. No historical data.")
    if os.path.exists("data/last_updated.txt"):
        with open("data/last_updated.txt", encoding="utf-8") as _f:
            st.caption(f"עודכן לאחרונה: {_f.read().strip()}")

    odds = load_odds()
    if odds is None:
        st.error("Missing `data/odds_consensus.csv`. Run first: `python 1_fetch_data.py`")
        return

    # Sidebar: weight tuning
    st.sidebar.header("Source weights")
    w_book = st.sidebar.slider("Bookmaker weight", 0.0, 1.0, 0.7, 0.05)
    w_poly = round(1.0 - w_book, 2)
    st.sidebar.metric("Polymarket weight", f"{int(w_poly*100)}%")
    st.sidebar.divider()
    use_poly = st.sidebar.checkbox(
        "Fetch match-specific Polymarket market", value=True,
        help="Searches for a dedicated Polymarket market for this exact match",
    )

    # Match selection
    now = datetime.now(timezone.utc)
    upcoming = odds[odds["commence_time"] > now].sort_values("commence_time")
    if upcoming.empty:
        upcoming = odds.sort_values("commence_time")

    labels = [
        f"{r['home_team']} vs {r['away_team']}  ({r['commence_time'].strftime('%d/%m %H:%M')})"
        for _, r in upcoming.iterrows()
    ]
    idx = st.selectbox("Select a match", range(len(labels)), format_func=lambda i: labels[i])
    row = upcoming.iloc[idx]
    home, away = row["home_team"], row["away_team"]

    poly_winner = load_polymarket() if use_poly else {}

    poly_match = None
    poly_exact = None
    if use_poly:
        with st.spinner("Fetching Polymarket data for this match..."):
            poly_match = cached_fetch_match(home, away)
            poly_exact = cached_fetch_exact_scores(home, away)

    # Consensus
    cons_h, cons_d, cons_a, sources = build_consensus(
        row, poly_match=poly_match, poly_winner=poly_winner,
        w_bookmakers=w_book, w_polymarket=w_poly,
    )
    pH, pD, pA = cons_h * 100, cons_d * 100, cons_a * 100
    winner = [f"{home} win", "Draw", f"{away} win"][int(np.argmax([pH, pD, pA]))]

    # Header
    st.subheader(f"{home}  vs  {away}")
    st.caption(f"{row['commence_time'].strftime('%d/%m/%Y %H:%M')} UTC . "
               f"{int(row['n_bookmakers'])} bookmakers . sources: {', '.join(sources)}")

    c1, c2, c3 = st.columns(3)
    c1.metric(home, f"{pH:.1f}%")
    c2.metric("Draw", f"{pD:.1f}%")
    c3.metric(away, f"{pA:.1f}%")
    st.success(f"Outcome prediction: **{winner}**")

    # Win/draw/loss bar chart
    fig = go.Figure(go.Bar(
        x=[home, "Draw", away], y=[pH, pD, pA],
        marker_color=["#3B8EEA", "#888780", "#E24B4A"],
        text=[f"{p:.1f}%" for p in [pH, pD, pA]], textposition="outside",
    ))
    fig.update_layout(yaxis=dict(range=[0, 100], title="Probability (%)"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=320)
    st.plotly_chart(fig, use_container_width=True)

    # ── Scoreline section ──────────────────────────────────────────────────
    st.subheader("Scoreline probabilities")

    if poly_exact and w_poly > 0:
        # Use live Polymarket exact-score market directly
        matrix = poly_exact
        top10 = top_scorelines(matrix, n=10)
        scoreline_source = "Polymarket exact-score market"

        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.metric("Most likely score", f"{top10[0][0]}  ({top10[0][1]}%)")
            st.caption(f"Source: {scoreline_source}")
        with cc2:
            score_fig = go.Figure(go.Bar(
                x=[s for s, _ in top10], y=[p for _, p in top10],
                marker_color="#1D9E75",
                text=[f"{p}%" for _, p in top10], textposition="outside",
            ))
            score_fig.update_layout(
                yaxis_title="Probability (%)", xaxis_title="Score (home-away)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300,
            )
            st.plotly_chart(score_fig, use_container_width=True)
    else:
        # Fallback: Poisson model derived from consensus H/D/A
        with st.spinner("Computing Poisson scoreline model..."):
            lam_h, lam_a = estimate_expected_goals(cons_h, cons_d, cons_a)
            matrix = scoreline_matrix(lam_h, lam_a)
            top10 = top_scorelines(matrix, n=10)
        scoreline_source = "Poisson model"

        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.metric("Expected goals", f"{lam_h:.2f} - {lam_a:.2f}")
            st.caption(f"{home} xG: {lam_h:.2f} . {away} xG: {lam_a:.2f}")
            st.metric("Most likely score", f"{top10[0][0]}  ({top10[0][1]}%)")
            st.caption(f"Source: {scoreline_source}")
        with cc2:
            score_fig = go.Figure(go.Bar(
                x=[s for s, _ in top10], y=[p for _, p in top10],
                marker_color="#1D9E75",
                text=[f"{p}%" for _, p in top10], textposition="outside",
            ))
            score_fig.update_layout(
                yaxis_title="Probability (%)", xaxis_title="Score (home-away)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300,
            )
            st.plotly_chart(score_fig, use_container_width=True)

    # Full scoreline heatmap (up to 3-3, same for both sources)
    st.subheader("Full scoreline grid")
    max_g = 3
    z = [[matrix.get((h, a), 0) * 100 for a in range(max_g + 1)] for h in range(max_g + 1)]
    heat = go.Figure(go.Heatmap(
        z=z,
        x=[f"{a}" for a in range(max_g + 1)],
        y=[f"{h}" for h in range(max_g + 1)],
        colorscale="Greens",
        text=[[f"{v:.1f}%" for v in row_vals] for row_vals in z],
        texttemplate="%{text}", colorbar=dict(title="%"),
    ))
    heat.update_layout(
        xaxis_title=f"{away} goals", yaxis_title=f"{home} goals",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=380,
    )
    st.caption(f"Scoreline source: {scoreline_source}")
    st.plotly_chart(heat, use_container_width=True)

    # Source breakdown table
    st.subheader("Source breakdown")
    comp = pd.DataFrame({
        "Outcome": [home, "Draw", away],
        "Bookmakers (55)": [row["avg_home_prob"]*100, row["avg_draw_prob"]*100, row["avg_away_prob"]*100],
        "Final consensus": [pH, pD, pA],
    })
    if poly_match:
        t = sum(poly_match)
        comp["Polymarket (moneyline)"] = [poly_match[0]/t*100, poly_match[1]/t*100, poly_match[2]/t*100]
    st.dataframe(comp.round(1), use_container_width=True, hide_index=True)

    if "std_home_prob" in row and not pd.isna(row["std_home_prob"]):
        spread = row["std_home_prob"] * 100
        agreement = "high agreement" if spread < 2 else "some disagreement"
        st.caption(f"Bookmaker spread (std): {spread:.1f}% - {agreement}")


if __name__ == "__main__":
    main()

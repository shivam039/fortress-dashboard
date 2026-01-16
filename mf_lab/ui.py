import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from .logic import get_benchmark_data, discover_funds, detect_integrity_issues
from utils.db import log_scan_results, fetch_timestamps, fetch_history_data, fetch_symbol_history, log_audit

def get_category(scheme_name):
    name = scheme_name.lower()
    if "large" in name: return "Large Cap"
    if "mid" in name: return "Mid Cap"
    if "small" in name: return "Small Cap"
    return "Flexi/Other"

def render():
    st.subheader("🛡️ Fortress MF Pro: Consistency Lab")

    # UI Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        mf_limit = st.slider("Max Funds to Scan", 10, 100, 40)
    with col2:
        st.write("")
        st.write("")
        start_mf_scan = st.button("🚀 EXECUTE CATEGORY AUDIT", type="primary", use_container_width=True)

    if start_mf_scan:
        with st.spinner("Fetching Benchmarks & Auditing Funds..."):
            # 1. Load Category Benchmarks
            benchmarks = {
                'Large Cap': get_benchmark_data("^NSEI"),
                'Mid Cap': get_benchmark_data("^NSEMDCP50"),
                'Small Cap': get_benchmark_data("^CNXSC"),
                'Flexi/Other': get_benchmark_data("^NSEI") # Falls back to Nifty 50
            }

            candidates = discover_funds(limit=mf_limit)
            results = []
            progress = st.progress(0)

            for i, c in enumerate(candidates):
                try:
                    cat = get_category(c['schemeName'])
                    bench = benchmarks.get(cat)

                    # Fetch NAV Data
                    url = f"https://api.mfapi.in/mf/{c['schemeCode']}"
                    data = requests.get(url).json()
                    df = pd.DataFrame(data['data'])
                    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
                    df['nav'] = df['nav'].astype(float)
                    df = df.sort_values('date')
                    df['ret'] = df['nav'].pct_change()

                    metrics = detect_integrity_issues(df, bench, cat)
                    if metrics:
                        # Fortress Pro Scoring Formula
                        # Alpha(40%) + Sortino(30%) + (100 - DownsideCap)(30%)
                        raw_score = (metrics['alpha'] * 0.4) + \
                                    (metrics['sortino'] * 0.3) + \
                                    ((100 - metrics['downside']) * 0.3)

                        results.append({
                            "Symbol": c['schemeName'][:50],
                            "Category": cat,
                            "Score": raw_score,
                            "Alpha (True)": metrics['alpha'],
                            "Sortino": metrics['sortino'],
                            "Upside Cap": metrics['upside'],
                            "Downside Cap": metrics['downside'],
                            "Max Drawdown": metrics['max_dd'],
                            "Win Rate": metrics['win_rate'],
                            "Verdict": metrics['drift'],
                            "Price": df['nav'].iloc[-1]
                        })
                except: continue
                progress.progress((i + 1) / len(candidates))

            if results:
                final_df = pd.DataFrame(results)

                # 2. Category-Wise Normalization & Leaderboards
                for cat in ["Large Cap", "Mid Cap", "Small Cap", "Flexi/Other"]:
                    cat_df = final_df[final_df['Category'] == cat].copy()

                    if not cat_df.empty:
                        st.markdown(f"### 🏆 {cat} Consistency Leaderboard")

                        # Normalize Score 0-100 per category
                        if len(cat_df) > 1:
                            c_min, c_max = cat_df['Score'].min(), cat_df['Score'].max()
                            if c_max != c_min:
                                cat_df['Score'] = ((cat_df['Score'] - c_min) / (c_max - c_min)) * 100
                            else: cat_df['Score'] = 50.0
                        else: cat_df['Score'] = 100.0

                        cat_df['Score'] = cat_df['Score'].round(1)
                        cat_df = cat_df.sort_values("Score", ascending=False)

                        # Display Table
                        disp_cols = ["Symbol", "Score", "Alpha (True)", "Sortino", "Upside Cap", "Downside Cap", "Verdict"]
                        st.dataframe(cat_df[disp_cols].style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)

                # 3. Save All Results to History
                log_scan_results(final_df, "scan_mf")
                st.success(f"Audit Complete. Records saved to database.")

                # 4. Scatter Plot (Global Analysis)
                st.subheader("📊 Global Map: Alpha vs Downside Protection")

                fig = px.scatter(final_df, x="Downside Cap", y="Alpha (True)",
                                 color="Verdict", size="Score", text="Symbol",
                                 hover_data=["Sortino", "Win Rate"])
                st.plotly_chart(fig, use_container_width=True)

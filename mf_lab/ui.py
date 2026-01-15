import streamlit as st
import pandas as pd
import numpy as np
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

    # MF Controls
    col_mf1, col_mf2 = st.columns([2, 1])
    with col_mf1:
        mf_limit = st.slider("Max Funds to Scan", 10, 100, 40)
    with col_mf2:
        st.write("")
        st.write("")
        start_mf_scan = st.button("🚀 EXECUTE AUDIT", type="primary", use_container_width=True)

    if start_mf_scan:
        with st.spinner("Fetching Benchmarks..."):
            # Pre-load benchmarks
            benchmarks = {}
            try:
                # Fallback handled in logic but repeated here for safety
                benchmarks['Large Cap'] = get_benchmark_data("^NSEI")
                if benchmarks['Large Cap'].empty: benchmarks['Large Cap'] = get_benchmark_data("^NSEI")

                benchmarks['Mid Cap'] = get_benchmark_data("^NSEMDCP50")
                if benchmarks['Mid Cap'].empty: benchmarks['Mid Cap'] = benchmarks['Large Cap']

                benchmarks['Small Cap'] = get_benchmark_data("^CNXSC")
                if benchmarks['Small Cap'].empty: benchmarks['Small Cap'] = benchmarks['Large Cap']

                benchmarks['Flexi/Other'] = benchmarks['Large Cap']
            except:
                st.error("Critical Benchmark Failure.")
                return

        candidates = discover_funds(limit=mf_limit)
        results = []

        progress = st.progress(0)
        status_text = st.empty()

        for i, c in enumerate(candidates):
            status_text.text(f"Auditing: {c['schemeName'][:30]}...")
            try:
                # Detect Category & Select Benchmark
                cat = get_category(c['schemeName'])
                bench = benchmarks.get(cat, benchmarks['Large Cap'])

                url = f"https://api.mfapi.in/mf/{c['schemeCode']}"
                data = requests.get(url).json()
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
                df['nav'] = df['nav'].astype(float)
                df = df.sort_values('date')
                df['ret'] = df['nav'].pct_change()

                # Risk/Consistency Metrics
                metrics = detect_integrity_issues(df, bench, cat)
                if metrics:
                    # New Scoring Logic
                    # fortress_score = (alpha*4) + (sortino*10) - abs(max_dd) + (win_rate*0.2)
                    raw_score = (metrics['alpha'] * 4) + (metrics['sortino'] * 10) - abs(metrics['max_dd']) + (metrics['win_rate'] * 0.2)

                    results.append({
                        "Symbol": c['schemeName'][:50], # Mapped to Symbol for DB compatibility
                        "Alpha (True)": round(metrics['alpha'], 2),
                        "Sortino": round(metrics['sortino'], 2),
                        "TE (Tracking Error)": round(metrics['te'], 2),
                        "Beta": round(metrics['beta'], 2),
                        "Max Drawdown": round(metrics['max_dd'], 2),
                        "Win Rate": round(metrics['win_rate'], 1),
                        "Verdict": metrics['drift'], # Mapped to Verdict
                        "Score": raw_score, # Intermediate Raw Score
                        "Price": df['nav'].iloc[-1] # Mapped to Price
                    })
            except Exception as e:
                 # print(f"Error processing {c.get('schemeName')}: {e}")
                 continue
            progress.progress((i + 1) / len(candidates))

        if results:
            final_df = pd.DataFrame(results)

            # Normalization (Z-Score + Scaling)
            if len(final_df) > 1:
                # Z-score normalization
                final_df['Score'] = (final_df['Score'] - final_df['Score'].mean()) / final_df['Score'].std()

                # Scale 0-100
                min_s = final_df['Score'].min()
                max_s = final_df['Score'].max()
                if max_s != min_s:
                    final_df['Score'] = ((final_df['Score'] - min_s) / (max_s - min_s)) * 100
                else:
                    final_df['Score'] = 50.0 # Default if all scores are identical

            final_df['Score'] = final_df['Score'].round(1)
            final_df = final_df.sort_values("Score", ascending=False)

            # Save to History
            log_scan_results(final_df, "scan_mf")
            fetch_timestamps.clear()
            fetch_history_data.clear()
            fetch_symbol_history.clear()
            log_audit("MF Scan Completed", "Mutual Funds", f"Saved {len(final_df)} funds.")

            st.success(f"Audit Complete. Found {len(final_df)} funds.")

            # Display Columns
            display_cols = ["Symbol", "Score", "Alpha (True)", "Sortino", "Max Drawdown", "Win Rate", "Beta", "Verdict"]
            st.dataframe(final_df[display_cols].style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)

            # Visualization
            st.subheader("📊 Consistency Map: Alpha vs Downside Protection")

            fig = px.scatter(final_df, x="Max Drawdown", y="Alpha (True)",
                             color="Verdict", size="Score", text="Symbol",
                             hover_data=["Sortino", "Win Rate"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No funds matched criteria or data fetch failed.")

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from .logic import get_benchmark_data, discover_funds, detect_integrity_issues
from utils.db import log_scan_results, fetch_timestamps, fetch_history_data, fetch_symbol_history, log_audit

def render():
    st.subheader("🛡️ Fortress MF Pro: Consistency Lab")

    # MF Controls
    col_mf1, col_mf2, col_mf3 = st.columns([2, 1, 1])
    with col_mf1:
        mf_limit = st.slider("Max Funds to Scan", 10, 100, 40)
    with col_mf2:
        mf_benchmark = st.text_input("Benchmark (Yahoo Ticker)", value="^NSEI")
    with col_mf3:
        st.write("")
        st.write("")
        start_mf_scan = st.button("🚀 EXECUTE AUDIT", type="primary", use_container_width=True)

    if start_mf_scan:
        bench = get_benchmark_data(mf_benchmark)
        if bench.empty:
            st.error("Failed to fetch benchmark data.")
        else:
            candidates = discover_funds(limit=mf_limit)
            results = []

            progress = st.progress(0)
            status_text = st.empty()

            for i, c in enumerate(candidates):
                status_text.text(f"Auditing: {c['schemeName'][:30]}...")
                try:
                    url = f"https://api.mfapi.in/mf/{c['schemeCode']}"
                    data = requests.get(url).json()
                    df = pd.DataFrame(data['data'])
                    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
                    df['nav'] = df['nav'].astype(float)
                    df = df.sort_values('date')
                    df['ret'] = df['nav'].pct_change()

                    # Risk/Consistency Metrics
                    metrics = detect_integrity_issues(df, bench, "Equity")
                    if metrics:
                        # Calculate Sortino
                        neg_ret = df['ret'][df['ret'] < 0]
                        sortino = (df['ret'].mean() * 252) / (neg_ret.std() * np.sqrt(252)) if neg_ret.std() > 0 else 0

                        fortress_score = round((metrics['alpha'] * 4) + (sortino * 12), 1)
                        # Normalize 0-100ish
                        # fortress_score = max(0, min(100, fortress_score)) # Optional normalization, but logic provided was specific.

                        results.append({
                            "Symbol": c['schemeName'][:50], # Mapped to Symbol for DB compatibility
                            "Alpha (True)": round(metrics['alpha'], 2),
                            "Sortino": round(sortino, 2),
                            "TE (Tracking Error)": round(metrics['te'], 2),
                            "Beta": round(metrics['beta'], 2),
                            "Verdict": metrics['drift'], # Mapped to Verdict
                            "Score": fortress_score, # Mapped to Score
                            "Price": df['nav'].iloc[-1] # Mapped to Price
                        })
                except Exception as e:
                     # print(f"Error processing {c.get('schemeName')}: {e}")
                     continue
                progress.progress((i + 1) / len(candidates))

            if results:
                final_df = pd.DataFrame(results).sort_values("Score", ascending=False)

                # Save to History
                log_scan_results(final_df, "scan_mf")
                fetch_timestamps.clear()
                fetch_history_data.clear()
                fetch_symbol_history.clear()
                log_audit("MF Scan Completed", "Mutual Funds", f"Saved {len(final_df)} funds.")

                st.success(f"Audit Complete. Found {len(final_df)} funds.")
                st.dataframe(final_df.style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)

                # Visualization
                st.subheader("📊 Consistency Map: Alpha vs Downside Protection")

                fig = px.scatter(final_df, x="TE (Tracking Error)", y="Alpha (True)",
                                 color="Verdict", size="Score", text="Symbol",
                                 hover_data=["Sortino", "Beta"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No funds matched criteria or data fetch failed.")

import io
import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from mf_lab.logic import DEFAULT_SCHEMES, backtest_vs_benchmark, classify_category, fetch_mf_snapshot
from utils.db import log_audit, log_scan_results

logger = logging.getLogger(__name__)


def render():
    st.subheader("🛡️ Fortress MF Pro: Consistency Lab")
    debug_mode = st.sidebar.toggle("MF Debug Mode", value=False)

    scheme_text = st.sidebar.text_area("Scheme codes (comma separated)", ",".join(DEFAULT_SCHEMES[:10]))
    scheme_codes = [s.strip() for s in scheme_text.split(",") if s.strip()]

    st.sidebar.markdown("### Simple Mode Filters")
    categories = st.sidebar.multiselect("Category", ["Equity", "Debt", "Hybrid"], default=["Equity", "Debt", "Hybrid"])
    min_aum = st.sidebar.number_input("Min AUM (₹ Cr)", min_value=0, value=500)
    min_sharpe = st.sidebar.number_input("Min Sharpe", min_value=0.0, value=1.0, step=0.1)

    try:
        with st.spinner("Loading data..."):
            df = fetch_mf_snapshot(scheme_codes)
    except Exception as e:
        logger.error(f"mf_lab error: {e}")
        st.warning("Data load failed - retry or check logs")
        return

    if df.empty:
        st.warning("Data load failed - retry or check logs")
        return

    df["Category"] = df["Scheme"].map(classify_category)
    df["AUM (₹ Cr)"] = min_aum + 1000  # placeholder until provider is wired
    filtered = df[(df["Category"].isin(categories)) & (df["AUM (₹ Cr)"] >= min_aum) & (df["Sharpe"] >= min_sharpe)].copy()
    filtered = filtered.fillna(0)

    display_cols = ["Scheme", "NAV", "1Y Return", "Sharpe", "Consistency Score", "Scheme Code", "Category"]
    st.dataframe(filtered[display_cols], use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export MF CSV", data=csv_bytes, file_name="mf_consistency.csv", mime="text/csv")

    scheme_choice = st.selectbox("Backtest scheme", filtered["Scheme Code"].tolist() if not filtered.empty else scheme_codes)
    if scheme_choice:
        bt = backtest_vs_benchmark(scheme_choice)
        if not bt.empty:
            st.line_chart(bt)

    try:
        top10 = filtered.head(10).copy()
        top10["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_scan_results(top10, table_name="scan_mf")
        log_audit("MF Daily Audit", "Mutual Funds", f"Top schemes logged: {len(top10)}")
    except Exception as e:
        logger.error(f"mf_lab error: {e}")

    if debug_mode:
        st.markdown("### Debug")
        st.dataframe(df, use_container_width=True)

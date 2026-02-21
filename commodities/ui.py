import logging
from datetime import datetime

import plotly.express as px
import streamlit as st

from commodities.logic import build_commodities_frame
from utils.db import log_audit, register_scan, save_scan_results

logger = logging.getLogger(__name__)


def render(broker_choice="Zerodha"):
    st.header("🌍 Commodities Intelligence Terminal")
    commodity = st.sidebar.selectbox("Commodity", ["All", "Gold", "Silver", "Crude", "Copper"])
    debug_mode = st.sidebar.toggle("Commodities Debug Mode", value=False)

    try:
        with st.spinner("Loading data..."):
            df = build_commodities_frame(None if commodity == "All" else commodity)
    except Exception as e:
        logger.error(f"commodities error: {e}")
        st.warning("Data load failed - retry or check logs")
        return

    if df.empty:
        st.warning("Data load failed - retry or check logs")
        return

    st.dataframe(df[["Commodity", "Price", "ATR", "Spread %", "Score", "USDINR"]], use_container_width=True)
    st.download_button("⬇️ Export Commodities CSV", df.to_csv(index=False).encode("utf-8"), "commodities_scan.csv", "text/csv")

    heat = px.imshow(df[["Spread %", "Score"]].T, aspect="auto", color_continuous_scale="RdYlGn", title="Arbitrage Heatmap")
    heat.update_xaxes(tickvals=list(range(len(df))), ticktext=df["Commodity"].tolist())
    st.plotly_chart(heat, use_container_width=True)

    try:
        scan_id = register_scan(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), universe="Commodities", scan_type="COMMODITY", status="Completed")
        save_scan_results(scan_id, df)
        log_audit("Commodity scan", "Commodities", f"Rows logged: {len(df)}")
    except Exception as e:
        logger.error(f"commodities error: {e}")

    if debug_mode:
        st.dataframe(df, use_container_width=True)

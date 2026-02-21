# streamlit_app.py - v9.6 MASTER TERMINAL
import streamlit as st

from utils.db import init_db
import mf_lab.ui
import stock_scanner.ui
import commodities.ui
import history.ui
from options_algo.ui import render as options_algo_render

init_db()

st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v9.6 — Institutional Terminal")

st.sidebar.title("Navigation")
debug_mode = st.sidebar.toggle("Global Debug Mode", value=False)
selected_view = st.sidebar.radio(
    "Select Module",
    ["🚀 Live Scanner", "🛡️ MF Consistency Lab", "🌍 Commodities Terminal", "🤖 Options Algos", "📜 Scan History"],
)
st.sidebar.markdown("---")

if selected_view == "🚀 Live Scanner":
    portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice, scoring_config = stock_scanner.ui.render_sidebar()
    stock_scanner.ui.render(portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice, scoring_config)
elif selected_view == "🛡️ MF Consistency Lab":
    try:
        mf_lab.ui.render()
    except Exception as e:
        st.warning("Data load failed - retry or check logs")
        if debug_mode:
            st.exception(e)
elif selected_view == "🌍 Commodities Terminal":
    broker_choice = st.sidebar.selectbox("Preferred Broker", ["Zerodha", "Dhan"], key="comm_broker")
    try:
        commodities.ui.render(broker_choice)
    except Exception as e:
        st.warning("Data load failed - retry or check logs")
        if debug_mode:
            st.exception(e)
elif selected_view == "🤖 Options Algos":
    broker_choice = st.sidebar.selectbox("Preferred Broker", ["Zerodha", "Dhan"], key="algo_broker")
    try:
        options_algo_render(broker_choice)
    except Exception as e:
        st.warning("Data load failed - retry or check logs")
        if debug_mode:
            st.exception(e)
else:
    history.ui.render()

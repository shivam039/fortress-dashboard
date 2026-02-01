# streamlit_app.py - v9.6 MASTER TERMINAL
import time, sqlite3
from datetime import datetime
import streamlit as st
import pandas as pd
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ---------------- CONFIG ----------------
try:
    from fortress_config import TICKER_GROUPS
except ImportError:
    st.error("Configuration file 'fortress_config.py' not found.")
    st.stop()

# ---------------- MODULE IMPORTS ----------------
from utils.db import (
    init_db, log_audit, get_table_name_from_universe,
    fetch_timestamps, fetch_history_data, fetch_symbol_history
)
import mf_lab.ui
import stock_scanner.ui
import commodities.ui
import history.ui
from options_algo.ui import render as options_algo_render

# Initialize Database
init_db()

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v9.6 — Institutional Terminal")

# ---------------- NAVIGATION ----------------
st.sidebar.title("Navigation")
nav_options = [
    "🚀 Live Scanner",
    "🛡️ MF Consistency Lab",
    "🌍 Commodities Terminal",
    "🤖 Options Algos",
    "📜 Scan History"
]
# Server-Side Navigation
selected_view = st.sidebar.radio("Select Module", nav_options)
st.sidebar.markdown("---")

# ---------------- MODULE ROUTING ----------------

if selected_view == "🚀 Live Scanner":
    # Sidebar: Stock Scanner Specifics
    # We call the scanner's sidebar renderer here
    portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice = stock_scanner.ui.render_sidebar()

    # Main Content
    stock_scanner.ui.render(portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice)

elif selected_view == "🛡️ MF Consistency Lab":
    # Sidebar handled internally by mf_lab.ui.render() (Asset Class, View Mode, Admin Tools)
    # They are wrapped in st.sidebar context in the module
    mf_lab.ui.render()

elif selected_view == "🌍 Commodities Terminal":
    # Sidebar: Broker Choice for Execution
    st.sidebar.subheader("Execution Settings")
    broker_choice = st.sidebar.selectbox("Preferred Broker", ["Zerodha", "Dhan"], key="comm_broker")

    commodities.ui.render(broker_choice)

elif selected_view == "🤖 Options Algos":
    # Sidebar: Broker Choice
    st.sidebar.subheader("Execution Settings")
    broker_choice = st.sidebar.selectbox("Preferred Broker", ["Zerodha", "Dhan"], key="algo_broker")

    options_algo_render(broker_choice)

elif selected_view == "📜 Scan History":
    history.ui.render()

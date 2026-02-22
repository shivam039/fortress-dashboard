import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time
import threading
import queue
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

from fortress_config import TICKER_GROUPS, INDEX_BENCHMARKS
from .logic import (
    check_institutional_fortress,
    apply_advanced_scoring,
    DEFAULT_SCORING_CONFIG,
    get_stock_data,
    backtest_top_picks,
)
from stock_scanner.config import ALL_COLUMNS
from utils.db import log_audit, get_table_name_from_universe, log_scan_results, fetch_timestamps, fetch_history_data, fetch_symbol_history, register_scan, save_scan_results, update_scan_status
from utils.broker_mappings import generate_zerodha_url, generate_dhan_url
import stock_scanner.pulse as pulse


@st.cache_data(ttl="10m")
def _apply_advanced_scoring_cached(df, scoring_config):
    """Cache heavy scoring pipeline for iterative UI reruns."""
    return apply_advanced_scoring(df, scoring_config)

def generate_action_link(row, broker_choice):
    qty = row.get("Position_Qty", 0)
    symbol = row["Symbol"]
    price = row.get("Price", 0)

    if broker_choice == "Zerodha":
        return generate_zerodha_url(symbol, qty)
    else:
        return generate_dhan_url(symbol, qty, price)

def get_column_config(display_cols, broker_choice):
    st_column_config = {}
    for col in display_cols:
        cfg = ALL_COLUMNS.get(col, {})
        fmt = cfg.get("format")
        if col == "Actions":
            label = f"⚡ Trade ({broker_choice})"
            st_column_config[col] = st.column_config.LinkColumn(label, display_text="⚡ Trade")
        elif cfg.get("type")=="progress":
            st_column_config[col] = st.column_config.ProgressColumn(cfg["label"],min_value=cfg["min"],max_value=cfg["max"])
        elif fmt:
            st_column_config[col] = st.column_config.NumberColumn(cfg["label"],format=fmt)
        else:
            st_column_config[col] = st.column_config.TextColumn(cfg.get("label", col))
    return st_column_config

def render_sidebar():
    st.sidebar.title("💰 Portfolio & Risk")
    # Persistence: Use keys to store in session_state
    if "portfolio_val" not in st.session_state: st.session_state["portfolio_val"] = 1000000
    if "risk_pct_slider" not in st.session_state: st.session_state["risk_pct_slider"] = 1.0
    if "selected_universe" not in st.session_state: st.session_state["selected_universe"] = list(TICKER_GROUPS.keys())[0]

    portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", value=1000000, step=50000, key="portfolio_val")
    risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.1, key="risk_pct_slider")/100

    # Broker Selection
    broker_choice = st.sidebar.selectbox("Preferred Broker", ["Zerodha", "Dhan"], key="broker_choice")

    selected_universe = st.sidebar.selectbox("Select Index", list(TICKER_GROUPS.keys()), key="selected_universe")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Advanced Scoring Weights")
    tech_w = st.sidebar.slider("Technical Weight %", 0, 100, 50, 1)
    fund_w = st.sidebar.slider("Fundamental Weight %", 0, 100, 25, 1)
    sent_w = st.sidebar.slider("Sentiment Weight %", 0, 100, 15, 1)
    ctx_w = st.sidebar.slider("Market Context / RS / MTF Weight %", 0, 100, 10, 1)
    total = tech_w + fund_w + sent_w + ctx_w
    if total <= 0:
        total = 100.0
    weights = {
        "technical": tech_w / total,
        "fundamental": fund_w / total,
        "sentiment": sent_w / total,
        "context": ctx_w / total,
    }
    st.sidebar.caption(
        f"Normalized Weights → Tech {weights['technical']*100:.1f}% | Fund {weights['fundamental']*100:.1f}% | "
        f"Sent {weights['sentiment']*100:.1f}% | Context {weights['context']*100:.1f}%"
    )

    enable_regime = st.sidebar.checkbox("Enable regime scaling", value=True)
    liquidity_cr_min = st.sidebar.number_input("Liquidity gate (₹ Cr, 20D avg)", min_value=0.0, value=8.0, step=0.5)
    market_cap_cr_min = st.sidebar.number_input("Market cap gate (₹ Cr)", min_value=0.0, value=1500.0, step=50.0)
    price_min = st.sidebar.number_input("Minimum price gate (₹)", min_value=0.0, value=80.0, step=5.0)

    if "market_pulse_data" not in st.session_state:
        st.session_state["market_pulse_data"] = None

    regime = {"Market_Regime": "Range", "Regime_Multiplier": 1.0, "VIX": 20.0}
    if st.session_state["market_pulse_data"]:
        regime = st.session_state["market_pulse_data"].get("regime", regime)

    st.sidebar.info(
        f"Regime: {regime['Market_Regime']} | Multiplier: {regime['Regime_Multiplier']:.2f} | India VIX: {regime['VIX']:.2f}"
    )

    # Sidebar Multiselect for Dynamic Columns
    default_full_list = list(ALL_COLUMNS.keys())
    if "selected_columns" not in st.session_state:
        st.session_state["selected_columns"] = default_full_list
    st.session_state["selected_columns"] = st.sidebar.multiselect(
        "Select Columns to Display",
        options=list(ALL_COLUMNS.keys()),
        default=st.session_state["selected_columns"],
    )
    selected_columns = st.session_state["selected_columns"]

    scoring_config = {
        "weights": weights,
        "enable_regime": enable_regime,
        "liquidity_cr_min": liquidity_cr_min,
        "market_cap_cr_min": market_cap_cr_min,
        "price_min": price_min,
        "max_debt_to_equity": DEFAULT_SCORING_CONFIG["max_debt_to_equity"],
        "min_interest_coverage": DEFAULT_SCORING_CONFIG["min_interest_coverage"],
        "regime": regime,
    }

    return portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice, scoring_config

def _save_scan(df, universe):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scan_id = register_scan(timestamp, universe=universe, scan_type="STOCK", status="Completed")

    df['Universe'] = universe # Add metadata
    save_scan_results(scan_id, df)

    # Clear cache after new scan so history tab updates
    fetch_timestamps.clear()
    fetch_history_data.clear()
    fetch_symbol_history.clear()

    log_audit("Scan Completed", universe, f"Saved {len(df)} records to unified history (ID: {scan_id})")
    return timestamp

def _display_scan_results(df, universe, broker_choice, scoring_config, timestamp=None):
    if df.empty:
        st.warning("No data retrieved.")
        return

    df = _apply_advanced_scoring_cached(df, scoring_config).sort_values("Score",ascending=False)
    filtered_out_df = df[df.get("Quality_Gate_Pass", True) == False].copy()
    actionable_df = df[df.get("Quality_Gate_Pass", True) == True].copy()

    st.success(f"Scan Complete: {len(actionable_df[actionable_df['Score']>=60])} actionable setups.")

    # --- GENERATE ACTIONS COLUMN ---
    # Universal Action Links: Enabled for all scan results (No Verdict Gate)
    df["Actions"] = df.apply(lambda row: generate_action_link(row, broker_choice), axis=1)

    # --- SECTOR INTELLIGENCE TERMINAL ---
    st.subheader("🔥 Sector Intelligence & Rotation")

    # Aggregate Sector Metrics
    if "Sector" in df.columns and "Velocity" in df.columns:
        sector_stats = df.groupby("Sector").agg({
            "Velocity": "mean",
            "Above_EMA200": "mean", # Breadth (0-1)
            "Score": "mean"
        }).reset_index()

        # Formatting
        sector_stats["Breadth (%)"] = (sector_stats["Above_EMA200"] * 100).round(1)
        sector_stats["Avg Score"] = sector_stats["Score"].round(1)
        sector_stats["Velocity"] = sector_stats["Velocity"].round(2)

        # Thesis Generation
        def get_thesis(row):
            if row["Score"] > 75 and row["Velocity"] > 0:
                return "🐂 Bullish Accumulation"
            elif row["Score"] < 35 and row["Breadth (%)"] < 40:
                return "❄️ Structural Weakness"
            elif row["Velocity"] > 2:
                return "🚀 High Momentum"
            else:
                return "⚖️ Neutral / Rotation"

        sector_stats["Thesis"] = sector_stats.apply(get_thesis, axis=1)

        # Classification
        def check_rise(row):
            if row['Velocity'] > 0 and row['Breadth (%)'] > 70: return "🔥 YES"
            return ""

        def check_fall(row):
            if row['Velocity'] < 0 or row['Breadth (%)'] < 40: return "❄️ YES"
            return ""

        sector_stats['On the Rise'] = sector_stats.apply(check_rise, axis=1)
        sector_stats['On the Fall'] = sector_stats.apply(check_fall, axis=1)

        # Display Dashboard
        st.dataframe(
            sector_stats[["Sector", "Thesis", "Velocity", "Breadth (%)", "Avg Score", "On the Rise", "On the Fall"]].sort_values("Velocity", ascending=False),
            use_container_width=True,
            column_config={
                "Velocity": st.column_config.NumberColumn("Momentum Vel", format="%.2f%%"),
                "Breadth (%)": st.column_config.ProgressColumn("Inst. Breadth", min_value=0, max_value=100, format="%.1f%%"),
                "Avg Score": st.column_config.ProgressColumn("Sector Strength", min_value=0, max_value=100)
            },
            hide_index=True
        )

    # --- UPDATED STRATEGIC PICKS (ALL COLUMNS) ---
    st.subheader("🎯 Strategic Picks")
    momentum_picks = df[df['Strategy'] == "Momentum Pick"]
    lt_picks = df[df['Strategy'] == "Long-Term Pick"]
    if universe == "Nifty Smallcap 250":
        momentum_picks = momentum_picks[momentum_picks["Score"] >= 60]
        lt_picks = lt_picks[lt_picks["Score"] >= 60]

    # Use all columns selected in the sidebar for these Strategic tables
    default_full_list = list(ALL_COLUMNS.keys())
    display_cols = [
        c for c in st.session_state.get("selected_columns", default_full_list)
        if c in df.columns
    ]
    st_column_config = get_column_config(display_cols, broker_choice)

    if not momentum_picks.empty:
        st.markdown(f"#### 🚀 Momentum Picks ({len(momentum_picks)})")
        st.dataframe(momentum_picks[display_cols], use_container_width=True, hide_index=True, column_config=st_column_config)

    if not lt_picks.empty:
        st.markdown(f"#### 💎 Long-Term Picks ({len(lt_picks)})")
        st.dataframe(lt_picks[display_cols], use_container_width=True, hide_index=True, column_config=st_column_config)

    st.markdown("#### 🧪 Backtesting Hooks")
    if st.button("Run Backtest for This Scan", use_container_width=True):
        if timestamp:
            bt_df = backtest_top_picks(timestamp)
            if bt_df.empty:
                st.info("No backtest data available for selected scan timestamp.")
            else:
                st.dataframe(bt_df, use_container_width=True, hide_index=True)
        else:
             st.info("Scan timestamp not available.")

    # Ensure 'Actions' is available in display if selected
    # Note: selected_columns might contain 'Actions', so we ensure it exists in df (done above)
    display_cols = [c for c in st.session_state.get("selected_columns", default_full_list) if c in df.columns]
    display_df = df[display_cols]

    # --- FILTERING FOR SMALLCAP 250 ---
    # Strictly filter display and export to show only 'Bullish' or 'Strong Bullish' verdicts (Score >= 60).
    # This applies only to the UI table and CSV, preserving 'df' for Sector Intelligence calculations.
    if universe == "Nifty Smallcap 250":
        display_df = display_df[display_df["Score"] >= 60]
        if display_df.empty:
            st.warning("No tickers met the strict criteria (Score >= 60) for Smallcap 250.")

    st_column_config = get_column_config(display_cols, broker_choice)

    st.dataframe(display_df,use_container_width=True,height=600,column_config=st_column_config)

    if not filtered_out_df.empty:
        with st.expander(f"Filtered Out ({len(filtered_out_df)}) - Hard Quality Gates", expanded=False):
            filtered_cols = [c for c in display_cols if c in filtered_out_df.columns]
            st.dataframe(filtered_out_df[filtered_cols], use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export Trades to CSV",data=csv,
                       file_name=f"Fortress_Trades_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv",use_container_width=True)

    # ---------------- HEATMAP ----------------
    if not df.empty and "Score" in df.columns:
        st.subheader("📊 Conviction Heatmap")
        plt.figure(figsize=(12,len(df)/2))
        df["Conviction_Band"] = df["Score"].apply(lambda x: "🔥 High (85+)" if x>=85 else "🚀 Pass (60-85)" if x>=60 else "🟡 Watch (<60)")
        heatmap_data = df.pivot_table(index="Symbol", columns="Conviction_Band", values="Score", fill_value=0)
        sns.heatmap(heatmap_data, annot=True, cmap="Greens", cbar=False, linewidths=0.5, linecolor='grey')
        st.pyplot(plt)
    else:
        st.info("Insufficient data for heatmap generation.")

@st.fragment
def _run_smallcap_scan_fragment(broker_choice, scoring_config):
    state = st.session_state.get("smallcap_scan_state")
    if not state:
        return

    universe = state["universe"]
    if universe != "Nifty Smallcap 250":
        return

    if state.get("status") == "running":
        tickers = state["tickers"]
        chunk_size = state["chunk_size"]
        i = state["index"]
        num_chunks = max(1, (len(tickers) + chunk_size - 1) // chunk_size)

        # Calculate progress correctly
        current_chunk_idx = i // chunk_size
        progress_val = min(current_chunk_idx / num_chunks, 1.0)

        with st.spinner("Scanning universe..."):
            progress_bar = st.progress(progress_val)
            status_text = st.empty()

            if i < len(tickers):
                chunk = tickers[i:i + chunk_size]
                status_text.text(f"Scanning {i}/{len(tickers)} tickers...")

                try:
                    batch_data = get_stock_data(chunk, period="1y", interval="1d", group_by="ticker")
                    for ticker in chunk:
                        try:
                            tkr_obj = yf.Ticker(ticker)
                            hist = batch_data[ticker].dropna() if len(chunk) > 1 else batch_data.dropna()
                            if not hist.empty and len(hist) >= 210:
                                res = check_institutional_fortress(
                                    ticker,
                                    hist,
                                    tkr_obj,
                                    state["portfolio_val"],
                                    state["risk_pct"],
                                    selected_universe=universe,
                                    regime_data=scoring_config.get("regime")
                                )
                                if res and res["Score"] >= 60:
                                    state["results"].append(res)
                        except Exception as inner_e:
                            state["errors"].append(f"{ticker}: {inner_e}")
                except Exception as e:
                    state["errors"].append(f"Chunk {i}: {e}")

                state["index"] = i + chunk_size
                time.sleep(1.2) # Throttling for Smallcap
                st.session_state["smallcap_scan_state"] = state
                st.rerun()
        else:
            # Scan Complete
            state["status"] = "completed"

            # Save results once
            df = pd.DataFrame(state["results"])
            if not df.empty:
                timestamp = _save_scan(df, universe)
                state["timestamp"] = timestamp

            st.session_state["smallcap_scan_state"] = state
            st.rerun()

    elif state.get("status") == "completed":
        results = state["results"]
        for err in state["errors"]:
            st.error(f"Batch Error: {err}")

        if results:
            df = pd.DataFrame(results)
            _display_scan_results(df, universe, broker_choice, scoring_config, timestamp=state.get("timestamp"))
        else:
             st.warning("No results found.")

def render(portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice, scoring_config):
    if "smallcap_scan_state" not in st.session_state:
        st.session_state["smallcap_scan_state"] = None
    # ---------------- SEARCH FEATURE ----------------
    search_symbol = st.text_input("🔍 Search Stock (Symbol)", placeholder="e.g., RELIANCE.NS")
    if search_symbol:
        search_symbol = search_symbol.upper().strip()
        if "." not in search_symbol:
            search_symbol += ".NS"

        try:
            with st.spinner(f"Analyzing {search_symbol}..."):
                search_tkr = yf.Ticker(search_symbol)
                search_hist = get_stock_data(search_symbol, period="2y", interval="1d")

            if not search_hist.empty:
                search_res = check_institutional_fortress(
                    search_symbol,
                    search_hist,
                    search_tkr,
                    portfolio_val,
                    risk_pct,
                    regime_data=scoring_config.get("regime")
                )
                if search_res:
                    search_df = pd.DataFrame([search_res])
                    search_df["Actions"] = search_df.apply(lambda row: generate_action_link(row, broker_choice), axis=1)

                    # Show columns based on sidebar selection
                    search_cols = [c for c in st.session_state["selected_columns"] if c in search_df.columns]
                    search_config = get_column_config(search_cols, broker_choice)

                    st.dataframe(search_df[search_cols], use_container_width=True, hide_index=True, column_config=search_config)
                else:
                    st.warning(f"Insufficient data or analysis failed for {search_symbol} (Need >210 candles).")
            else:
                st.error(f"No data found for {search_symbol}. Check ticker symbol.")
        except Exception as e:
            st.error(f"Search Error: {str(e)}")

    # ---------------- MARKET PULSE ----------------
    st.subheader("🌐 Market Pulse")
    col_pulse_btn, col_pulse_ph = st.columns([1, 4])
    with col_pulse_btn:
        if st.button("Refresh Market Pulse", use_container_width=True):
            with st.spinner("Updating Market Pulse..."):
                pulse_data = pulse.fetch_market_pulse_data()
                st.session_state["market_pulse_data"] = pulse_data
                st.rerun()

    if st.session_state.get("market_pulse_data"):
        with st.expander("Show/Hide Market Pulse Details", expanded=True):
             pulse.render_market_pulse(st.session_state["market_pulse_data"])
    else:
        st.info("Click 'Refresh Market Pulse' to load latest indices and regime data.")

    # ---------------- MAIN SCAN ----------------
    execute_scan = st.button("🚀 EXECUTE SYSTEM SCAN", type="primary", use_container_width=True)
    results = None

    if execute_scan and selected_universe == "Nifty Smallcap 250":
        tickers = TICKER_GROUPS[selected_universe]
        chunk_size = 50
        st.session_state["smallcap_scan_state"] = {
            "universe": selected_universe,
            "tickers": tickers,
            "chunk_size": chunk_size,
            "index": 0,
            "results": [],
            "errors": [],
            "portfolio_val": portfolio_val,
            "risk_pct": risk_pct,
            "status": "running"
        }
        st.rerun()

    smallcap_state = st.session_state.get("smallcap_scan_state")
    if smallcap_state and selected_universe == "Nifty Smallcap 250":
        _run_smallcap_scan_fragment(broker_choice, scoring_config)

    elif execute_scan and selected_universe != "Nifty Smallcap 250":
        tickers = TICKER_GROUPS[selected_universe]
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Log Start
        log_audit("Scan Started", selected_universe, f"Scanning {len(tickers)} tickers")

        # --- CHUNKED PROCESSING LOGIC ---
        chunk_size = len(tickers) # No smallcap throttle needed

        def _scan_worker(result_queue):
            local_results = []
            errors = []
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i : i + chunk_size]
                try:
                    batch_data = get_stock_data(chunk, period="1y", interval="1d", group_by="ticker")
                    for ticker in chunk:
                        try:
                            tkr_obj = yf.Ticker(ticker)
                            hist = batch_data[ticker].dropna() if len(chunk) > 1 else batch_data.dropna()
                            if not hist.empty and len(hist) >= 210:
                                res = check_institutional_fortress(
                                    ticker,
                                    hist,
                                    tkr_obj,
                                    portfolio_val,
                                    risk_pct,
                                    selected_universe=selected_universe,
                                    regime_data=scoring_config.get("regime")
                                )
                                if res and (selected_universe != "Nifty Smallcap 250" or res["Score"] >= 60):
                                    local_results.append(res)
                        except Exception as inner_e:
                            errors.append(f"{ticker}: {inner_e}")
                except Exception as e:
                    errors.append(f"Chunk {i}: {e}")
                result_queue.put({"progress": min(i + chunk_size, len(tickers))})

            result_queue.put({"done": True, "results": local_results, "errors": errors})

        result_queue = queue.Queue()
        worker = threading.Thread(target=_scan_worker, args=(result_queue,), daemon=True)
        worker.start()
        done = False
        while not done:
            msg = result_queue.get()
            if "progress" in msg:
                scanned = msg["progress"]
                progress_bar.progress(min(scanned / len(tickers), 1.0))
                status_text.text(f"Scanned {scanned}/{len(tickers)} tickers...")
            if msg.get("done"):
                results.extend(msg.get("results", []))
                for err in msg.get("errors", []):
                    st.error(f"Batch Error: {err}")
                done = True

    if results is not None:
        if results:
            df = pd.DataFrame(results)
            timestamp = _save_scan(df, selected_universe)
            _display_scan_results(df, selected_universe, broker_choice, scoring_config, timestamp=timestamp)
        else:
            st.warning("No data retrieved. Check internet or ticker config.")
            log_audit("Scan Failed", selected_universe, "No data retrieved")

    st.caption("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns | ATR SL | Analyst Dispersion | Full Logic")

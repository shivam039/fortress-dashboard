import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from fortress_config import TICKER_GROUPS, INDEX_BENCHMARKS
from .logic import check_institutional_fortress, apply_advanced_scoring, DEFAULT_SCORING_CONFIG
from .config import ALL_COLUMNS
from utils.db import log_audit, get_table_name_from_universe, log_scan_results, fetch_timestamps, fetch_history_data, fetch_symbol_history, register_scan, save_scan_results, update_scan_status
from utils.broker_mappings import generate_zerodha_url, generate_dhan_url

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
    portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", value=1000000, step=50000)
    risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.1)/100

    # Broker Selection
    broker_choice = st.sidebar.selectbox("Preferred Broker", ["Zerodha", "Dhan"])

    selected_universe = st.sidebar.selectbox("Select Index", list(TICKER_GROUPS.keys()))

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Advanced Scoring")
    tech_w = st.sidebar.slider("Technical Weight", 0.0, 1.0, 0.50, 0.05)
    fund_w = st.sidebar.slider("Fundamental Weight", 0.0, 1.0, 0.25, 0.05)
    sent_w = st.sidebar.slider("Sentiment Weight", 0.0, 1.0, 0.15, 0.05)
    ctx_w = st.sidebar.slider("Context / RS / MTF Weight", 0.0, 1.0, 0.10, 0.05)
    total = tech_w + fund_w + sent_w + ctx_w
    if total <= 0:
        total = 1.0
    weights = {
        "technical": tech_w / total,
        "fundamental": fund_w / total,
        "sentiment": sent_w / total,
        "context": ctx_w / total,
    }

    enable_regime = st.sidebar.checkbox("Enable regime scaling", value=True)
    liquidity_cr_min = st.sidebar.number_input("Liquidity gate (₹ Cr, 20D avg)", min_value=0.0, value=8.0, step=0.5)

    # Sidebar Multiselect for Dynamic Columns
    selected_columns = st.sidebar.multiselect(
        "Select Columns to Display", options=list(ALL_COLUMNS.keys()), default=list(ALL_COLUMNS.keys())
    )

    scoring_config = {
        "weights": weights,
        "enable_regime": enable_regime,
        "liquidity_cr_min": liquidity_cr_min,
        "market_cap_cr_min": DEFAULT_SCORING_CONFIG["market_cap_cr_min"],
        "price_min": DEFAULT_SCORING_CONFIG["price_min"],
        "max_debt_to_equity": DEFAULT_SCORING_CONFIG["max_debt_to_equity"],
        "min_interest_coverage": DEFAULT_SCORING_CONFIG["min_interest_coverage"],
    }

    return portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice, scoring_config

def render(portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice, scoring_config):
    # ---------------- SEARCH FEATURE ----------------
    search_symbol = st.text_input("🔍 Search Stock (Symbol)", placeholder="e.g., RELIANCE.NS")
    if search_symbol:
        search_symbol = search_symbol.upper().strip()
        if "." not in search_symbol:
            search_symbol += ".NS"

        try:
            with st.spinner(f"Analyzing {search_symbol}..."):
                search_tkr = yf.Ticker(search_symbol)
                search_hist = yf.download(search_symbol, period="2y", progress=False)

            if not search_hist.empty:
                search_res = check_institutional_fortress(search_symbol, search_hist, search_tkr, portfolio_val, risk_pct)
                if search_res:
                    search_df = pd.DataFrame([search_res])
                    search_df["Actions"] = search_df.apply(lambda row: generate_action_link(row, broker_choice), axis=1)

                    # Show columns based on sidebar selection
                    search_cols = [c for c in selected_columns if c in search_df.columns]
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
    pulse_cols = st.columns(len(INDEX_BENCHMARKS))
    for i,(name,symbol) in enumerate(INDEX_BENCHMARKS.items()):
        try:
            idx_data = yf.download(symbol, period="1y", progress=False)
            p_close = idx_data["Close"].iloc[-1]
            p_ema = ta.ema(idx_data["Close"],200).iloc[-1]
            p_status = "🟢 BULL" if p_close>p_ema else "🔴 BEAR"
            pulse_cols[i].metric(name,f"{p_close:,.0f}",p_status)
        except: pass

    # ---------------- MAIN SCAN ----------------
    if st.button("🚀 EXECUTE SYSTEM SCAN",type="primary",use_container_width=True):
        tickers = TICKER_GROUPS[selected_universe]
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Log Start
        log_audit("Scan Started", selected_universe, f"Scanning {len(tickers)} tickers")

        for i,ticker in enumerate(tickers):
            status_text.text(f"Scanning {ticker} ({i+1}/{len(tickers)})")
            try:
                tkr = yf.Ticker(ticker)
                hist = yf.download(ticker, period="2y", progress=False)
                if not hist.empty:
                    res = check_institutional_fortress(ticker,hist,tkr,portfolio_val,risk_pct)
                    if res: results.append(res)
                time.sleep(0.7)
            except: pass
            progress_bar.progress((i+1)/len(tickers))

        if results:
            df = pd.DataFrame(results)

            # Sector rotation bonus (top 3 sectors by 3M perf proxy = avg Ret_90D)
            sector_perf = df.groupby("Sector", as_index=False)["Ret_90D"].mean().sort_values("Ret_90D", ascending=False)
            top_sectors = set(sector_perf.head(3)["Sector"].tolist())
            df["Sector_Rotation_Bonus"] = df["Sector"].isin(top_sectors).astype(int) * 10
            df["Context_Raw"] = pd.to_numeric(df.get("Context_Raw", 0), errors="coerce").fillna(0) + df["Sector_Rotation_Bonus"]

            df = apply_advanced_scoring(df, scoring_config).sort_values("Score",ascending=False)
            status_text.success(f"Scan Complete: {len(df[df['Score']>=60])} actionable setups.")

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

            # --- STRATEGY PICKS ---
            st.subheader("🎯 Strategic Picks")
            c1, c2 = st.columns(2)

            momentum_picks = df[df['Strategy'] == "Momentum Pick"]
            lt_picks = df[df['Strategy'] == "Long-Term Pick"]

            with c1:
                st.markdown(f"#### 🚀 Momentum Picks ({len(momentum_picks)})")
                if not momentum_picks.empty:
                    st.dataframe(momentum_picks[['Symbol', 'Price', 'RSI', 'Target_10D', 'Score', 'Actions']], use_container_width=True, hide_index=True,
                                 column_config={
                                     "Actions": st.column_config.LinkColumn("Execute"),
                                     "Target_10D": st.column_config.NumberColumn("10D Target", format="₹%.2f")
                                 })

            with c2:
                st.markdown(f"#### 💎 Long-Term Picks ({len(lt_picks)})")
                if not lt_picks.empty:
                    st.dataframe(lt_picks[['Symbol', 'Price', 'Target_10D', 'Dispersion_Alert', 'Score', 'Actions']], use_container_width=True, hide_index=True,
                                 column_config={
                                     "Actions": st.column_config.LinkColumn("Execute"),
                                     "Target_10D": st.column_config.NumberColumn("10D Target", format="₹%.2f")
                                 })

            # Log Logic
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scan_id = register_scan(timestamp, universe=selected_universe, scan_type="STOCK", status="Completed")

            df['Universe'] = selected_universe # Add metadata
            save_scan_results(scan_id, df)

            # Clear cache after new scan so history tab updates
            fetch_timestamps.clear()
            fetch_history_data.clear()
            fetch_symbol_history.clear()

            log_audit("Scan Completed", selected_universe, f"Saved {len(df)} records to unified history (ID: {scan_id})")

            # Ensure 'Actions' is available in display if selected
            # Note: selected_columns might contain 'Actions', so we ensure it exists in df (done above)
            display_cols = [c for c in selected_columns if c in df.columns]
            display_df = df[display_cols]

            st_column_config = get_column_config(display_cols, broker_choice)

            st.dataframe(display_df,use_container_width=True,height=600,column_config=st_column_config)

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

        else:
            st.warning("No data retrieved. Check internet or ticker config.")
            log_audit("Scan Failed", selected_universe, "No data retrieved")

    st.caption("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns | ATR SL | Analyst Dispersion | Full Logic")

# streamlit_app.py - v9.4 MASTER TERMINAL (Dynamic Columns + Heatmap Safety)
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta

# ---------------- CONFIG ----------------
try:
    from fortress_config import TICKER_GROUPS, SECTOR_MAP, INDEX_BENCHMARKS
except ImportError:
    st.error("Configuration file 'fortress_config.py' not found.")
    st.stop()

# Import logic from shared module
try:
    from fortress_logic import init_db, log_scan_results, check_institutional_fortress
except ImportError:
    st.error("Logic module 'fortress_logic.py' not found.")
    st.stop()

init_db()

# ---------------- UI ----------------
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns Terminal")

# Sidebar Controls
st.sidebar.title("💰 Portfolio & Risk")
portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", value=1000000, step=50000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.1)/100
selected_universe = st.sidebar.selectbox("Select Index", list(TICKER_GROUPS.keys()))

# ---------------- COLUMN CONFIG ----------------
ALL_COLUMNS = {
    "Symbol": {"label":"Symbol"},
    "Verdict": {"label":"Verdict"},
    "Score": {"label":"Conviction", "type":"progress", "min":0, "max":100},
    "Price": {"label":"Price ₹", "format":"₹%.2f"},
    "RSI": {"label":"RSI", "format":"%.1f"},
    "News": {"label":"News"},
    "Events": {"label":"Events"},
    "Sector": {"label":"Sector"},
    "Position_Qty": {"label":"Qty", "format":"%d"},
    "Stop_Loss": {"label":"SL Price", "format":"₹%.2f"},
    "Target_10D": {"label":"10D Target", "format":"₹%.2f"},
    "Analysts": {"label":"Analyst Count", "format":"%d"},
    "Tgt_High": {"label":"High Target", "format":"₹%d"},
    "Tgt_Median": {"label":"Median Target", "format":"₹%d"},
    "Tgt_Low": {"label":"Low Target", "format":"₹%d"},
    "Tgt_Mean": {"label":"Mean Target", "format":"₹%d"},
    "Dispersion_Alert": {"label":"Dispersion"},
    "Ret_30D": {"label":"30D Backtest", "format":"%.2f%%"},
    "Ret_60D": {"label":"60D Backtest", "format":"%.2f%%"},
    "Ret_90D": {"label":"90D Backtest", "format":"%.2f%%"}
}

# Sidebar Multiselect for Dynamic Columns
selected_columns = st.sidebar.multiselect(
    "Select Columns to Display", options=list(ALL_COLUMNS.keys()), default=list(ALL_COLUMNS.keys())
)

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
        df = pd.DataFrame(results).sort_values("Score",ascending=False)
        status_text.success(f"Scan Complete: {len(df[df['Score']>=60])} actionable setups.")
        log_scan_results(df)

        display_df = df[selected_columns]

        st_column_config = {}
        for col in selected_columns:
            cfg = ALL_COLUMNS[col]
            fmt = cfg.get("format")
            if cfg.get("type")=="progress":
                st_column_config[col] = st.column_config.ProgressColumn(cfg["label"],min_value=cfg["min"],max_value=cfg["max"])
            elif fmt:
                st_column_config[col] = st.column_config.NumberColumn(cfg["label"],format=fmt)
            else:
                st_column_config[col] = st.column_config.TextColumn(cfg["label"])

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

st.caption("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns | ATR SL | Analyst Dispersion | Full Logic")

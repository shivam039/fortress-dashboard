# fortress_app.py - v9.3 MASTER TERMINAL (Dynamic Columns)
import subprocess, sys, time, sqlite3
from datetime import datetime
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
try:
    from fortress_config import TICKER_GROUPS, SECTOR_MAP, INDEX_BENCHMARKS
except ImportError:
    st.error("Missing fortress_config.py! Ensure TICKER_GROUPS, SECTOR_MAP, INDEX_BENCHMARKS exist.")
    st.stop()

# ---------------- DB INIT ----------------
def init_db():
    conn = sqlite3.connect("fortress_history.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS scan_history 
                      (date TEXT, symbol TEXT, score REAL, verdict TEXT, price REAL, 
                       target_10d REAL, rsi REAL, analysts INTEGER, dispersion TEXT)''')
    conn.commit()
    conn.close()

def log_scan_results(df):
    conn = sqlite3.connect("fortress_history.db")
    today = datetime.now().strftime("%Y-%m-%d")
    for _, row in df.iterrows():
        conn.execute("INSERT INTO scan_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (today, row['Symbol'], row['Score'], row['Verdict'], row['Price'], 
                      row['Target_10D'], row['RSI'], row['Analysts'], row['Dispersion_Alert']))
    conn.commit()
    conn.close()

init_db()

# ---------------- UI ----------------
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v9.3 — Dynamic Columns Terminal")

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
    "Dispersion_Alert": {"label":"Dispersion"}
}

# Sidebar Multiselect for Dynamic Columns
selected_columns = st.sidebar.multiselect(
    "Select Columns to Display", options=list(ALL_COLUMNS.keys()), default=list(ALL_COLUMNS.keys())
)

# ---------------- CORE ENGINE ----------------
def check_institutional_fortress(ticker, data, ticker_obj, portfolio_value, risk_per_trade):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data) < 210: return None

        close, high, low = data["Close"], data["High"], data["Low"]

        ema200 = ta.ema(close, 200).iloc[-1]
        rsi = ta.rsi(close, 14).iloc[-1]
        atr = ta.atr(high, low, close, 14).iloc[-1]

        st_df = ta.supertrend(high, low, close, 10, 3)
        trend_col = [c for c in st_df.columns if c.startswith("SUPERTd")][0]
        trend_dir = int(st_df[trend_col].iloc[-1])
        price = float(close.iloc[-1])
        tech_base = price>ema200 and trend_dir==1

        sl_distance = atr*1.5
        sl_price = round(price-sl_distance,2)
        target_10d = round(price + atr*1.8,2)
        risk_amount = portfolio_value*risk_per_trade
        pos_size = int(risk_amount / sl_distance) if sl_distance>0 else 0

        conviction = 0
        score_mod = 0
        news_sentiment = "Neutral"
        event_status = "✅ Safe"

        try:
            news = ticker_obj.news or []
            titles = " ".join(n.get("title","").lower() for n in news[:5])
            if any(k in titles for k in ["fraud","investigation","default","bankruptcy","scam","legal"]):
                news_sentiment = "🚨 BLACK SWAN"
                score_mod -= 40
        except: pass
        try:
            cal = ticker_obj.calendar
            if isinstance(cal,pd.DataFrame) and not cal.empty:
                next_date = pd.to_datetime(cal.iloc[0,0]).date()
                days_to = (next_date - datetime.now().date()).days
                if 0<=days_to<=7:
                    event_status = f"🚨 EARNINGS ({next_date.strftime('%d-%b')})"
                    score_mod -= 20
        except: pass

        analyst_count = target_high = target_low = target_median = target_mean = 0
        try:
            info = ticker_obj.info or {}
            analyst_count = info.get("numberOfAnalystOpinions",0)
            target_high = info.get("targetHighPrice",0)
            target_low = info.get("targetLowPrice",0)
            target_median = info.get("targetMedianPrice",0)
            target_mean = info.get("targetMeanPrice",0)
        except: pass

        if tech_base:
            conviction += 60
            if 48<=rsi<=62: conviction+=20
            elif 40<=rsi<=72: conviction+=10
            conviction += score_mod

        dispersion_pct = ((target_high-target_low)/price)*100 if price>0 else 0
        dispersion_alert = "⚠️ High Dispersion" if dispersion_pct>30 else "✅"
        if dispersion_pct>30: conviction -= 10

        conviction = max(0,min(100,conviction))
        verdict = "🔥 HIGH" if conviction>=85 else "🚀 PASS" if conviction>=60 else "🟡 WATCH" if tech_base else "❌ FAIL"

        return {
            "Symbol": ticker,
            "Verdict": verdict,
            "Score": conviction,
            "Price": round(price,2),
            "RSI": round(rsi,1),
            "News": news_sentiment,
            "Events": event_status,
            "Sector": SECTOR_MAP.get(ticker,"General"),
            "Position_Qty": pos_size,
            "Stop_Loss": sl_price,
            "Target_10D": target_10d,
            "Analysts": analyst_count,
            "Tgt_High": target_high,
            "Tgt_Median": target_median,
            "Tgt_Low": target_low,
            "Tgt_Mean": target_mean,
            "Dispersion_Alert": dispersion_alert
        }
    except: return None

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

        # Dynamic column selection
        display_df = df[selected_columns]

        # Dynamic column_config for Streamlit
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

        st.subheader("📊 Conviction Heatmap")
        df["Conviction_Band"] = df["Score"].apply(lambda x: "🔥 High (85+)" if x>=85 else "🚀 Pass (60-85)" if x>=60 else "🟡 Watch (<60)")
        plt.figure(figsize=(12,len(df)/2))
        heatmap_data = df.pivot_table(index="Symbol", columns="Conviction_Band", values="Score", fill_value=0)
        sns.heatmap(heatmap_data, annot=True, cmap="Greens", cbar=False, linewidths=0.5, linecolor='grey')
        st.pyplot(plt)

    else:
        st.warning("No data retrieved. Check internet or ticker config.")

st.caption("🛡️ Fortress 95 Pro v9.3 — Dynamic Columns | ATR SL | Analyst Dispersion | Full Logic")

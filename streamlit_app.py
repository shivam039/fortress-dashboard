# fortress_app.py - v8.0 MASTER PRODUCTION FILE
import subprocess, sys, time, sqlite3
from datetime import datetime
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np

# Import config (Ensure these variables exist in your fortress_config.py)
try:
    from fortress_config import TICKER_GROUPS, SECTOR_MAP, INDEX_BENCHMARKS
except ImportError:
    st.error("Missing fortress_config.py! Please ensure ticker lists are defined.")
    st.stop()

# ---------------- DEPENDENCIES ----------------
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

# ---------------- DB INITIALIZATION ----------------
def init_db():
    conn = sqlite3.connect("fortress_history.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS scan_history 
                      (date TEXT, symbol TEXT, score REAL, verdict TEXT, price REAL, target_10d REAL)''')
    conn.commit()
    conn.close()

def log_scan_results(df):
    conn = sqlite3.connect("fortress_history.db")
    today = datetime.now().strftime("%Y-%m-%d")
    for _, row in df.iterrows():
        conn.execute("INSERT INTO scan_history VALUES (?, ?, ?, ?, ?, ?)",
                     (today, row['Symbol'], row['Score'], row['Verdict'], row['Price'], row['Target_10D']))
    conn.commit()
    conn.close()

init_db()

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v8.0 — Production Terminal")

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.title("💰 Portfolio & Risk")
portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", value=1000000, step=50000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.1) / 100
selected_universe = st.sidebar.selectbox("Select Index", list(TICKER_GROUPS.keys()))

# ================= CORE ENGINE =================
def check_institutional_fortress(ticker, data, ticker_obj, portfolio_value, risk_per_trade):
    try:
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if len(data) < 210: return None

        close, high, low = data["Close"], data["High"], data["Low"]
        
        # 1. Technical Indicators
        ema200 = ta.ema(close, 200).iloc[-1]
        rsi = ta.rsi(close, 14).iloc[-1]
        atr = ta.atr(high, low, close, length=14).iloc[-1]
        
        st_df = ta.supertrend(high, low, close, 10, 3)
        trend_col = [c for c in st_df.columns if c.startswith("SUPERTd")][0]
        trend_dir = int(st_df[trend_col].iloc[-1])

        price = float(close.iloc[-1])
        tech_base = price > ema200 and trend_dir == 1

        # 2. Risk Management (ATR Logic)
        sl_distance = atr * 1.5
        sl_price = round(price - sl_distance, 2)
        target_10d = round(price + (atr * 1.8), 2)
        
        risk_amount = portfolio_value * risk_per_trade
        pos_size = int(risk_amount / sl_distance) if sl_distance > 0 else 0

        # 3. Guardrails (News & Earnings)
        conviction = 0
        score_mod = 0
        news_sentiment = "Neutral"
        event_status = "✅ Safe"

        # News Sentinel
        try:
            news = ticker_obj.news or []
            titles = " ".join(n.get("title", "").lower() for n in news[:5])
            if any(k in titles for k in ["fraud", "investigation", "default", "bankruptcy", "scam", "legal"]):
                news_sentiment = "🚨 BLACK SWAN"
                score_mod -= 40
        except: pass

        # Earnings Blocker
        try:
            cal = ticker_obj.calendar
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                next_date = pd.to_datetime(cal.iloc[0, 0]).date()
                days_to = (next_date - datetime.now().date()).days
                if 0 <= days_to <= 7:
                    event_status = f"🚨 EARNINGS ({next_date.strftime('%d-%b')})"
                    score_mod -= 20
        except: pass

        # 4. Scoring logic
        if tech_base:
            conviction += 60
            if 48 <= rsi <= 62: conviction += 20
            elif 40 <= rsi <= 72: conviction += 10
            conviction += score_mod
        
        conviction = max(0, min(100, conviction))
        verdict = "🔥 HIGH" if conviction >= 85 else "🚀 PASS" if conviction >= 60 else "🟡 WATCH" if tech_base else "❌ FAIL"

        return {
            "Symbol": ticker,
            "Verdict": verdict,
            "Score": conviction,
            "Price": round(price, 2),
            "Shares_to_Buy": pos_size,
            "Stop_Loss": sl_price,
            "Target_10D": target_10d,
            "RSI": round(rsi, 1),
            "News": news_sentiment,
            "Events": event_status,
            "Sector": SECTOR_MAP.get(ticker, "General")
        }
    except Exception as e:
        return None

# ================= MAIN EXECUTION =================
st.subheader("🌐 Market Condition")
pulse_cols = st.columns(len(INDEX_BENCHMARKS))
for i, (name, symbol) in enumerate(INDEX_BENCHMARKS.items()):
    try:
        idx_data = yf.download(symbol, period="1y", progress=False)
        p_close = idx_data["Close"].iloc[-1]
        p_ema = ta.ema(idx_data["Close"], 200).iloc[-1]
        p_status = "🟢 BULL" if p_close > p_ema else "🔴 BEAR"
        pulse_cols[i].metric(name, f"{p_close:,.0f}", p_status)
    except: pass

if st.button("🚀 EXECUTE SYSTEM SCAN", type="primary", use_container_width=True):
    tickers = TICKER_GROUPS[selected_universe]
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanning {ticker}... ({i+1}/{len(tickers)})")
        try:
            tkr = yf.Ticker(ticker)
            hist = yf.download(ticker, period="2y", progress=False)
            if not hist.empty:
                res = check_institutional_fortress(ticker, hist, tkr, portfolio_val, risk_pct)
                if res: results.append(res)
            time.sleep(0.7) # Rate limit protection
        except: pass
        progress_bar.progress((i + 1) / len(tickers))

    if results:
        df = pd.DataFrame(results).sort_values("Score", ascending=False)
        status_text.success(f"Scan Complete: {len(df[df['Score'] >= 60])} actionable setups.")
        
        # Logging to local DB for backtesting
        log_scan_results(df)
        
        st.dataframe(
            df, 
            use_container_width=True, 
            height=500,
            column_config={
                "Score": st.column_config.ProgressColumn("Conviction", min_value=0, max_value=100),
                "Shares_to_Buy": st.column_config.NumberColumn("Quantity", format="%d"),
                "Stop_Loss": st.column_config.NumberColumn("SL Price", format="₹%.2f"),
                "Target_10D": st.column_config.NumberColumn("10D Target", format="₹%.2f")
            }
        )
        
        # CSV Export for Broker
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Trades to CSV", data=csv, 
                           file_name=f"Fortress_Trades_{datetime.now().strftime('%Y%m%d')}.csv", 
                           mime="text/csv", use_container_width=True)
    else:
        st.warning("No data retrieved. Check internet or ticker config.")

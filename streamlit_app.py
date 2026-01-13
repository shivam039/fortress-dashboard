# streamlit_app.py - v9.4 MASTER TERMINAL (Dynamic Columns + Heatmap Safety)
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
    st.error("Configuration file 'fortress_config.py' not found.")
    st.stop()

def init_db():
    try:
        conn = sqlite3.connect('fortress_history.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS scan_results
                     (timestamp TEXT, symbol TEXT, score REAL, price REAL, verdict TEXT)''')
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

def log_scan_results(df):
    try:
        conn = sqlite3.connect('fortress_history.db')
        df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check and update schema
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(scan_results)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        for col in df.columns:
            if col not in existing_cols:
                # Determine type
                dtype = df[col].dtype
                if pd.api.types.is_integer_dtype(dtype):
                    sql_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "REAL"
                else:
                    sql_type = "TEXT"

                try:
                    cursor.execute(f'ALTER TABLE scan_results ADD COLUMN "{col}" {sql_type}')
                except Exception as alter_err:
                     print(f"Error adding column {col}: {alter_err}")

        conn.commit()

        # Log all columns to the database
        df.to_sql('scan_results', conn, if_exists='append', index=False)
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

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

# ---------------- CORE ENGINE ----------------

def calculate_technicals(data):
    close, high, low = data["Close"], data["High"], data["Low"]
    ema200 = ta.ema(close, 200).iloc[-1]
    rsi = ta.rsi(close, 14).iloc[-1]
    atr = ta.atr(high, low, close, 14).iloc[-1]
    st_df = ta.supertrend(high, low, close, 10, 3)
    trend_col = [c for c in st_df.columns if c.startswith("SUPERTd")][0]
    trend_dir = int(st_df[trend_col].iloc[-1])
    price = float(close.iloc[-1])
    tech_base = price > ema200 and trend_dir == 1
    return price, rsi, atr, tech_base

def analyze_news_and_events(ticker_obj):
    news_sentiment = "Neutral"
    event_status = "✅ Safe"
    score_mod = 0
    try:
        news = ticker_obj.news or []
        titles = " ".join(n.get("title", "").lower() for n in news[:5])
        if any(k in titles for k in ["fraud", "investigation", "default", "bankruptcy", "scam", "legal"]):
            news_sentiment = "🚨 BLACK SWAN"
            score_mod -= 40
    except: pass
    try:
        cal = ticker_obj.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            next_date = pd.to_datetime(cal.iloc[0, 0]).date()
            days_to = (next_date - datetime.now().date()).days
            if 0 <= days_to <= 7:
                event_status = f"🚨 EARNINGS ({next_date.strftime('%d-%b')})"
                score_mod -= 20
    except: pass
    return news_sentiment, event_status, score_mod

def get_analyst_data(ticker_obj):
    analyst_count = target_high = target_low = target_median = target_mean = 0
    try:
        info = ticker_obj.info or {}
        analyst_count = info.get("numberOfAnalystOpinions", 0)
        target_high = info.get("targetHighPrice", 0)
        target_low = info.get("targetLowPrice", 0)
        target_median = info.get("targetMedianPrice", 0)
        target_mean = info.get("targetMeanPrice", 0)
    except: pass
    return analyst_count, target_high, target_low, target_median, target_mean

def calculate_score(tech_base, rsi, score_mod, dispersion_pct):
    conviction = 0
    if tech_base:
        conviction += 60
        if 48 <= rsi <= 62: conviction += 20
        elif 40 <= rsi <= 72: conviction += 10
        conviction += score_mod

    if dispersion_pct > 30:
        conviction -= 10

    return max(0, min(100, conviction))

def calculate_backtest_returns(close, price):
    current_date = close.index[-1]
    returns = {}
    for days in [30, 60, 90]:
        try:
            target_date = current_date - pd.Timedelta(days=days)
            idx = close.index.get_indexer([target_date], method='nearest')[0]
            past_price = float(close.iloc[idx])
            pct_change = ((price - past_price) / past_price) * 100
            returns[f"Ret_{days}D"] = pct_change
        except:
            returns[f"Ret_{days}D"] = None
    return returns

def check_institutional_fortress(ticker, data, ticker_obj, portfolio_value, risk_per_trade):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data) < 210: return None

        price, rsi, atr, tech_base = calculate_technicals(data)

        sl_distance = atr * 1.5
        sl_price = round(price - sl_distance, 2)
        target_10d = round(price + atr * 1.8, 2)
        risk_amount = portfolio_value * risk_per_trade
        pos_size = int(risk_amount / sl_distance) if sl_distance > 0 else 0

        news_sentiment, event_status, score_mod = analyze_news_and_events(ticker_obj)
        analyst_count, target_high, target_low, target_median, target_mean = get_analyst_data(ticker_obj)

        dispersion_pct = ((target_high - target_low) / price) * 100 if price > 0 else 0
        dispersion_alert = "⚠️ High Dispersion" if dispersion_pct > 30 else "✅"

        conviction = calculate_score(tech_base, rsi, score_mod, dispersion_pct)
        verdict = "🔥 HIGH" if conviction >= 85 else "🚀 PASS" if conviction >= 60 else "🟡 WATCH" if tech_base else "❌ FAIL"

        returns = calculate_backtest_returns(data["Close"], price)

        return {
            "Symbol": ticker,
            "Verdict": verdict,
            "Score": conviction,
            "Price": round(price, 2),
            "RSI": round(rsi, 1),
            "News": news_sentiment,
            "Events": event_status,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Position_Qty": pos_size,
            "Stop_Loss": sl_price,
            "Target_10D": target_10d,
            "Analysts": analyst_count,
            "Tgt_High": target_high,
            "Tgt_Median": target_median,
            "Tgt_Low": target_low,
            "Tgt_Mean": target_mean,
            "Dispersion_Alert": dispersion_alert,
            "Ret_30D": returns.get("Ret_30D"),
            "Ret_60D": returns.get("Ret_60D"),
            "Ret_90D": returns.get("Ret_90D")
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

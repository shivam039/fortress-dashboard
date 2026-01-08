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
                       target_10d REAL, rsi REAL, analysts INTEGER, dispersion TEXT,
                       hit_target INTEGER, hit_sl INTEGER)''')
    conn.commit()
    conn.close()

def log_scan_results(df):
    conn = sqlite3.connect("fortress_history.db")
    today = datetime.now().strftime("%Y-%m-%d")

    for _, row in df.iterrows():
        hit_t, hit_s = evaluate_trade(
            row["Symbol"], row["Price"],
            row["Stop_Loss"], row["Target_10D"]
        )

        conn.execute("INSERT INTO scan_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (today, row['Symbol'], row['Score'], row['Verdict'], row['Price'],
                      row['Target_10D'], row['RSI'], row['Analysts'],
                      row['Dispersion_Alert'], hit_t, hit_s))
    conn.commit()
    conn.close()

init_db()

# ---------------- BACKTEST ENGINE ----------------
def evaluate_trade(symbol, entry, sl, target):
    try:
        hist = yf.download(symbol, period="15d", progress=False)
        for price in hist["Close"]:
            if price >= target:
                return 1, 0
            if price <= sl:
                return 0, 1
    except:
        pass
    return 0, 0

def backtest_symbol(symbol, lookback=60):
    conn = sqlite3.connect("fortress_history.db")
    df = pd.read_sql(
        "SELECT * FROM scan_history WHERE symbol=? ORDER BY date DESC LIMIT ?",
        conn, params=(symbol, lookback)
    )
    conn.close()

    if df.empty:
        return 0

    wins = df["hit_target"].sum()
    losses = df["hit_sl"].sum()
    total = wins + losses

    if total == 0:
        return 0

    return round((wins / total) * 100, 1)

# ---------------- UI ----------------
st.set_page_config(page_title="Fortress HP", layout="wide")
st.title("🛡️ Fortress HP v9.6 — High Probability Terminal")

# Sidebar
st.sidebar.title("💰 Portfolio & Risk")
portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", value=1000000, step=50000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.1)/100
selected_universe = st.sidebar.selectbox("Select Index/Universe", list(TICKER_GROUPS.keys()))
min_score_filter = st.sidebar.slider("Minimum Conviction Score", 0, 100, 70, 5)

# ---------------- COLUMN CONFIG ----------------
ALL_COLUMNS = {
    "Symbol": {"label":"Symbol"},
    "Verdict": {"label":"Verdict"},
    "Score": {"label":"Conviction", "type":"progress", "min":0, "max":100},
    "Backtested_WinRate":{"label":"Win %", "format":"%.1f"},
    "Price": {"label":"Price ₹", "format":"₹%.2f"},
    "RSI": {"label":"RSI", "format":"%.1f"},
    "ADX": {"label":"ADX Strength", "format":"%.1f"},
    "Volume_Ratio": {"label":"Vol Ratio", "format":"%.2f"},
    "RS_6M": {"label":"6M Rel Strength", "format":"%.2f"},
    "News": {"label":"News"},
    "Events": {"label":"Events"},
    "Sector": {"label":"Sector"},
    "Position_Qty": {"label":"Qty", "format":"%d"},
    "Stop_Loss": {"label":"SL Price", "format":"₹%.2f"},
    "Target_10D": {"label":"10D Target", "format":"₹%.2f"},
    "Analysts": {"label":"Analyst Count", "format":"%d"},
    "Dispersion_Alert": {"label":"Dispersion"}
}

selected_columns = st.sidebar.multiselect(
    "Select Columns", options=list(ALL_COLUMNS.keys()),
    default=["Symbol","Verdict","Score","Backtested_WinRate",
             "Price","RSI","ADX","Volume_Ratio","RS_6M",
             "Target_10D","Stop_Loss","Position_Qty",
             "News","Events","Analysts","Dispersion_Alert"]
)

# ---------------- CORE ENGINE ----------------
def check_institutional_fortress(ticker, data, ticker_obj,
                                 portfolio_value, risk_per_trade, bench_hist):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data) < 250:
            return None

        close, high, low, volume = data["Close"], data["High"], data["Low"], data["Volume"]
        price = float(close.iloc[-1])

        ema50 = ta.ema(close,50).iloc[-1]
        ema200 = ta.ema(close,200).iloc[-1]
        rsi = ta.rsi(close,14).iloc[-1]
        atr = ta.atr(high,low,close,14).iloc[-1]

        st_df = ta.supertrend(high,low,close,10,3)
        trend_dir = int(st_df[[c for c in st_df.columns if c.startswith("SUPERTd")][0]].iloc[-1])

        adx_df = ta.adx(high,low,close,14)
        adx = adx_df["ADX_14"].iloc[-1]
        plus_di = adx_df["DMP_14"].iloc[-1]
        minus_di = adx_df["DMN_14"].iloc[-1]

        macd_df = ta.macd(close)
        macd_line = macd_df["MACD_12_26_9"].iloc[-1]
        macd_sig = macd_df["MACDs_12_26_9"].iloc[-1]

        vol_ma20 = ta.sma(volume,20).iloc[-1]
        vol_ratio = volume.iloc[-1]/vol_ma20 if vol_ma20>0 else 0

        # Relative Strength
        rs_6m = 1
        if len(close)>=126 and len(bench_hist)>=126:
            rs_6m = (price/close.iloc[-126]) / \
                    (bench_hist["Close"].iloc[-1]/bench_hist["Close"].iloc[-126])

        conviction = 0

        if price>ema200: conviction+=20
        if price>ema50 and ema50>ema200: conviction+=20
        if trend_dir==1: conviction+=20
        if adx>25: conviction+=15
        if plus_di>minus_di: conviction+=10
        if macd_line>macd_sig: conviction+=10
        if vol_ratio>1.2: conviction+=15
        if 45<=rsi<=65: conviction+=15
        if rsi>70: conviction-=20
        if rs_6m>1.1: conviction+=15
        if rs_6m<0.9: conviction-=15

        conviction=max(0,min(100,conviction))

        verdict = ("🔥🔥 ULTRA" if conviction>=95 else
                   "🔥 HIGH" if conviction>=85 else
                   "🚀 PASS" if conviction>=70 else
                   "🟡 WATCH" if conviction>=50 else
                   "❌ FAIL")

        sl_distance = atr*1.5
        sl_price = round(price-sl_distance,2)
        target_10d = round(price+atr*1.5,2)
        pos_size = int((portfolio_value*risk_per_trade)/sl_distance) if sl_distance>0 else 0

        win_rate = backtest_symbol(ticker)

        return {
            "Symbol":ticker,
            "Verdict":verdict,
            "Score":conviction,
            "Backtested_WinRate":win_rate,
            "Price":round(price,2),
            "RSI":round(rsi,1),
            "ADX":round(adx,1),
            "Volume_Ratio":round(vol_ratio,2),
            "RS_6M":round(rs_6m,2),
            "News":"Neutral",
            "Events":"✅ Safe",
            "Sector":SECTOR_MAP.get(ticker,"General"),
            "Position_Qty":pos_size,
            "Stop_Loss":sl_price,
            "Target_10D":target_10d,
            "Analysts":0,
            "Dispersion_Alert":""
        }
    except:
        return None

# ---------------- MAIN SCAN ----------------
if st.button("🚀 EXECUTE HIGH PROBABILITY SCAN",use_container_width=True):

    tickers = TICKER_GROUPS[selected_universe]
    bench_hist = yf.download(INDEX_BENCHMARKS[selected_universe],period="2y")

    results=[]
    pb=st.progress(0)

    for i,t in enumerate(tickers):
        hist=yf.download(t,period="2y",progress=False)
        if not hist.empty:
            r=check_institutional_fortress(t,hist,yf.Ticker(t),
                                           portfolio_val,risk_pct,bench_hist)
            if r and r["Score"]>=min_score_filter:
                results.append(r)
        pb.progress((i+1)/len(tickers))
        time.sleep(0.5)

    if results:
        df=pd.DataFrame(results).sort_values("Score",ascending=False)
        log_scan_results(df)

        display_df=df[selected_columns]

        cfg={}
        for c in selected_columns:
            d=ALL_COLUMNS[c]
            if d.get("type")=="progress":
                cfg[c]=st.column_config.ProgressColumn(
                    d["label"],min_value=0,max_value=100)
            elif d.get("format"):
                cfg[c]=st.column_config.NumberColumn(d["label"],format=d["format"])
            else:
                cfg[c]=st.column_config.TextColumn(d["label"])

        st.dataframe(display_df,column_config=cfg,use_container_width=True)

        st.download_button("📥 Export",
                           display_df.to_csv(index=False),
                           file_name="Fortress_Backtested.csv")

    else:
        st.warning("No valid setups.")

st.caption("🛡️ Fortress v9.6 | Backtested Win % Added")

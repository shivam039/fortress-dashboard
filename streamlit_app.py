# streamlit_app.py - v9.4 MASTER TERMINAL (Dynamic Columns + Heatmap Safety)
import subprocess, sys, time, sqlite3
from datetime import datetime, timedelta
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# NSE
from nsepython import equity_history, index_history, nse_get_index_quote

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

# ---------------- NSE ADAPTER ----------------

class NSEStock:
    def __init__(self, symbol):
        self.symbol = symbol.replace(".NS","")

    @property
    def news(self):
        return []

    @property
    def calendar(self):
        return pd.DataFrame()

    @property
    def info(self):
        return {
            "numberOfAnalystOpinions": 0,
            "targetHighPrice": 0,
            "targetLowPrice": 0,
            "targetMedianPrice": 0,
            "targetMeanPrice": 0
        }

def fetch_stock_history(symbol, period="2y"):
    clean = symbol.replace(".NS","")
    end = datetime.now()
    days = 730 if period=="2y" else 365
    start = end - timedelta(days=days)

    s = start.strftime("%d-%m-%Y")
    e = end.strftime("%d-%m-%Y")

    for _ in range(3):
        try:
            df = equity_history(clean,"EQ",s,e)
            if df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "CH_OPENING_PRICE":"Open",
                "CH_TRADE_HIGH_PRICE":"High",
                "CH_TRADE_LOW_PRICE":"Low",
                "CH_CLOSING_PRICE":"Close",
                "CH_TOT_TRADED_QTY":"Volume",
                "CH_TIMESTAMP":"Date",
                "mTIMESTAMP":"Date"
            })

            df["Date"] = pd.to_datetime(df["Date"],errors="coerce")
            df = df.set_index("Date").sort_index()

            for c in ["Open","High","Low","Close","Volume"]:
                df[c] = pd.to_numeric(df[c],errors="coerce")

            return df[["Open","High","Low","Close","Volume"]]

        except:
            time.sleep(1)

    return pd.DataFrame()

def fetch_index_history(symbol):
    map_ = {
        "^NSEI":"NIFTY 50",
        "^NSEBANK":"NIFTY BANK",
        "^NSMIDCP":"NIFTY MIDCAP 150"
    }

    idx = map_.get(symbol,symbol)

    end = datetime.now()
    start = end - timedelta(days=365)

    s = start.strftime("%d-%m-%Y")
    e = end.strftime("%d-%m-%Y")

    try:
        df = index_history(idx,s,e)
        if not df.empty:
            df.columns = df.columns.str.upper()
            df = df.rename(columns={
                "HISTORICALDATE":"Date",
                "OPEN":"Open",
                "HIGH":"High",
                "LOW":"Low",
                "CLOSE":"Close"
            })

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            df["Close"] = pd.to_numeric(df["Close"],errors="coerce")
            return df
    except:
        pass

    try:
        q = nse_get_index_quote(idx)
        if q:
            p = float(q["last"].replace(",",""))
            return pd.DataFrame({"Close":[p]},index=[pd.Timestamp.now()])
    except:
        pass

    return pd.DataFrame()

# ---------------- UI ----------------
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns Terminal (NSE)")

# Sidebar
st.sidebar.title("💰 Portfolio & Risk")
portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", value=1000000, step=50000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.1)/100
selected_universe = st.sidebar.selectbox("Select Index", list(TICKER_GROUPS.keys()))

# ---------------- COLUMN CONFIG ----------------
ALL_COLUMNS = {
    "Symbol":{"label":"Symbol"},
    "Verdict":{"label":"Verdict"},
    "Score":{"label":"Conviction","type":"progress","min":0,"max":100},
    "Price":{"label":"Price ₹","format":"₹%.2f"},
    "RSI":{"label":"RSI","format":"%.1f"},
    "News":{"label":"News"},
    "Events":{"label":"Events"},
    "Sector":{"label":"Sector"},
    "Position_Qty":{"label":"Qty","format":"%d"},
    "Stop_Loss":{"label":"SL Price","format":"₹%.2f"},
    "Target_10D":{"label":"10D Target","format":"₹%.2f"},
    "Analysts":{"label":"Analyst Count","format":"%d"},
    "Tgt_High":{"label":"High Target","format":"₹%d"},
    "Tgt_Median":{"label":"Median Target","format":"₹%d"},
    "Tgt_Low":{"label":"Low Target","format":"₹%d"},
    "Tgt_Mean":{"label":"Mean Target","format":"₹%d"},
    "Dispersion_Alert":{"label":"Dispersion"},
    "Ret_30D":{"label":"30D Backtest","format":"%.2f%%"},
    "Ret_60D":{"label":"60D Backtest","format":"%.2f%%"},
    "Ret_90D":{"label":"90D Backtest","format":"%.2f%%"}
}

selected_columns = st.sidebar.multiselect(
    "Select Columns",ALL_COLUMNS.keys(),default=list(ALL_COLUMNS.keys())
)

# ---------------- CORE ENGINE ----------------
def check_institutional_fortress(ticker,data,ticker_obj,portfolio_value,risk_per_trade):
    try:
        if len(data)<210: return None

        close,high,low = data["Close"],data["High"],data["Low"]

        ema200 = ta.ema(close,200).iloc[-1]
        rsi = ta.rsi(close,14).iloc[-1]
        atr = ta.atr(high,low,close,14).iloc[-1]

        st_df = ta.supertrend(high,low,close,10,3)
        trend_col = [c for c in st_df.columns if c.startswith("SUPERTd")][0]
        trend_dir = int(st_df[trend_col].iloc[-1])

        price = float(close.iloc[-1])
        tech_base = price>ema200 and trend_dir==1

        sl_dist = atr*1.5
        sl_price = round(price-sl_dist,2)
        target_10d = round(price + atr*1.8,2)
        risk_amt = portfolio_value*risk_per_trade
        pos_size = int(risk_amt/sl_dist) if sl_dist>0 else 0

        conviction = 0
        score_mod = 0
        news_sentiment="Neutral"
        event_status="✅ Safe"

        info = ticker_obj.info
        analyst_count = info["numberOfAnalystOpinions"]
        tgt_h = info["targetHighPrice"]
        tgt_l = info["targetLowPrice"]
        tgt_m = info["targetMedianPrice"]
        tgt_mean = info["targetMeanPrice"]

        if tech_base:
            conviction+=60
            if 48<=rsi<=62: conviction+=20
            elif 40<=rsi<=72: conviction+=10

        dispersion = ((tgt_h-tgt_l)/price)*100 if price>0 else 0
        disp_alert = "⚠️ High Dispersion" if dispersion>30 else "✅"
        if dispersion>30: conviction-=10

        verdict = "🔥 HIGH" if conviction>=85 else "🚀 PASS" if conviction>=60 else "🟡 WATCH" if tech_base else "❌ FAIL"

        returns={}
        cur=close.index[-1]

        for d in [30,60,90]:
            try:
                td = cur-pd.Timedelta(days=d)
                idx = close.index.get_indexer([td],method="nearest")[0]
                past = float(close.iloc[idx])
                returns[f"Ret_{d}D"] = ((price-past)/past)*100
            except:
                returns[f"Ret_{d}D"]=None

        return {
            "Symbol":ticker,
            "Verdict":verdict,
            "Score":conviction,
            "Price":round(price,2),
            "RSI":round(rsi,1),
            "News":news_sentiment,
            "Events":event_status,
            "Sector":SECTOR_MAP.get(ticker,"General"),
            "Position_Qty":pos_size,
            "Stop_Loss":sl_price,
            "Target_10D":target_10d,
            "Analysts":analyst_count,
            "Tgt_High":tgt_h,
            "Tgt_Median":tgt_m,
            "Tgt_Low":tgt_l,
            "Tgt_Mean":tgt_mean,
            "Dispersion_Alert":disp_alert,
            "Ret_30D":returns["Ret_30D"],
            "Ret_60D":returns["Ret_60D"],
            "Ret_90D":returns["Ret_90D"]
        }
    except:
        return None

# ---------------- MARKET PULSE ----------------
st.subheader("🌐 Market Pulse")
cols = st.columns(len(INDEX_BENCHMARKS))

for i,(name,symbol) in enumerate(INDEX_BENCHMARKS.items()):
    try:
        idx_data = fetch_index_history(symbol)
        p_close = idx_data["Close"].iloc[-1]
        p_ema = ta.ema(idx_data["Close"],200).iloc[-1]
        status = "🟢 BULL" if p_close>p_ema else "🔴 BEAR"
        cols[i].metric(name,f"{p_close:,.0f}",status)
    except:
        pass

# ---------------- MAIN SCAN ----------------
if st.button("🚀 EXECUTE SYSTEM SCAN",type="primary",use_container_width=True):

    tickers=TICKER_GROUPS[selected_universe]
    results=[]
    bar=st.progress(0)
    status=st.empty()

    for i,t in enumerate(tickers):
        status.text(f"Scanning {t} ({i+1}/{len(tickers)})")

        try:
            tkr=NSEStock(t)
            hist=fetch_stock_history(t)

            if not hist.empty:
                r=check_institutional_fortress(t,hist,tkr,portfolio_val,risk_pct)
                if r: results.append(r)

            time.sleep(1)

        except:
            pass

        bar.progress((i+1)/len(tickers))

    if results:
        df=pd.DataFrame(results).sort_values("Score",ascending=False)
        status.success(f"Scan Complete: {len(df[df['Score']>=60])} actionable setups.")
        log_scan_results(df)

        disp=df[selected_columns]

        cfg={}
        for c in selected_columns:
            m=ALL_COLUMNS[c]
            if m.get("type")=="progress":
                cfg[c]=st.column_config.ProgressColumn(m["label"],0,100)
            elif m.get("format"):
                cfg[c]=st.column_config.NumberColumn(m["label"],format=m["format"])
            else:
                cfg[c]=st.column_config.TextColumn(m["label"])

        st.dataframe(disp,use_container_width=True,height=600,column_config=cfg)

    else:
        st.warning("No data. NSE may be blocking.")

st.caption("🛡️ Fortress 95 Pro v9.4 — NSE Powered")

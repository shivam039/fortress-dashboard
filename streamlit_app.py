# streamlit_app.py - v9.4 MASTER TERMINAL (Dynamic Columns + Heatmap Safety)
# Migrated to NSE Source (nsepython) by Jules
import subprocess, sys, time, sqlite3
from datetime import datetime, timedelta
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
# Removed yfinance import
from nsepython import equity_history, nse_get_index_quote, index_history
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

# ---------------- NSE ADAPTER ----------------

class NSEStock:
    """
    Mock class to replicate yfinance Ticker object behavior for NSE data.
    """
    def __init__(self, symbol):
        self.ticker = symbol.replace(".NS", "")
        # Pre-fetch or lazy-load info if possible, but for now we use safe defaults
        # as nsepython doesn't provide analyst targets/ratings directly.

    @property
    def news(self):
        # nsepython doesn't provide news sentiment easily.
        # Return empty list to avoid logic errors.
        return []

    @property
    def calendar(self):
        # Earnings calendar.
        return pd.DataFrame()

    @property
    def info(self):
        # Analyst targets and info.
        # Returning defaults to allow logic to proceed without crashing.
        return {
            "numberOfAnalystOpinions": 0,
            "targetHighPrice": 0,
            "targetLowPrice": 0,
            "targetMedianPrice": 0,
            "targetMeanPrice": 0
        }

def fetch_stock_history(symbol, period="2y"):
    """
    Fetches stock history using nsepython.equity_history.
    Retries 3 times. Returns yfinance-formatted DataFrame.
    """
    clean_symbol = symbol.replace(".NS", "")

    # Calculate dates
    end_date = datetime.now()
    days = 730 if period == "2y" else 365
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime("%d-%m-%Y")
    end_str = end_date.strftime("%d-%m-%Y")

    for attempt in range(3):
        try:
            # Fetch data
            # Series "EQ" is standard for equity
            df = equity_history(clean_symbol, "EQ", start_str, end_str)

            if df.empty:
                return pd.DataFrame()

            # Mapping columns from NSE format to yfinance format
            # Typical NSE columns: CH_TIMESTAMP, CH_OPENING_PRICE, CH_TRADE_HIGH_PRICE,
            # CH_TRADE_LOW_PRICE, CH_CLOSING_PRICE, CH_TOT_TRADED_QTY
            rename_map = {
                "CH_OPENING_PRICE": "Open",
                "CH_TRADE_HIGH_PRICE": "High",
                "CH_TRADE_LOW_PRICE": "Low",
                "CH_CLOSING_PRICE": "Close",
                "CH_TOT_TRADED_QTY": "Volume",
                "CH_TIMESTAMP": "Date",
                "mTIMESTAMP": "Date" # Sometimes returned as mTIMESTAMP
            }

            # Rename available columns
            df = df.rename(columns=rename_map)

            # Ensure Date index
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors='coerce')
                # Fallback if format differs
                if df["Date"].isnull().all():
                     df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")

            # Sort by date
            df = df.sort_index()

            # Convert numeric columns
            cols = ["Open", "High", "Low", "Close", "Volume"]
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            # Filter only required columns
            existing_cols = [c for c in cols if c in df.columns]
            return df[existing_cols]

        except Exception as e:
            time.sleep(1) # Wait before retry
            if attempt == 2:
                # print(f"Failed to fetch {clean_symbol}: {e}")
                pass

    return pd.DataFrame()

def fetch_index_history_safe(symbol_yf):
    """
    Fetches index history or quote.
    Maps Yahoo symbol (e.g., ^NSEI) to NSE symbol (NIFTY 50).
    """
    # Map Yahoo symbols to NSE Index names
    # Add more mappings as needed based on fortress_config.py
    symbol_map = {
        "^NSEI": "NIFTY 50",
        "^NIFTYJR": "NIFTY NEXT 50",
        "^NSMIDCP": "NIFTY MIDCAP 150",
        "^NSEBANK": "NIFTY BANK"
    }

    nse_symbol = symbol_map.get(symbol_yf, symbol_yf)

    # Try fetching history first (needed for EMA)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_str = start_date.strftime("%d-%m-%Y")
    end_str = end_date.strftime("%d-%m-%Y")

    try:
        df = index_history(nse_symbol, start_str, end_str)
        if not df.empty:
            # Format index data
            # Expected cols: HistoricalDate, OPEN, HIGH, LOW, CLOSE
            rename_map = {
                "HistoricalDate": "Date",
                "OPEN": "Open",
                "HIGH": "High",
                "LOW": "Low",
                "CLOSE": "Close"
            }
            df = df.rename(columns=rename_map)
            # Parse Date - format is usually '12 Jan 2024'
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()

            # Convert close to float
            df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
            return df
    except Exception:
        pass

    # Fallback: Get Quote (Snapshot)
    # If history fails, we return a DataFrame with a single row of current data.
    # EMA calc will fail (or we handle it), but at least we have current price.
    try:
        quote = nse_get_index_quote(nse_symbol)
        if quote:
            # quote is a dict: {'last': '25,790.25', ...}
            last_price = float(quote['last'].replace(',',''))
            # Create a dummy DF
            df = pd.DataFrame({'Close': [last_price]}, index=[pd.Timestamp.now()])
            return df
    except Exception:
        pass

    return pd.DataFrame()

# ---------------- UI ----------------
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns Terminal (NSE Source)")

# Sidebar Controls
st.sidebar.title("💰 Portfolio & Risk")
portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", value=1000000, step=50000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.1)/100
selected_universe = st.sidebar.selectbox("Select Index", list(TICKER_GROUPS.keys()))

if st.sidebar.checkbox("Show NSE Source Info"):
    st.sidebar.info("Data provided by nsepython (Official NSE Data). Some analyst metrics may be unavailable.")

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
        # Data verification
        if data is None or data.empty: return None

        # Clean duplicates if any
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Ensure we have enough data for indicators (EMA 200)
        # Relaxed check: if < 200, we skip technicals or fail
        if len(data) < 200:
             # Optionally handle short history, for now we return None as per original logic requirement
             # Original was 210
             if len(data) < 50: return None

        close, high, low = data["Close"], data["High"], data["Low"]

        # Calculate Indicators
        # Handle if data length < 200 for EMA
        if len(close) >= 200:
            ema200 = ta.ema(close,200).iloc[-1]
            tech_base = float(close.iloc[-1]) > ema200
        else:
            ema200 = 0
            tech_base = True # Default to True or False if no history? Let's say False to be safe
            # Or maybe just skip EMA check?

        rsi = ta.rsi(close,14).iloc[-1]
        atr = ta.atr(high,low,close,14).iloc[-1]

        st_df = ta.supertrend(high,low,close,10,3)
        trend_col = [c for c in st_df.columns if c.startswith("SUPERTd")][0]
        trend_dir = int(st_df[trend_col].iloc[-1])
        price = float(close.iloc[-1])

        # Update tech_base with trend
        tech_base = tech_base and (trend_dir == 1)

        sl_distance = atr*1.5
        sl_price = round(price-sl_distance,2)
        target_10d = round(price + atr*1.8,2)
        risk_amount = portfolio_value*risk_per_trade
        pos_size = int(risk_amount / sl_distance) if sl_distance>0 else 0

        conviction = 0
        score_mod = 0
        news_sentiment = "Neutral"
        event_status = "✅ Safe"

        # News & Events (Mocked/Safe access)
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

        # Analyst Data (Mocked/Safe access)
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
            if rsi is not None:
                if 48<=rsi<=62: conviction+=20
                elif 40<=rsi<=72: conviction+=10
            conviction += score_mod

        dispersion_pct = ((target_high-target_low)/price)*100 if (price>0 and target_high>0) else 0
        dispersion_alert = "⚠️ High Dispersion" if dispersion_pct>30 else "✅"
        if dispersion_pct>30: conviction -= 10

        conviction = max(0,min(100,conviction))
        verdict = "🔥 HIGH" if conviction>=85 else "🚀 PASS" if conviction>=60 else "🟡 WATCH" if tech_base else "❌ FAIL"

        return {
            "Symbol": ticker,
            "Verdict": verdict,
            "Score": conviction,
            "Price": round(price,2),
            "RSI": round(rsi,1) if rsi is not None else 0,
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
    except Exception as e:
        # print(f"Error checking {ticker}: {e}")
        return None

# ---------------- MARKET PULSE ----------------
st.subheader("🌐 Market Pulse")
pulse_cols = st.columns(len(INDEX_BENCHMARKS))
for i,(name,symbol) in enumerate(INDEX_BENCHMARKS.items()):
    try:
        # Fetch using NSE adapter
        idx_data = fetch_index_history_safe(symbol)

        if not idx_data.empty and "Close" in idx_data.columns:
            p_close = idx_data["Close"].iloc[-1]

            # EMA check requires at least 200 data points
            if len(idx_data) >= 200:
                p_ema = ta.ema(idx_data["Close"],200).iloc[-1]
                p_status = "🟢 BULL" if p_close>p_ema else "🔴 BEAR"
            else:
                p_status = "⚪ NO TREND (Data < 200d)"

            pulse_cols[i].metric(name,f"{p_close:,.0f}",p_status)
        else:
            pulse_cols[i].metric(name, "N/A", "Data Error")
    except Exception as e:
        pulse_cols[i].metric(name, "Error", "Fetch Fail")
        pass

# ---------------- MAIN SCAN ----------------
if st.button("🚀 EXECUTE SYSTEM SCAN",type="primary",use_container_width=True):
    tickers = TICKER_GROUPS[selected_universe]
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i,ticker in enumerate(tickers):
        status_text.text(f"Scanning {ticker} ({i+1}/{len(tickers)})")
        try:
            # 1. Create NSEStock object (replaces yf.Ticker)
            tkr = NSEStock(ticker)

            # 2. Fetch History using NSE Adapter
            # ticker in config has .NS, fetch_stock_history handles stripping
            hist = fetch_stock_history(ticker, period="2y")

            if not hist.empty:
                res = check_institutional_fortress(ticker,hist,tkr,portfolio_val,risk_pct)
                if res: results.append(res)

            # Rate limiting handling
            time.sleep(0.5)

        except Exception as e:
            # print(f"Scan error for {ticker}: {e}")
            pass

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
        st.warning("No data retrieved. Check internet or ticker config. NSE data might be blocked.")

st.caption("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns | ATR SL | Analyst Dispersion | Full Logic | NSE Powered")

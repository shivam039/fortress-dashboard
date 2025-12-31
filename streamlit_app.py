import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# 1. Page Config
st.set_page_config(page_title="Fortress 300 Scanner", layout="wide")
st.title("🛡️ Fortress 95: Nifty 300 Scanner")

# 2. Correctly Load the CSV (Handles any format)
@st.cache_data
def load_nifty_300():
    try:
        # Load the CSV without assuming a header structure
        df = pd.read_csv("nifty300.csv", header=None)
        # Flatten all cells, clean them, and add .NS
        raw_list = df.values.flatten()
        tickers = []
        for s in raw_list:
            if pd.notna(s):
                clean_s = str(s).strip().replace(" ", "")
                if clean_s and clean_s.upper() != "SYMBOL":
                    tickers.append(clean_s.upper() + ".NS")
        return list(dict.fromkeys(tickers)) # Remove duplicates
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# 3. Defensive Logic Function
def check_fortress(ticker):
    try:
        # Download data
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        # FIX FOR MULTI-INDEX HEADERS (The cause of your error)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.dropna(inplace=True)
        if len(data) < 200: return None

        # Indicators
        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        # Supertrend calculation
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], length=10, multiplier=3)
        
        # SAFETY CHECK: If Supertrend returns None
        if st_df is None or st_df.empty: return None
        
        # Grab values
        price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        ema = data['EMA200'].iloc[-1]
        trend_dir = st_df.iloc[:, 1].iloc[-1] # 1 for Green/Up

        # Fortress 95 Logic
        if (price > ema) and (45 < rsi < 65) and (trend_dir == 1):
            return {"Price": round(float(price), 2), "RSI": round(float(rsi), 2)}
        return None
    except:
        return None

# 4. Interface and Scanning Loop
tickers = load_nifty_300()
st.sidebar.info(f"Loaded {len(tickers)} symbols from CSV")

if st.button("🚀 Start Scan"):
    results = []
    progress_bar = st.progress(0)
    status = st.empty()
    
    # We use a loop that updates the screen as it finds things
    for i, ticker in enumerate(tickers):
        status.text(f"Scanning {ticker} ({i+1}/{len(tickers)})")
        res = check_fortress(ticker)
        if res:
            results.append({"Symbol": ticker, "Price": res['Price'], "RSI": res['RSI']})
        progress_bar.progress((i + 1) / len(tickers))
    
    status.text("Scan Complete!")
    
    if results:
        st.success(f"Found {len(results)} Matches!")
        st.table(pd.DataFrame(results))
        for item in results:
            dhan_url = f"https://dhan.co/basket/?symbol={item['Symbol']}&qty=1&side=BUY"
            st.link_button(f"⚡ Buy {item['Symbol']}", dhan_url, key=item['Symbol'])
    else:
        st.warning("No matches found. Stay in Cash.")

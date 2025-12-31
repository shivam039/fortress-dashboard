import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# 1. Page Configuration
st.set_page_config(page_title="Fortress 95 Scanner", layout="wide")
st.title("🛡️ Fortress 95: Static 300 Scanner")

# 2. Load Local Static File
@st.cache_data
def load_static_tickers():
    try:
        # Reads the file directly from your GitHub repository folder
        df = pd.read_csv("nifty300.csv")
        # Standardizing tickers for Yahoo Finance (.NS suffix)
        tickers = []
        for symbol in df.values.flatten():
            if pd.notna(symbol):
                tickers.append(str(symbol).strip() + ".NS")
        return tickers
    except Exception as e:
        st.error(f"Error loading nifty300.csv: {e}")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# 3. Defensive Fortress Logic
def check_fortress(ticker):
    try:
        # Fetching 1 year of data for EMA200 calculation
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        # Clean data for indicators
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        
        if len(data) < 200:
            return None

        # Technical Indicators
        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)

        if st_df is None or st_df.empty:
            return None

        # Logic Variables
        price = float(data['Close'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        ema = float(data['EMA200'].iloc[-1])
        trend = st_df.iloc[:, 1].iloc[-1] # 1 for Green/Uptrend

        # Fortress 95 Criteria
        is_fortress = (price > ema) and (45 < rsi < 65) and (trend == 1)

        if is_fortress:
            return {"Price": round(price, 2), "RSI": round(rsi, 2)}
        return None
    except:
        return None

# 4. Main UI and Execution
tickers = load_static_tickers()
st.sidebar.write(f"Total Stocks in List: {len(tickers)}")

if st.button(f"🚀 Start Scanning {len(tickers)} Stocks"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanning {ticker} ({i+1}/{len(tickers)})")
        res = check_fortress(ticker)
        if res:
            results.append({"Symbol": ticker, "Price": res['Price'], "RSI": res['RSI']})
        progress_bar.progress((i + 1) / len(tickers))

    status_text.text("Scan Complete!")
    
    if results:
        st.success(f"Found {len(results)} High-Probability Matches!")
        res_df = pd.DataFrame(results)
        st.table(res_df)
        
        # Dhan Order Buttons
        for row in results:
            dhan_url = f"https://dhan.co/basket/?symbol={row['Symbol']}&qty=1&side=BUY"
            st.link_button(f"⚡ Buy {row['Symbol']}", dhan_url, key=f"btn_{row['Symbol']}")
    else:
        st.warning("No matches found. Stay in cash and wait for the next setup.")

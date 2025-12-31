import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd

# 1. Page Config
st.set_page_config(page_title="Fortress Dynamic Scanner", layout="wide")

# 2. Dynamic Ticker Loader
@st.cache_data # This saves the list so it doesn't re-download every second
def get_nifty_indices(index_name):
    urls = {
        "Nifty 50": "https://raw.githubusercontent.com/anirban-s/Nifty-Indices-Ticker-List/main/nifty50.csv",
        "Nifty Next 50": "https://raw.githubusercontent.com/anirban-s/Nifty-Indices-Ticker-List/main/niftynext50.csv",
        "Nifty Midcap 100": "https://raw.githubusercontent.com/anirban-s/Nifty-Indices-Ticker-List/main/niftymidcap100.csv"
    }
    try:
        df = pd.read_csv(urls[index_name])
        # Add .NS to tickers for Yahoo Finance
        return [str(symbol) + ".NS" for symbol in df['Symbol'].tolist()]
    except:
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS"] # Fallback

# 3. Fortress Logic (The Brain)
def check_fortress(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        if len(data) < 200: return None

        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        
        if st_df is None: return None
        
        price, rsi, ema = data['Close'].iloc[-1], data['RSI'].iloc[-1], data['EMA200'].iloc[-1]
        trend = st_df.iloc[:, 1].iloc[-1]

        if (price > ema) and (45 < rsi < 65) and (trend == 1):
            return {"Price": round(price, 2), "RSI": round(rsi, 2)}
    except:
        return None

# 4. Sidebar UI
st.sidebar.title("Settings")
selected_index = st.sidebar.selectbox("Select Index to Scan", ["Nifty 50", "Nifty Next 50", "Nifty Midcap 100"])
target_tickers = get_nifty_indices(selected_index)

st.title(f"🛡️ Fortress 95 Scanner: {selected_index}")
st.write(f"Monitoring {len(target_tickers)} stocks dynamically.")

# 5. Execution
if st.button(f"🔍 Scan {selected_index}"):
    results = []
    progress_text = st.empty()
    bar = st.progress(0)
    
    for i, ticker in enumerate(target_tickers):
        progress_text.text(f"Scanning {ticker} ({i+1}/{len(target_tickers)})")
        res = check_fortress(ticker)
        if res:
            results.append({"Symbol": ticker, "Price": res['Price'], "RSI": res['RSI']})
        bar.progress((i + 1) / len(target_tickers))
    
    progress_text.text("Scan Complete!")
    
    if results:
        st.success(f"Found {len(results)} Matches!")
        df_results = pd.DataFrame(results)
        st.table(df_results)
        
        # Action Buttons
        for res in results:
            dhan_url = f"https://dhan.co/basket/?symbol={res['Symbol']}&qty=1&side=BUY"
            st.link_button(f"⚡ Buy {res['Symbol']}", dhan_url)
    else:
        st.warning("No Fortress signals found. Cash is a position!")

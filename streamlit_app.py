import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf

st.set_page_config(page_title="Fortress 300 Scanner", layout="wide")
st.title("🛡️ Fortress 95: Static 300 Scanner")

@st.cache_data
def load_static_tickers():
    try:
        # Step 1: Read the CSV
        df = pd.read_csv("nifty300.csv", header=None) 
        
        # Step 2: Flatten every single cell into one long list
        all_elements = df.values.flatten()
        
        tickers = []
        for item in all_elements:
            if pd.notna(item): # Skip empty cells
                # Clean the text and add .NS for Yahoo Finance
                symbol = str(item).strip().replace('"', '').replace("'", "")
                if symbol and symbol != "Symbol": # Skip the header word if present
                    tickers.append(symbol + ".NS")
        
        # Step 3: Remove duplicates to stay efficient
        return list(dict.fromkeys(tickers)) 
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

def check_fortress(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        if len(data) < 200: return None

        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)

        if st_df is None or st_df.empty: return None

        price = float(data['Close'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        ema = float(data['EMA200'].iloc[-1])
        trend = st_df.iloc[:, 1].iloc[-1] 

        if (price > ema) and (45 < rsi < 65) and (trend == 1):
            return {"Price": round(price, 2), "RSI": round(rsi, 2)}
    except:
        return None

tickers = load_static_tickers()
if st.button(f"🚀 Start Full Scan ({len(tickers)} Stocks)"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanning {ticker}...")
        res = check_fortress(ticker)
        if res:
            results.append({"Symbol": ticker, "Price": res['Price'], "RSI": res['RSI']})
        progress_bar.progress((i + 1) / len(tickers))

    status_text.text("Scan Complete!")
    if results:
        st.success(f"Found {len(results)} High-Probability Entries!")
        st.table(pd.DataFrame(results))
        for row in results:
            dhan_url = f"https://dhan.co/basket/?symbol={row['Symbol']}&qty=1&side=BUY"
            st.link_button(f"⚡ Buy {row['Symbol']}", dhan_url, key=f"btn_{row['Symbol']}")
    else:
        st.warning("No matches found today.")

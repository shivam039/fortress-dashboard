import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import os

# 1. Page Setup
st.set_page_config(page_title="Fortress 300 Scanner", layout="wide")
st.title("🛡️ Fortress 95: Nifty 300 Scanner")

# 2. Updated Vertical CSV Loader
@st.cache_data
def load_nifty_300():
    file_path = "nifty300.csv"
    
    if not os.path.exists(file_path):
        st.error(f"❌ File dorakaledhu: '{file_path}' mee GitHub lo ledhu.")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    try:
        # Vertical list ni chadavataniki simple logic
        df = pd.read_csv(file_path, header=None, encoding='utf-8-sig')
        
        tickers = []
        for s in df.values.flatten():
            if pd.notna(s):
                clean_s = str(s).strip().replace(" ", "")
                if clean_s and clean_s.upper() != "SYMBOL":
                    tickers.append(clean_s.upper() + ".NS")
        
        final_list = list(dict.fromkeys(tickers))
        return final_list
    except Exception as e:
        st.error(f"🚨 Error: {e}")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# 3. Fortress Logic
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

        price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        ema = data['EMA200'].iloc[-1]
        trend_dir = st_df.iloc[:, 1].iloc[-1]

        if (price > ema) and (45 < rsi < 65) and (trend_dir == 1):
            return {"Price": round(float(price), 2), "RSI": round(float(rsi), 2)}
        return None
    except:
        return None

# 4. User Interface & Counter
tickers = load_nifty_300()

# Ikada counter kanipisthundi
st.sidebar.metric("Total Stocks Loaded", len(tickers))
st.sidebar.write("Mee CSV nundi inni stocks scan ki ready ga unnayi.")

if st.button(f"🚀 {len(tickers)} Stocks Scan Start Cheyandi"):
    results = []
    progress_bar = st.progress(0)
    status = st.empty()
    
    for i, ticker in enumerate(tickers):
        status.text(f"Scanning: {ticker} ({i+1}/{len(tickers)})")
        res = check_fortress(ticker)
        if res:
            results.append({"Symbol": ticker, "Price": res['Price'], "RSI": res['RSI']})
        progress_bar.progress((i + 1) / len(tickers))
    
    status.text("Scan Poorthi Ayindhi!")
    
    if results:
        st.success(f"Mothaniki {len(results)} Buy Signals Dorikayi!")
        st.table(pd.DataFrame(results))
        for item in results:
            dhan_url = f"https://dhan.co/basket/?symbol={item['Symbol']}&qty=1&side=BUY"
            st.link_button(f"⚡ Buy {item['Symbol']} on Dhan", dhan_url, key=item['Symbol'])
    else:
        st.warning("Eeroju Fortress logic ki thaggattu stock edi ledhu. Wait cheyandi.")

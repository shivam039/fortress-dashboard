import streamlit as st
import pandas_ta as ta
import yfinance as yf
from dhanhq import dhanhq

# 1. Dashboard Layout
st.set_page_config(page_title="Fortress 95 Sentry", layout="wide")
st.title("🛡️ Fortress 95: Fail-Proof Dashboard")

# 2. Connection to Dhan (Keys are pulled from Secret Settings)
client_id = st.secrets["DHAN_CLIENT_ID"]
access_token = st.secrets["DHAN_ACCESS_TOKEN"]
dhan = dhanhq(client_id, access_token)

# 3. The Logic Function
def check_fortress(ticker):
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if data.empty: return "Error"
    
    # Indicators
    data['EMA200'] = ta.ema(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
    
    # Criteria
    price = data['Close'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    ema = data['EMA200'].iloc[-1]
    trend = st_df['SUPERTd_10_3.0'].iloc[-1]
    
    # Fortress 95 Hit Logic
    is_fortress = (price > ema) and (45 < rsi < 65) and (trend == 1)
    return "🔥 BUY SIGNAL" if is_fortress else "Wait..."

# 4. User Interface
stocks = ["TITAN.NS", "VEDL.NS", "HINDCOPPER.NS", "RELIANCE.NS"]
if st.button("Run Fortress Scan"):
    for s in stocks:
        status = check_fortress(s)
        st.write(f"**{s}**: {status}")

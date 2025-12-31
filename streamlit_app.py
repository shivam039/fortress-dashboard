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
    # 1. Download data
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if len(data) < 200: return "Need 200+ days of data"
    
    # 2. Calculate Indicators
    data['EMA200'] = ta.ema(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    
    # 3. Supertrend Calculation
    # This returns a table with 4 columns: [Trend, Direction, Long, Short]
    st_df = ta.supertrend(data['High'], data['Low'], data['Close'], length=10, multiplier=3)
    
    # 4. Grab Direction by position (iloc[:, 1] means the 2nd column)
    # 1 = Bullish (Green), -1 = Bearish (Red)
    trend_direction = st_df.iloc[:, 1].iloc[-1]
    
    # 5. Fortress 95 Logic
    price = data['Close'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    ema = data['EMA200'].iloc[-1]
    
    # All 3 must be true
    is_fortress = (price > ema) and (45 < rsi < 65) and (trend_direction == 1)
    
    if is_fortress:
        return "🔥 CRITICAL ENTRY"
    else:
        return "STAY IN CASH"

# 4. User Interface
stocks = ["TITAN.NS", "VEDL.NS", "HINDCOPPER.NS", "RELIANCE.NS"]
if st.button("Run Fortress Scan"):
    for s in stocks:
        status = check_fortress(s)
        st.write(f"**{s}**: {status}")

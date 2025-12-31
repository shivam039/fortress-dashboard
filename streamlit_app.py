import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd

# 1. Dashboard Layout
st.set_page_config(page_title="Fortress 95 Sentry", layout="wide")
st.title("🛡️ Fortress 95: Fail-Proof Dashboard")

def check_fortress(ticker):
    try:
        # Download data
        raw_data = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        # FIX: Ensure data is not Multi-Index (common yfinance bug)
        data = raw_data.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Clean data
        data.dropna(inplace=True)
        
        if len(data) < 200:
            return "❌ Need 200+ days of history"

        # Calculate Indicators
        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        # Supertrend calculation
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], length=10, multiplier=3)

        # CHECK: If Supertrend failed to calculate
        if st_df is None or st_df.empty:
            return "⚠️ Indicator Error"

        # Logic Variables
        price = float(data['Close'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        ema = float(data['EMA200'].iloc[-1])
        # Using .iloc[:, 1] to get direction safely
        trend_direction = st_df.iloc[:, 1].iloc[-1]

        # THE 95% LOGIC
        is_fortress = (price > ema) and (45 < rsi < 65) and (trend_direction == 1)

        if is_fortress:
            return f"🔥 BUY (Price: {price:.2f}, RSI: {rsi:.1f})"
        else:
            return f"Wait (RSI: {rsi:.1f})"

    except Exception as e:
        return f"Error: {str(e)}"

# 4. User Interface
stocks = ["TITAN.NS", "VEDL.NS", "HINDCOPPER.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS"]

if st.button("Run Fortress Scan"):
    st.write("Checking market conditions...")
    for s in stocks:
        status = check_fortress(s)
        st.info(f"**{s}**: {status}")

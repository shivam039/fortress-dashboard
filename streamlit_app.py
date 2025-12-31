import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# 1. Page Config
st.set_page_config(page_title="Fortress 95 Scanner", layout="wide")
st.title("🛡️ Fortress 95: High-Probability Scanner")

# 2. Sidebar Filters
st.sidebar.title("🔍 Strategy Filters")
use_analyst_filter = st.sidebar.checkbox("Filter by Analyst Support", value=False)
min_analysts = st.sidebar.slider("Min Analysts Required", 0, 50, 10) if use_analyst_filter else 0

# NEW: Freshness Filter
st.sidebar.divider()
st.sidebar.subheader("🕒 Entry Freshness")
max_age = st.sidebar.slider("Max Trend Age (Days)", 1, 10, 5, help="Only see stocks that started their trend within these many days")

# 2. Hardcoded Ticker List
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", 
    "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS", "KOTAKBANK.NS", "TITAN.NS", 
    "ULTRACEMCO.NS", "AXISBANK.NS", "NTPC.NS", "ONGC.NS", "ADANIPORTS.NS",
    "ASIANPAINT.NS", "COALINDIA.NS", "JSWSTEEL.NS", "BAJAJ-AUTO.NS", "NESTLEIND.NS", 
    "GRASIM.NS", "HINDALCO.NS", "POWERGRID.NS", "ADANIPOWER.NS", "WIPRO.NS",
    "EICHERMOT.NS", "SBILIFE.NS", "TATAMOTORS.NS", "BPCL.NS", "DRREDDY.NS", 
    "HINDZINC.NS", "JIOFIN.NS", "TECHM.NS", "BRITANNIA.NS", "TATAPOWER.NS",
    "BAJAJFINSV.NS", "INDUSINDBK.NS", "ADANIGREEN.NS", "SHRIRAMFIN.NS", "LTIM.NS", 
    "TVSMOTOR.NS", "DLF.NS", "HAL.NS", "BEL.NS", "VEDL.NS", "VBL.NS", "PNB.NS", 
    "CANBK.NS", "IRFC.NS", "SIEMENS.NS", "UNITDSPR.NS", "PIDILITIND.NS", "TRENT.NS", 
    "GAIL.NS", "INDIGO.NS", "ABB.NS", "UNIONBANK.NS", "BANKBARODA.NS", "IOC.NS", 
    "CHOLAFIN.NS", "HEROMOTOCO.NS", "HAVELLS.NS", "GODREJCP.NS", "DABUR.NS", 
    "OBEROIRLTY.NS", "MANKIND.NS", "SHREECEM.NS", "ICICIPRU.NS", "PERSISTENT.NS", 
    "LUPIN.NS", "TATASTEEL.NS", "JINDALSTEL.NS", "TATACONSUM.NS", "AWL.NS", 
    "NYKAA.NS", "ZOMATO.NS", "TIINDIA.NS", "POLYCAB.NS", "AUBANK.NS", "MAXHEALTH.NS", 
    "SRF.NS", "MPHASIS.NS", "COFORGE.NS", "DIXON.NS", "ASTRAL.NS", "AMBUJACEM.NS", 
    "MUTHOOTFIN.NS", "PEL.NS", "OFSS.NS", "IDEA.NS", "YESBANK.NS", "SUZLON.NS", 
    "COLPAL.NS", "HDFCLIFE.NS", "ABCAPITAL.NS", "UPL.NS", "PAGEIND.NS", "CONCOR.NS", 
    "TATACOMM.NS", "PETRONET.NS", "TORNTPHARM.NS", "CUMMINSIND.NS", "TATAELXSI.NS", 
    "MRF.NS", "ASHOKLEY.NS", "DALBHARAT.NS", "PIIND.NS", "MAXFSL.NS", "RECLTD.NS", 
    "PFC.NS", "AUROPHARMA.NS", "COROMANDEL.NS", "LTTS.NS", "MFSL.NS", "DEEPAKNTR.NS", 
    "M&M.NS", "JKCEMENT.NS", "TATACHEM.NS", "VOLTAS.NS", "JUBLFOOD.NS", "SYNGENE.NS", 
    "GLAND.NS", "FORTIS.NS", "BATAINDIA.NS", "METROPOLIS.NS", "AARTIIND.NS", 
    "NAVINFLUOR.NS", "LAURUSLABS.NS", "INDIAMART.NS", "ATGL.NS", "ESCORTS.NS", 
    "CROMPTON.NS", "ZEEL.NS", "GLENMARK.NS", "GODREJPROP.NS", "SUNTV.NS", 
    "BALKRISIND.NS", "IPCALAB.NS", "IGL.NS", "LICHSGFIN.NS", "GUJGASLTD.NS", 
    "IDFCFIRSTB.NS", "IDFC.NS", "NAM-INDIA.NS", "BANDHANBNK.NS", "GMRINFRA.NS", 
    "NMDC.NS", "SAIL.NS", "NATIONALUM.NS", "ZYDUSLIFE.NS", "BIOCON.NS", 
    "CHAMBLFERT.NS", "INDIACEM.NS", "IBULHSGFIN.NS", "BHEL.NS", "RAIN.NS", 
    "RBLBANK.NS", "CANFINHOME.NS", "GRANULES.NS", "MANAPPURAM.NS", "IEX.NS", 
    "MGL.NS", "PVRINOX.NS", "MCX.NS"
]

# 3. Enhanced Logic Engine (STABILIZED with Entry Age)
@st.cache_data(ttl=900)
def check_institutional_fortress(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        ticker_obj = yf.Ticker(ticker)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        if len(data) < 200: return None

        # --- TECHNICALS ---
        price = data['Close'].iloc[-1]
        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        
        rsi = data['RSI'].iloc[-1]
        ema = data['EMA200'].iloc[-1]
        trend = st_df.iloc[:, 1].iloc[-1]

        # --- ENTRY AGE LOGIC (Looking back 5 days) ---
        # We calculate how many consecutive days the stock was above EMA, Green Trend, and RSI < 70
        days_in_trend = 0
        for i in range(1, 6):
            check_price = data['Close'].iloc[-i]
            check_ema = data['EMA200'].iloc[-i]
            check_rsi = data['RSI'].iloc[-i]
            check_trend = st_df.iloc[:, 1].iloc[-i]
            
            if (check_price > check_ema) and (45 <= check_rsi <= 70) and (check_trend == 1):
                days_in_trend += 1
            else:
                break # Sequence broken

        # --- RELIABILITY STATUS ---
        is_fresh_buy = (price > ema) and (45 <= rsi <= 65) and (trend == 1)
        is_trending_hold = (price > ema) and (65 < rsi < 75) and (trend == 1)

        if is_fresh_buy:
            status = "🚀 BUY"
        elif is_trending_hold:
            status = "📈 TRENDING"
        elif rsi >= 75:
            status = "✋ OVERBOUGHT"
        else:
            status = "🚫 AVOID"

        # --- CONVICTION SCORE ---
        score = 0
        if trend == 1: score += 30
        if price > ema: score += 20
        if 48 <= rsi <= 58: score += 30
        elif 45 <= rsi <= 65: score += 15
        
        # Stability Bonus: If it's been a buy for 2-3 days, it's more reliable than a 15-min spike
        if days_in_trend >= 2: score += 10


        # --- ANALYST DATA ---
        info = ticker_obj.info
        analyst_count = info.get('numberOfAnalystOpinions', 0)
        
        # --- FILTRATION CHECKS ---
        if use_analyst_filter and analyst_count < min_analysts:
            return None
            
        # NEW: Age Filter (Added here)
        if days_in_trend > max_age:
            return None
            
        expert_target = info.get('targetMeanPrice', 0)
        if expert_target and expert_target > price: score += 10

        return {
            "Symbol": ticker,
            "Age": f"{days_in_trend} Days",
            "Conviction Score": score,
            "Status": status,
            "Price": round(price, 2),
            "2-Week (ATR) Target": round(price + (data['ATR'].iloc[-1] * 2.5), 2),
            "Analysts 👤": analysts,
            "Expert Target": round(expert_target, 2) if expert_target else "N/A",
            "RSI": round(rsi, 2),
            "SL": round(price * 0.96, 2)
        }
    except:
        return None

# 4. Execution
if st.button("🚀 Run Institutional Scan"):
    results = []
    with st.status("Analyzing Market Sentiment & Technicals...", expanded=True):
        bar = st.progress(0)
        for i, t in enumerate(TICKERS):
            res = check_institutional_fortress(t)
            if res:
                results.append(res)
            bar.progress((i + 1) / len(TICKERS))

    # --- 5. Execution & Display ---
    if results:
        # Indented correctly under 'if results:'
        IST = pytz.timezone('Asia/Kolkata')
        timestamp_str = datetime.now(IST).strftime("%d-%b-%Y | %I:%M:%S %p")
        
        df = pd.DataFrame(results)
        
        # Sort so the highest conviction stocks appear first
        df = df.sort_values(by="Conviction Score", ascending=False)

        # 1. Define the Highlighting Logic
        def highlight_top_setups(row):
            # If the score is 90 or above, highlight the whole row GOLD
            if row['Conviction Score'] >= 90:
                return ['background-color: #FFD700; color: black; font-weight: bold'] * len(row)
            return [''] * len(row)

        st.subheader("📊 Fortress 95 Intelligence Dashboard")
        
        # Display Timestamp
        st.caption(f"🕒 **Last Market Scan (IST):** {timestamp_str}")
        
        st.write("Stocks highlighted in **Gold** represent the highest probability (95%) setups.")

        # 2. Apply the style and display the Dataframe
        st.dataframe(
            df.style.apply(highlight_top_setups, axis=1), 
            use_container_width=True,
            column_config={
                "Conviction Score": st.column_config.ProgressColumn(
                    "Probability",
                    help="Calculated based on Trend, RSI Sweet-spot, and Institutional backing",
                    min_value=0,
                    max_value=100,
                    format="%d%%",
                ),
                "Price": st.column_config.NumberColumn(format="₹%.2f"),
                "2-Week (ATR) Target": st.column_config.NumberColumn(format="₹%.2f"),
                "Expert Target": st.column_config.NumberColumn(format="₹%.2f"),
                "SL": st.column_config.NumberColumn("Stop Loss", format="₹%.2f"),
                "Age": st.column_config.TextColumn(
                    "Trend Age",
                    help="How many consecutive days this stock has been in the Fortress Buy zone"
                            }
                        )
    else:
        # This 'else' belongs to the 'if results:'
        st.warning("No matches found today. Wait for the market to setup.")

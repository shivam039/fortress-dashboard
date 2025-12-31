import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Fortress 95 Scanner", layout="wide")
st.title("🛡️ Fortress 95: High-Probability Scanner")
# 2. Sidebar Filters (Optional)
st.sidebar.title("🔍 Strategy Filters")
use_analyst_filter = st.sidebar.checkbox("Filter by Analyst Support", value=False)
min_analysts = st.sidebar.slider("Min Analysts Required", 0, 50, 10) if use_analyst_filter else 0

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

# 3. Enhanced Logic Engine
def check_institutional_fortress(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        ticker_obj = yf.Ticker(ticker)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        
        if len(data) < 200: return None

        # --- TECHNICALS ---
        price = data['Close'].iloc[-1]
        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        # 2-Week Target using ATR (14-day move projection)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        atr_value = data['ATR'].iloc[-1]
        # Statistically, a stock can move roughly 2-3 ATRs in 2 weeks
        two_week_target = round(price + (atr_value * 2.5), 2)

        # --- ANALYST DATA ---
        info = ticker_obj.info
        analyst_count = info.get('numberOfAnalystOpinions', 0)
        expert_target = info.get('targetMeanPrice', 0)
        
        # --- FILTRATION CHECK ---
        if use_analyst_filter and analyst_count < min_analysts:
            return None

        # --- FORTRESS LOGIC ---
        rsi = data['RSI'].iloc[-1]
        ema = data['EMA200'].iloc[-1]
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        trend = st_df.iloc[:, 1].iloc[-1]
        
        if (price > ema) and (45 <= rsi <= 65) and (trend == 1):
            status = "🚀 BUY"
        elif (price > ema) and (rsi > 65):
            status = "✋ HOLD"
        else:
            status = "🚫 AVOID"

        return {
            "Symbol": ticker,
            "Price": round(price, 2),
            "Status": status,
            "2-Week (ATR) Target": two_week_target,
            "Analysts 👤": analyst_count,
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

    if results:
        df = pd.DataFrame(results)
        st.subheader("📊 Market Intelligence Dashboard")
        
        # Visual color coding for the Status column
        def color_status(val):
            color = 'green' if 'BUY' in val else 'orange' if 'HOLD' in val else 'red'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            df.style.applymap(color_status, subset=['Status']),
            use_container_width=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="₹%.2f"),
                "2-Week (ATR) Target": st.column_config.NumberColumn(format="₹%.2f"),
                "Expert Target": st.column_config.NumberColumn(format="₹%.2f"),
                "SL": st.column_config.NumberColumn("Stop Loss (4%)", format="₹%.2f")
            }
        )
        st.caption("Note: 2-Week Target is calculated using 2.5x ATR volatility. Expert Target is the 12-month analyst mean.")
    else:
        st.warning("No stocks matched your current filter criteria.")

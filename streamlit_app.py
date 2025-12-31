import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Fortress 95 Scanner", layout="wide")
st.title("🛡️ Fortress 95: High-Probability Scanner")

# 2. Hardcoded Ticker List (Bypasses CSV errors)
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

# 3. Fortress Logic Engine
def check_fortress(ticker):
    try:
        # Fetching 1 year of data for EMA200
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        # Fixing yfinance Multi-index bug
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data.dropna(inplace=True)
        if len(data) < 200: return None

        # Technical Indicators
        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], length=10, multiplier=3)

        if st_df is None or st_df.empty: return None

        # Logic Variables
        price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        ema = data['EMA200'].iloc[-1]
        # Supertrend direction: 1 = Bullish, -1 = Bearish
        trend_dir = st_df.iloc[:, 1].iloc[-1] 

        # Fortress 95 Rule: Price > EMA, RSI 45-65, Trend is Green
        is_fortress = (price > ema) and (45 < rsi < 65) and (trend_dir == 1)

        if is_fortress:
            return {"Price": round(float(price), 2), "RSI": round(float(rsi), 2)}
        return None
    except:
        return None

# 4. Sidebar Metric
st.sidebar.metric("Database", f"{len(TICKERS)} Stocks")
st.sidebar.info("Strategy: Price > EMA200 + RSI(45-65) + Supertrend(Green)")

# 5. Execution Interface
if st.button(f"🚀 Start Scan ({len(TICKERS)} Heavyweights)"):
    st.write("🔍 Scanning market data... please wait.")
    progress_bar = st.progress(0)
    found_signals = []
    status_msg = st.empty()

    for i, s in enumerate(TICKERS):
        status_msg.text(f"Checking {s} ({i+1}/{len(TICKERS)})")
        result = check_fortress(s)
        if result:
            found_signals.append({"Symbol": s, "Price": result['Price'], "RSI": result['RSI']})
        
        # Visual Progress update
        progress_bar.progress((i + 1) / len(TICKERS))

    status_msg.text("✅ Scan Completed!")
    
    if found_signals:
        st.success(f"Found {len(found_signals)} stocks matching Fortress 95 criteria!")
        df_results = pd.DataFrame(found_signals)
        st.table(df_results)
        
        # Dhan Order Placement Buttons
        for stock in found_signals:
            dhan_url = f"https://dhan.co/basket/?symbol={stock['Symbol']}&qty=1&side=BUY"
            st.link_button(f"⚡ Buy {stock['Symbol']} on Dhan", dhan_url, key=stock['Symbol'])
    else:
        st.warning("No matches found. Market conditions may be overextended or bearish.")

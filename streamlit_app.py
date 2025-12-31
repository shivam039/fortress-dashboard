import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import time

# 1. Page Configuration
st.set_page_config(page_title="Fortress 95 Scanner", layout="wide")
st.title("🛡️ Fortress 95: High-Probability Scanner")

# 2. Reset Keys (The absolute fix for Duplicate Key Error)
if 'scan_id' not in st.session_state:
    st.session_state.scan_id = 0

# 3. Hardcoded Ticker List
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

# 4. Logic Engine
def check_fortress(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        if len(data) < 200: return None

        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], length=10, multiplier=3)

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

# 5. Sidebar
st.sidebar.metric("Stocks Ready", len(TICKERS))

# 6. Scanning Execution
if st.button("🚀 Start Scan"):
    # This is the secret: st.empty() creates a fresh hole in the page
    # that wipes itself clean every time the script reruns.
    main_container = st.empty()
    
    # Increment scan_id to refresh the "namespace"
    st.session_state.scan_id += 1
    
    with st.status("Searching for Fortress Entries...", expanded=True) as status:
        found_signals = []
        progress_bar = st.progress(0)
        for i, s in enumerate(TICKERS):
            res = check_fortress(s)
            if res:
                found_signals.append({"Symbol": s, "Price": res['Price'], "RSI": res['RSI']})
            progress_bar.progress((i + 1) / len(TICKERS))
        status.update(label="Scan Complete!", state="complete", expanded=False)

    # We now draw EVERYTHING inside that fresh placeholder
    with main_container.container():
        if found_signals:
            st.success(f"Found {len(found_signals)} Matches!")
            
            for idx, stock in enumerate(found_signals):
                # Use a very specific key format that includes the scan_id
                # This makes the button completely new to Streamlit
                btn_key = f"link_{stock['Symbol']}_{idx}_s{st.session_state.scan_id}"
                
                c1, c2, c3 = st.columns([2, 3, 2])
                with c1:
                    st.markdown(f"### {stock['Symbol']}")
                with c2:
                    st.write(f"Price: **{stock['Price']}** \nRSI: **{stock['RSI']}**")
                with c3:
                    dhan_url = f"https://dhan.co/basket/?symbol={stock['Symbol']}&qty=1&side=BUY"
                    st.link_button("⚡ Buy on Dhan", dhan_url, key=btn_key)
                st.divider()
        else:
            st.warning("No Fortress signals found. Market is sideways/bearish.")

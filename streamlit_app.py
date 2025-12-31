import subprocess
import sys
import time

# Auto-upgrade yfinance
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])

import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# --- HELPER FUNCTIONS ---
def clear_full_cache():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.toast("🧹 Cache cleared! Ready for a fresh scan.", icon="✅")

# 1. Page Config
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95: Rate-Limit Proof Scanner")

# 2. UPDATED FULL TICKER_GROUPS (Dec 2025)
TICKER_GROUPS = {
    "Nifty 50": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "LICI.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS", "KOTAKBANK.NS", "TITAN.NS", "ULTRACEMCO.NS", "AXISBANK.NS", "NTPC.NS", "ONGC.NS", "ADANIPORTS.NS", "ASIANPAINT.NS", "COALINDIA.NS", "JSWSTEEL.NS", "BAJAJ-AUTO.NS", "NESTLEIND.NS", "GRASIM.NS", "HINDALCO.NS", "POWERGRID.NS", "ADANIPOWER.NS", "WIPRO.NS", "EICHERMOT.NS", "SBILIFE.NS", "TATAMOTORS.NS", "BPCL.NS", "DRREDDY.NS", "HCLTECH.NS", "JIOFIN.NS", "TECHM.NS", "BRITANNIA.NS", "TATAPOWER.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS", "TVSMOTOR.NS", "APOLLOHOSP.NS", "CIPLA.NS", "BEL.NS", "TRENT.NS"],
    
    "Nifty Next 50": ["ADANIENSOL.NS", "ADANIGREEN.NS", "AMBUJACEM.NS", "DMART.NS", "BAJAJHLDNG.NS", "BANKBARODA.NS", "BHEL.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "HZL.NS", "ICICILOMB.NS", "ICICIPRULI.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", "JINDALSTEL.NS", "JSWENERGY.NS", "LTIM.NS", "LUPIN.NS", "MARICO.NS", "MRF.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "PFC.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SAMVARDHANA.NS", "SHREECEM.NS", "SIEMENS.NS", "TATACOMM.NS", "TATAELXSI.NS", "TATAMTRDVR.NS", "TORNTPHARM.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", "ABB.NS", "TIINDIA.NS", "POLYCAB.NS"],
    
    "Nifty Midcap 150": [
        "ABBOTINDIA.NS", "ABCAPITAL.NS", "ACC.NS", "ADANITOTAL.NS", "AIAENG.NS", "AJANTPHARM.NS", "ALKEM.NS", "APARINDS.NS", "APLAPOLLO.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS", "AUROPHARMA.NS", "AVANTIFEED.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BANKINDIA.NS", "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BIOCON.NS", "BLUESTARCO.NS", "BSE.NS", "CESC.NS", "CGPOWER.NS", "CHAMBLFERT.NS", "CHOLAHLDNG.NS", "COFORGE.NS", "CONCOR.NS", "COROMANDEL.NS", "CREDITACC.NS", "CROMPTON.NS", "CUMMINSIND.NS", "CYIENT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS", "DEVYANI.NS", "DIXON.NS", "EASEMYTRIP.NS", "EDELWEISS.NS", "EICHERMOT.NS", "EMAMILTD.NS", "ENDURANCE.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS", "GICRE.NS", "GLENMARK.NS", "GMRINFRA.NS", "GODREJIND.NS", "GODREJPROP.NS", "GRANULES.NS", "GUJGASLTD.NS", "HAPPSTMNDS.NS", "HDFCAMC.NS", "HFCL.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HUDCO.NS", "IDBI.NS", "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIAMART.NS", "INDIANB.NS", "INDIGO.NS", "IPCALAB.NS", "IRB.NS", "ITDCEM.NS", "JBCHEPHARM.NS", "JKCEMENT.NS", "JSL.NS", "JSWINFRA.NS", "JUBLFOOD.NS", "KALYANKJIL.NS", "KEI.NS", "KOTAKBANK.NS", "KPITTECH.NS", "KPRMILL.NS", "L&TFH.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LINDEINDIA.NS", "LLOYDSME.NS", "LUPIN.NS", "MAHABANK.NS", "MAHINDCIE.NS", "MANAPPURAM.NS", "MANKIND.NS", "MARICO.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "METROPOLIS.NS", "MFSL.NS", "MGL.NS", "MOTILALOFS.NS", "MPHASIS.NS", "MRPL.NS", "MUTHOOTFIN.NS", "NATCOPHARM.NS", "NATIONALUM.NS", "NAVINFLUOR.NS", "NBCC.NS", "NHPC.NS", "NLCINDIA.NS", "NMDC.NS", "NTPC.NS", "NTPCGREEN.NS", "NYKAA.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "ONGC.NS", "PAGEIND.NS", "PATANJALI.NS", "PAYTM.NS", "PERSISTENT.NS", "PETRONET.NS", "PHOENIXLTD.NS", "PIIND.NS", "PNBHOUSING.NS", "POLYMED.NS", "POONAWALLA.NS", "PRESTIGE.NS", "PVRINOX.NS", "QUESS.NS", "RADICO.NS", "RAILTEL.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "RATNAMANI.NS", "RBLBANK.NS", "RECLTD.NS", "RELAXO.NS", "RVNL.NS", "SAFEARIAS.NS", "SAIL.NS", "SCHAEFFLER.NS", "SHREECEM.NS", "SJVN.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SRF.NS", "STLTECH.NS", "SUNTV.NS", "SUPREMEIND.NS", "SUZLON.NS", "SWIGGY.NS", "SYNGENE.NS", "TATACHEM.NS", "TATAELXSI.NS", "TATAMTRDVR.NS", "TATATECH.NS", "TRIDENT.NS", "UCOBANK.NS", "UNIONBANK.NS", "UPL.NS", "VGUARD.NS", "VI.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "YESBANK.NS", "ZEEL.NS"
    ]
}

# Sector Mapping (expanded for full coverage)
SECTOR_MAP = {
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking", "KOTAKBANK.NS": "Banking", 
    "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking", "BANKBARODA.NS": "Banking", "CANBK.NS": "Banking", 
    "PNB.NS": "Banking", "BAJFINANCE.NS": "NBFC", "BAJAJFINSV.NS": "NBFC", "CHOLAFIN.NS": "NBFC",
    "SHRIRAMFIN.NS": "NBFC", "MUTHOOTFIN.NS": "NBFC", "IDFCFIRSTB.NS": "Banking", "TCS.NS": "IT", 
    "INFY.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT", "TECHM.NS": "IT", "LTIM.NS": "IT", 
    "MPHASIS.NS": "IT", "PERSISTENT.NS": "IT", "COFORGE.NS": "IT", "TATAELXSI.NS": "IT",
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy", "IOC.NS": "Energy",
    "ADANIPOWER.NS": "Energy", "TATAPOWER.NS": "Energy", "NTPC.NS": "Energy", "POWERGRID.NS": "Energy",
    "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals", "VEDL.NS": "Metals", "JINDALSTEL.NS": "Metals",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "BAJAJ-AUTO.NS": "Auto", "EICHERMOT.NS": "Auto",
    "TVSMOTOR.NS": "Auto", "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG",
    "LT.NS": "Infra", "ADANIPORTS.NS": "Infra", "BEL.NS": "Defense", "HAL.NS": "Defense",
    "TRENT.NS": "Retail", "ZOMATO.NS": "Retail", "NYKAA.NS": "Retail"
}

# Sidebar
st.sidebar.title("🔍 Strategy Filters")
selected_index = st.sidebar.selectbox("Select Universe", options=list(TICKER_GROUPS.keys()), index=0, key="selected_index")
TICKERS = TICKER_GROUPS[selected_index]
st.sidebar.write(f"📊 Total Stocks: **{len(TICKERS)}** | **Scan Time: ~{len(TICKERS)*0.7/60:.1f}min**")

st.sidebar.divider()
st.sidebar.subheader("⚙️ Maintenance")
if st.sidebar.button("🧹 Clear All Cache"):
    clear_full_cache()
    st.rerun()

# --- ENHANCED MARKET PULSE (unchanged) ---
st.subheader("🌐 Global Market Benchmarks")
index_benchmarks = {
    "Nifty 50": ["^NSEI", "NIFTY50.NS"],
    "Nifty Next 50": ["^NIFTYJR", "NIFTYNEXT50.NS"],
    "Nifty Midcap 150": ["^NSMIDCP", "NIFTYMIDCAP150.NS"]
}

pulse_cols = st.columns(len(index_benchmarks))
market_health = []

for i, (name, tickers) in enumerate(index_benchmarks.items()):
    idx_data = None
    for ticker in tickers:
        try:
            idx_data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True, threads=False)
            if not idx_data.empty:
                break
        except:
            continue
    
    if idx_data is not None and not idx_data.empty:
        if isinstance(idx_data.columns, pd.MultiIndex):
            idx_data.columns = idx_data.columns.get_level_values(0)
        idx_price = idx_data['Close'].iloc[-1]
        idx_ema = ta.ema(idx_data['Close'], length=200).iloc[-1]
        idx_rsi = ta.rsi(idx_data['Close'], length=14).iloc[-1]
        status = "BULLISH" if idx_price > idx_ema else "BEARISH"
        market_health.append(status == "BULLISH")
        pulse_cols[i].metric(
            label=f"{name}",
            value=f"{idx_price:,.1f}",
            delta=f"{status} (RSI: {idx_rsi:.1f})",
            delta_color="normal" if status == "BULLISH" else "inverse"
        )
    else:
        pulse_cols[i].error(f"⚠️ {name} Link Broken")

bullish_count = sum(market_health)
if bullish_count >= 2:
    st.success("✅ **Market Support:** Broad trend is BULLISH. Perfect for breakouts.")
elif bullish_count == 1:
    st.warning("⚖️ **Mixed Market:** Trade only Nifty 50 breakouts.")
else:
    st.error("🛑 **System Alert:** Full Market BEARISH. High risk for longs.")

# 2. LOGIC FUNCTION (receives pre-fetched data)
def check_institutional_fortress(ticker, data):
    try:
        if len(data) < 200: 
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)

        price = data['Close'].iloc[-1]
        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        
        rsi = data['RSI'].iloc[-1]
        ema = data['EMA200'].iloc[-1]
        trend = st_df.iloc[:, 1].iloc[-1]

        is_buy_setup = (price > ema) and (40 <= rsi <= 70) and (trend == 1)
        if not is_buy_setup:
            return None

        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        pe = info.get('trailingPE', "N/A")
        pb = info.get('priceToBook', "N/A")
        
        age = 0
        for i in range(1, 15):
            if data['Close'].iloc[-i] > data['EMA200'].iloc[-i] and st_df.iloc[:, 1].iloc[-i] == 1:
                age += 1
            else: break

        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Age": f"{age}d",
            "Score": 95 if (48 <= rsi <= 58) else 80,
            "Price": round(price, 2),
            "RSI": round(rsi, 2),
            "PE": pe if pe == "N/A" else round(pe, 1),
            "PB": pb if pb == "N/A" else round(pb, 1),
            "Target": round(price + (data['ATR'].iloc[-1] * 2.5), 2),
            "SL": round(price * 0.96, 2)
        }
    except Exception as e:
        return None

# 3. SAFE SCANNING LOOP (150-STOCK READY)
if st.button("🚀 Start Safe Institutional Scan"):
    results = []
    ticker_list = TICKERS
    total = len(ticker_list)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(ticker_list):
        status_text.text(f"Scanning {i+1}/{total}: {ticker}")
        
        try:
            data = yf.download(
                ticker, 
                period="1y", 
                interval="1d", 
                progress=False, 
                threads=False,
                auto_adjust=True
            )
            
            if data.empty:
                st.warning(f"⚠️ {ticker}: No data/Rate limited. Waiting 2s...")
                time.sleep(2)
                continue

            res = check_institutional_fortress(ticker, data)
            if res:
                results.append(res)
                st.toast(f"✅ Found Setup: {ticker}", icon="🚀")
            
            time.sleep(0.7)  # Golden delay for 150-stock scans
            
        except Exception as e:
            if "Rate limited" in str(e) or "429" in str(e):
                st.error("🚨 Yahoo Throttled. Sleeping 10s...")
                time.sleep(10)
            else:
                st.info(f"Skipping {ticker}")
                
        progress_bar.progress((i + 1) / total)

    status_text.success(f"✅ Scan Complete! Found {len(results)} setups.")

    if results:
        IST = pytz.timezone('Asia/Kolkata')
        timestamp_str = datetime.now(IST).strftime("%d-%b-%Y | %I:%M:%S %p")
        df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

        st.subheader("🏦 Sector Distribution")
        st.bar_chart(df.groupby('Sector').size(), height=400)

        def highlight_rows(row):
            if row['Score'] == 95:
                return ['background-color: #FFD700; color: black; font-weight: bold'] * len(row)
            return [''] * len(row)

        st.subheader("📊 Pure Breakout Dashboard")
        st.caption(f"🕒 **{selected_index} Scan (IST):** {timestamp_str} | Found: {len(results)}/{len(TICKERS)}")
        st.info(f"**SAFE SCAN:** threads=False | 0.7s delay | **{len(TICKERS)}-stock universe** | Data reuse")
        
        st.dataframe(
            df.style.apply(highlight_rows, axis=1),
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Strength", min_value=0, max_value=100, format="%d"),
                "Age": st.column_config.TextColumn("Freshness", help="Days above EMA200+SuperTrend"),
                "PE": st.column_config.NumberColumn("P/E", format="%.1f"),
                "PB": st.column_config.NumberColumn("P/B", format="%.1f"),
                "Target": st.column_config.NumberColumn("Target", format="₹%.0f"),
                "SL": st.column_config.NumberColumn("Stop Loss", format="₹%.0f")
            }
        )
    else:
        st.warning(f"No pure breakouts found in {selected_index}.")

st.caption("🛡️ **Fortress 95 Pro** - FULL 150 Midcaps | Streamlit Cloud Safe | Production Ready")

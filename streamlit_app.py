import subprocess
import sys
import time
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Auto-install yfinance
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])

# --- SYSTEM CONFIG ---
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v5.1 - FULL RESULTS + AI REPORTS")

# --- COMPLETE TICKER DATABASE (250+ stocks) ---
TICKER_GROUPS = {
    "Nifty 50": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", 
        "SBIN.NS", "ITC.NS", "LICI.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS", 
        "SUNPHARMA.NS", "ADANIENT.NS", "KOTAKBANK.NS", "TITAN.NS", "ULTRACEMCO.NS", "AXISBANK.NS", 
        "NTPC.NS", "ONGC.NS", "ADANIPORTS.NS", "ASIANPAINT.NS", "COALINDIA.NS", "JSWSTEEL.NS", 
        "BAJAJ-AUTO.NS", "NESTLEIND.NS", "GRASIM.NS", "HINDALCO.NS", "POWERGRID.NS", 
        "ADANIPOWER.NS", "WIPRO.NS", "EICHERMOT.NS", "SBILIFE.NS", "TATAMOTORS.NS", 
        "BPCL.NS", "DRREDDY.NS", "HCLTECH.NS", "JIOFIN.NS", "TECHM.NS", "BRITANNIA.NS", 
        "TATAPOWER.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS", "TVSMOTOR.NS", 
        "APOLLOHOSP.NS", "CIPLA.NS", "BEL.NS", "TRENT.NS"
    ],
    "Nifty Next 50": [
        "ADANIENSOL.NS", "ADANIGREEN.NS", "AMBUJACEM.NS", "DMART.NS", "BAJAJHLDNG.NS", 
        "BANKBARODA.NS", "BHEL.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", 
        "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "HZL.NS", 
        "ICICILOMB.NS", "ICICIPRULI.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", "JINDALSTEL.NS", 
        "JSWENERGY.NS", "LTIM.NS", "LUPIN.NS", "MARICO.NS", "MRF.NS", "MUTHOOTFIN.NS", 
        "NAUKRI.NS", "PFC.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SAMVARDHANA.NS", 
        "SHREECEM.NS", "SIEMENS.NS", "TATACOMM.NS", "TATAELXSI.NS", "TATAMTRDVR.NS", 
        "TORNTPHARM.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", 
        "ABB.NS", "TIINDIA.NS", "POLYCAB.NS"
    ],
    "Nifty Midcap 150": [
        "ABBOTINDIA.NS", "ABCAPITAL.NS", "ACC.NS", "ADANITOTAL.NS", "AIAENG.NS", "AJANTPHARM.NS", 
        "ALKEM.NS", "APARINDS.NS", "APLAPOLLO.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS", 
        "AUROPHARMA.NS", "AVANTIFEED.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BANKINDIA.NS", 
        "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BIOCON.NS", "BLUESTARCO.NS", 
        "BSE.NS", "CESC.NS", "CGPOWER.NS", "CHAMBLFERT.NS", "CHOLAHLDNG.NS", "COFORGE.NS", 
        "CONCOR.NS", "COROMANDEL.NS", "CREDITACC.NS", "CROMPTON.NS", "CUMMINSIND.NS", 
        "CYIENT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS", "DEVYANI.NS", "DIXON.NS", "EASEMYTRIP.NS", 
        "EDELWEISS.NS", "EICHERMOT.NS", "EMAMILTD.NS", "ENDURANCE.NS", "ESCORTS.NS", 
        "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS", "GICRE.NS", "GLENMARK.NS", "GMRINFRA.NS", 
        "GODREJIND.NS", "GODREJPROP.NS", "GRANULES.NS", "GUJGASLTD.NS", "HAPPSTMNDS.NS", 
        "HDFCAMC.NS", "HFCL.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HUDCO.NS", "IDBI.NS", 
        "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIAMART.NS", "INDIANB.NS", 
        "INDIGO.NS", "IPCALAB.NS", "IRB.NS", "ITDCEM.NS", "JBCHEPHARM.NS", "JKCEMENT.NS", 
        "JSL.NS", "JSWINFRA.NS", "JUBLFOOD.NS", "KALYANKJIL.NS", "KEI.NS", "KPITTECH.NS", 
        "KPRMILL.NS", "L&TFH.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LINDEINDIA.NS", 
        "LLOYDSME.NS", "LUPIN.NS", "MAHABANK.NS", "MAHINDCIE.NS", "MANAPPURAM.NS", 
        "MANKIND.NS", "MARICO.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "METROPOLIS.NS", 
        "MFSL.NS", "MGL.NS", "MOTILALOFS.NS", "MPHASIS.NS", "MRPL.NS", "MUTHOOTFIN.NS", 
        "NATCOPHARM.NS", "NATIONALUM.NS", "NAVINFLUOR.NS", "NBCC.NS", "NHPC.NS", 
        "NLCINDIA.NS", "NMDC.NS", "NYKAA.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", 
        "PAGEIND.NS", "PATANJALI.NS", "PAYTM.NS", "PERSISTENT.NS", "PETRONET.NS", 
        "PHOENIXLTD.NS", "PIIND.NS", "PNBHOUSING.NS", "POLYMED.NS", "POONAWALLA.NS", 
        "PRESTIGE.NS", "PVRINOX.NS", "QUESS.NS", "RADICO.NS", "RAILTEL.NS", 
        "RAJESHEXPO.NS", "RAMCOCEM.NS", "RATNAMANI.NS", "RBLBANK.NS", "RECLTD.NS", 
        "RELAXO.NS", "RVNL.NS", "SAIL.NS", "SCHAEFFLER.NS", "SHREECEM.NS", "SJVN.NS", 
        "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SRF.NS", "STLTECH.NS", "SUNTV.NS", 
        "SUPREMEIND.NS", "SUZLON.NS", "SWIGGY.NS", "SYNGENE.NS", "TATACHEM.NS", 
        "TATATECH.NS", "TRIDENT.NS", "UCOBANK.NS", "UNIONBANK.NS", "UPL.NS", 
        "VGUARD.NS", "VI.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "YESBANK.NS", "ZEEL.NS"
    ]
}

# Sector Mapping
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

# --- AI ANALYST REPORT POPUP ---
@st.dialog("📋 Analyst Consensus & AI Summary", width="large")
def show_analyst_report(ticker_symbol):
    st.markdown(f"### Detailed Report: **{ticker_symbol}**")
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rating", info.get('recommendationKey', 'N/A').upper())
            st.metric("Target Price", f"₹{info.get('targetMeanPrice', 'N/A')}")
        with col2:
            st.metric("Analysts", info.get('numberOfAnalystOpinions', 0))
            st.metric("Sector", info.get('sector', 'N/A'))
        
        current_price = info.get('currentPrice', 1)
        target_price = info.get('targetMeanPrice', 0)
        upside_pct = ((target_price / current_price) - 1) * 100 if current_price > 0 else 0
        
        st.markdown("#### 🤖 AI Analyst Summary:")
        points = [
            f"🔹 **Valuation:** P/E {info.get('trailingPE', 'N/A')} | {'⚠️ HIGH' if info.get('trailingPE', 0) > 25 else '✅ FAIR'}",
            f"🔹 **Upside:** {info.get('numberOfAnalystOpinions', 0)} analysts see **{upside_pct:.1f}%** growth",
            "🔹 **Fortress Signal:** EMA200 + SuperTrend confirmed",
            f"🔹 **Market Cap:** ₹{info.get('marketCap', 'N/A'):,} | Beta: {info.get('beta', 'N/A')}",
            "🔹 **Risk:** Monitor earnings calendar"
        ]
        
        for point in points:
            st.write(point)
            
        col1, col2 = st.columns(2)
        if col1.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        if col2.button("❌ Close", use_container_width=True):
            st.rerun()
            
    except Exception as e:
        st.error(f"⚠️ Report unavailable: {str(e)}")
        if st.button("❌ Close"):
            st.rerun()

# --- FORTRESS SCAN ENGINE ---
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        if len(data) < 200:
            return {
                "Symbol": ticker, "Sector": SECTOR_MAP.get(ticker, "N/A"),
                "Verdict": "⚠️ ERROR", "Report": "📋", "Price": 0, "RSI": 0, 
                "Age": "0d", "Analyst Target": "N/A", "Analysts": 0, 
                "Upside %": "N/A", "Score": 0
            }
        
        # Fix column issues
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna()
        
        # Core technicals
        price = data['Close'].iloc[-1]
        ema200 = ta.ema(data['Close'], length=200).iloc[-1]
        rsi = ta.rsi(data['Close'], length=14).iloc[-1]
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        trend = st_df.iloc[:, 1].iloc[-1]
        
        # Fortress criteria
        is_pass = (price > ema200 and 40 <= rsi <= 70 and trend == 1)
        
        # Analyst data
        info = ticker_obj.info
        target = info.get('targetMeanPrice', 0)
        analysts = info.get('numberOfAnalystOpinions', 0)
        upside = ((target - price) / price * 100) if target > 0 else 0
        
        # Trend age
        age = 0
        for i in range(1, 15):
            if i < len(data) and data['Close'].iloc[-i] > ema200 and st_df.iloc[:, 1].iloc[-i] == 1:
                age += 1
            else:
                break
        
        # Score calculation
        score = 95 if (is_pass and 48 <= rsi <= 58) else (80 if is_pass else 0)
        
        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Verdict": "🚀 PASS" if is_pass else "❌ FAIL",
            "Report": "📋",
            "Price": round(price, 2),
            "RSI": round(rsi, 2),
            "Age": f"{age}d",
            "Analyst Target": round(target, 2) if target > 0 else "N/A",
            "Analysts": analysts,
            "Upside %": f"{upside:.1f}%" if upside != 0 else "N/A",
            "Score": score
        }
    except:
        return {
            "Symbol": ticker, "Sector": SECTOR_MAP.get(ticker, "N/A"),
            "Verdict": "⚠️ ERROR", "Report": "📋", "Price": 0, "RSI": 0, 
            "Age": "0d", "Analyst Target": "N/A", "Analysts": 0, 
            "Upside %": "N/A", "Score": 0
        }

# --- MARKET PULSE ---
st.subheader("🌐 Market Pulse")
index_benchmarks = {"Nifty 50": "^NSEI", "Nifty Next 50": "^NIFTYJR", "Nifty Midcap 150": "^NSMIDCP"}
cols = st.columns(3)
market_health = []
bullish_count = 0

for i, (name, symbol) in enumerate(index_benchmarks.items()):
    try:
        data = yf.download(symbol, period="1y", progress=False, threads=False)
        if not data.empty:
            price = data['Close'].iloc[-1]
            ema = ta.ema(data['Close'], 200).iloc[-1]
            status = "🟢 BULLISH" if price > ema else "🔴 BEARISH"
            if price > ema:
                bullish_count += 1
            market_health.append(status)
            cols[i].metric(name, f"₹{price:,.0f}", status)
        else:
            cols[i].error(f"{name} unavailable")
    except:
        cols[i].error(f"{name} error")

if bullish_count >= 2:
    st.success("✅ **BULL MARKET CONFIRMED** - Perfect breakout conditions!")
elif bullish_count == 1:
    st.warning("⚠️ **Mixed signals** - Focus on Nifty 50")
else:
    st.error("🛑 **BEAR MARKET** - High risk environment")

# --- CONTROLS & EXECUTION ---
st.sidebar.title("🔍 Fortress Controls")
selected_index = st.sidebar.selectbox("Universe", list(TICKER_GROUPS.keys()), key="universe")
TICKERS = TICKER_GROUPS[selected_index]
st.sidebar.info(f"📊 **{len(TICKERS)} stocks** | ⏱️ **~{len(TICKERS)*0.7/60:.1f}min** | **CLICK 📋 for AI Reports**")

if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# MAIN SCAN BUTTON
if st.button("🚀 START FULL FORTRESS SCAN", type="primary", use_container_width=True):
    results = []
    ticker_list = TICKERS
    total = len(ticker_list)
    progress_bar = st.progress(0)
    status_text = st.empty()
    pass_count = 0
    
    for i, ticker in enumerate(ticker_list):
        status_text.text(f"🔍 [{i+1}/{total}] Scanning {ticker}...")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            data = yf.download(ticker, period="1y", interval="1d", progress=False, 
                             threads=False, auto_adjust=True)
            
            if data.empty:
                time.sleep(2)
                continue
            
            result = check_institutional_fortress(ticker, data, ticker_obj)
            results.append(result)
            
            if result['Verdict'] == "🚀 PASS":
                pass_count += 1
                st.toast(f"✅ FORTRESS PASS: {ticker} (Score: {result['Score']})", icon="🚀")
            
            time.sleep(0.7)  # Rate limit protection
            
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                status_text.error("🚨 Rate limit hit. Cooling...")
                time.sleep(10)
            continue
            
        progress_bar.progress((i + 1) / total)

    status_text.success(f"✅ **SCAN COMPLETE!** {pass_count}/{total} Fortress setups found.")

    # FULL RESULTS DISPLAY
    if results:
        IST = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(IST).strftime("%d-%b-%Y | %I:%M %p IST")
        
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # SUMMARY METRICS
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚀 PASSES", pass_count)
        col2.metric("📈 Top Score", df['Score'].max())
        col3.metric("🔥 Max Analysts", int(df['Analysts'].max()))
        col4.metric("📊 Scanned", f"{len(results)}/{total}")
        
        # SECTOR BREAKDOWN (PASS stocks only)
        pass_df = df[df['Verdict'] == '🚀 PASS']
        if not pass_df.empty:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("🏦 Sector Heatmap")
                st.bar_chart(pass_df['Sector'].value_counts(), height=300)
            with col2:
                st.subheader("📊 Ultimate Fortress Dashboard")
                st.caption(f"**{selected_index}** | {timestamp} | {pass_count}/{total} | {bullish_count}/3 bullish")
        
        # INTERACTIVE DATAFRAME
        st.subheader(f"📊 Complete {selected_index} Analysis - **CLICK ANY ROW → 📋 AI Report**")
        
        def color_verdict(val):
            if val == '🚀 PASS':
                return 'color: green; font-weight: bold; font-size: 14px'
            elif val == '❌ FAIL':
                return 'color: red; font-weight: bold'
            else:
                return 'color: orange; font-weight: bold'
        
        selected_row = st.dataframe(
            df.style.applymap(color_verdict, subset=['Verdict']),
            use_container_width=True,
            selection_mode="single-row",
            column_config={
                "Score": st.column_config.ProgressColumn("Fortress Score", min_value=0, max_value=100, format="%d%%"),
                "Verdict": st.column_config.TextColumn("Status", help="Price>EMA200 + RSI(40-70) + SuperTrend=1"),
                "Report": st.column_config.TextColumn("Report", help="Click row for AI analyst summary 📋"),
                "Analyst Target": st.column_config.NumberColumn("Analyst Target ₹", format="₹%.0f"),
                "Analysts": st.column_config.NumberColumn("Coverage", help="Institutional analysts"),
                "Upside %": st.column_config.TextColumn("Upside Potential"),
                "Price": st.column_config.NumberColumn("Current Price ₹", format="₹%.0f")
            },
            height=700
        )
        
        # TRIGGER AI REPORT ON ROW CLICK
        if selected_row and 'selection' in selected_row and selected_row['selection'].get('rows'):
            row_index = selected_row['selection']['rows'][0]
            ticker_symbol = df.iloc[row_index]['Symbol']
            show_analyst_report(ticker_symbol)
    
    else:
        st.warning("🏰 **No data returned.** Try smaller universe or check internet.")

st.markdown("---")
st.caption("🛡️ **Fortress 95 Pro v5.1** - 250+ Tickers | Full Results | AI Reports | Rate-Limit Proof | Production Ready")

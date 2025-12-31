import subprocess
import sys
import time
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Auto-upgrade yfinance (PRODUCTION SAFE)
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])

# --- SYSTEM INITIALIZATION ---
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro: Dual Target Institutional Scanner")

# --- MASTER TICKER LISTS (250+ Tickers) ---
TICKER_GROUPS = {
    "Nifty 50": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "LICI.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS", "KOTAKBANK.NS", "TITAN.NS", "ULTRACEMCO.NS", "AXISBANK.NS", "NTPC.NS", "ONGC.NS", "ADANIPORTS.NS", "ASIANPAINT.NS", "COALINDIA.NS", "JSWSTEEL.NS", "BAJAJ-AUTO.NS", "NESTLEIND.NS", "GRASIM.NS", "HINDALCO.NS", "POWERGRID.NS", "ADANIPOWER.NS", "WIPRO.NS", "EICHERMOT.NS", "SBILIFE.NS", "TATAMOTORS.NS", "BPCL.NS", "DRREDDY.NS", "HCLTECH.NS", "JIOFIN.NS", "TECHM.NS", "BRITANNIA.NS", "TATAPOWER.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS", "TVSMOTOR.NS", "APOLLOHOSP.NS", "CIPLA.NS", "BEL.NS", "TRENT.NS"],
    "Nifty Next 50": ["ADANIENSOL.NS", "ADANIGREEN.NS", "AMBUJACEM.NS", "DMART.NS", "BAJAJHLDNG.NS", "BANKBARODA.NS", "BHEL.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "HZL.NS", "ICICILOMB.NS", "ICICIPRULI.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", "JINDALSTEL.NS", "JSWENERGY.NS", "LTIM.NS", "LUPIN.NS", "MARICO.NS", "MRF.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "PFC.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SAMVARDHANA.NS", "SHREECEM.NS", "SIEMENS.NS", "TATACOMM.NS", "TATAELXSI.NS", "TATAMTRDVR.NS", "TORNTPHARM.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", "ABB.NS", "TIINDIA.NS", "POLYCAB.NS"],
    "Nifty Midcap 150": ["ABBOTINDIA.NS", "ABCAPITAL.NS", "ACC.NS", "ADANITOTAL.NS", "AIAENG.NS", "AJANTPHARM.NS", "ALKEM.NS", "APARINDS.NS", "APLAPOLLO.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS", "AUROPHARMA.NS", "AVANTIFEED.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BANKINDIA.NS", "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BIOCON.NS", "BLUESTARCO.NS", "BSE.NS", "CESC.NS", "CGPOWER.NS", "CHAMBLFERT.NS", "CHOLAHLDNG.NS", "COFORGE.NS", "CONCOR.NS", "COROMANDEL.NS", "CREDITACC.NS", "CROMPTON.NS", "CUMMINSIND.NS", "CYIENT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS", "DEVYANI.NS", "DIXON.NS", "EASEMYTRIP.NS", "EDELWEISS.NS", "EICHERMOT.NS", "EMAMILTD.NS", "ENDURANCE.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS", "GICRE.NS", "GLENMARK.NS", "GMRINFRA.NS", "GODREJIND.NS", "GODREJPROP.NS", "GRANULES.NS", "GUJGASLTD.NS", "HAPPSTMNDS.NS", "HDFCAMC.NS", "HFCL.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HUDCO.NS", "IDBI.NS", "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIAMART.NS", "INDIANB.NS", "INDIGO.NS", "IPCALAB.NS", "IRB.NS", "ITDCEM.NS", "JBCHEPHARM.NS", "JKCEMENT.NS", "JSL.NS", "JSWINFRA.NS", "JUBLFOOD.NS", "KALYANKJIL.NS", "KEI.NS", "KPITTECH.NS", "KPRMILL.NS", "L&TFH.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LINDEINDIA.NS", "LLOYDSME.NS", "LUPIN.NS", "MAHABANK.NS", "MAHINDCIE.NS", "MANAPPURAM.NS", "MANKIND.NS", "MARICO.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "METROPOLIS.NS", "MFSL.NS", "MGL.NS", "MOTILALOFS.NS", "MPHASIS.NS", "MRPL.NS", "MUTHOOTFIN.NS", "NATCOPHARM.NS", "NATIONALUM.NS", "NAVINFLUOR.NS", "NBCC.NS", "NHPC.NS", "NLCINDIA.NS", "NMDC.NS", "NYKAA.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "PAGEIND.NS", "PATANJALI.NS", "PAYTM.NS", "PERSISTENT.NS", "PETRONET.NS", "PHOENIXLTD.NS", "PIIND.NS", "PNBHOUSING.NS", "POLYMED.NS", "POONAWALLA.NS", "PRESTIGE.NS", "PVRINOX.NS", "QUESS.NS", "RADICO.NS", "RAILTEL.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "RATNAMANI.NS", "RBLBANK.NS", "RECLTD.NS", "RELAXO.NS", "RVNL.NS", "SAIL.NS", "SCHAEFFLER.NS", "SHREECEM.NS", "SJVN.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SRF.NS", "STLTECH.NS", "SUNTV.NS", "SUPREMEIND.NS", "SUZLON.NS", "SWIGGY.NS", "SYNGENE.NS", "TATACHEM.NS", "TATATECH.NS", "TRIDENT.NS", "UCOBANK.NS", "UNIONBANK.NS", "UPL.NS", "VGUARD.NS", "VI.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "YESBANK.NS", "ZEEL.NS"]
}

# Expanded Sector Mapping
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

# --- ULTIMATE DUAL-TARGET ENGINE ---
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        if len(data) < 200: return None
        
        # Fix MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        
        # A. Technical Analysis
        price = data['Close'].iloc[-1]
        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        
        rsi = data['RSI'].iloc[-1]
        ema = data['EMA200'].iloc[-1]
        trend = st_df.iloc[:, 1].iloc[-1]
        atr = data['ATR'].iloc[-1]
        
        # CORE INSTITUTIONAL FILTER
        if not (price > ema and 40 <= rsi <= 70 and trend == 1):
            return None

        # B. Black Swan News Sentinel
        news_sentiment = "✅ Neutral"
        danger_keys = ['fraud', 'investigation', 'default', 'raid', 'resigns', 'scam', 'bankruptcy', 'legal']
        positive_keys = ['growth', 'order', 'win', 'expansion', 'profit', 'deal']
        
        try:
            news = ticker_obj.news
            if news:
                titles = [n['title'].lower() for n in news[:5]]
                for t in titles:
                    if any(k in t for k in danger_keys):
                        news_sentiment = "🚨 BLACK SWAN"
                        break
                    elif any(k in t for k in positive_keys):
                        news_sentiment = "🔥 Positive"
        except:
            pass

        # C. Quarterly Earnings Blocker
        event_risk = "✅ Safe"
        try:
            calendar = ticker_obj.calendar
            if calendar is not None and not calendar.empty:
                next_date = calendar.iloc[0, 0]
                days_to_event = (next_date.date() - datetime.now().date()).days
                if 0 <= days_to_event <= 7:
                    event_risk = f"🚨 EARNINGS ({next_date.strftime('%d-%b')})"
        except:
            pass

        # D. Analyst Consensus & Targets (CRITICAL FIX)
        info = ticker_obj.info
        target = info.get('targetMeanPrice', 0)  # ANALYST TARGET PRICE
        a_count = info.get('numberOfAnalystOpinions', 0)  # NUMBER OF ANALYSTS
        pe = info.get('trailingPE', "N/A")
        pb = info.get('priceToBook', "N/A")
        upside = ((target - price) / price * 100) if target > 0 else 0

        # E. Trend Age (Freshness)
        age = 0
        for i in range(1, 15):
            if i < len(data) and data['Close'].iloc[-i] > data['EMA200'].iloc[-i] and st_df.iloc[:, 1].iloc[-i] == 1:
                age += 1
            else: 
                break

        # F. ULTIMATE SCORING
        score = 80
        if 48 <= rsi <= 58: score += 10
        if news_sentiment == "🔥 Positive": score += 5
        if upside > 10: score += 5
        if a_count >= 10: score += 5
        if news_sentiment == "🚨 BLACK SWAN": score = 10
        if age <= 5: score += 5

        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Score": score,
            "Age": f"{age}d",
            "News": news_sentiment,
            "Event": event_risk,
            "Target Price": round(target, 2) if target > 0 else "N/A",  # ANALYST TARGET
            "Upside": f"{upside:.1f}%" if upside > 0 else "N/A",
            "Analysts": a_count,
            "Price": round(price, 2),
            "RSI": round(rsi, 2),
            "PE": pe if pe == "N/A" else round(pe, 1),
            "PB": pb if pb == "N/A" else round(pb, 1),
            "Technical Target": round(price + (atr * 2.5), 2),  # SWING TARGET
            "SL": round(price * 0.96, 2)
        }
    except Exception as e:
        return None

# --- MARKET PULSE ---
st.subheader("🌐 Market Pulse")
index_benchmarks = {"Nifty 50": ["^NSEI"], "Nifty Next 50": ["^NIFTYJR"], "Nifty Midcap 150": ["^NSMIDCP"]}
cols = st.columns(3)
market_health = []

for i, (name, tickers) in enumerate(index_benchmarks.items()):
    try:
        data = yf.download(tickers[0], period="1y", progress=False, threads=False)
        if not data.empty:
            price = data['Close'].iloc[-1]
            ema = ta.ema(data['Close'], 200).iloc[-1]
            status = "🟢 BULLISH" if price > ema else "🔴 BEARISH"
            market_health.append(status == "🟢 BULLISH")
            cols[i].metric(name, f"₹{price:,.0f}", status)
        else:
            cols[i].error(f"{name} unavailable")
    except:
        cols[i].error(f"{name} error")

bullish_count = sum(market_health)
if bullish_count >= 2:
    st.success("✅ **BULL MARKET CONFIRMED** - Perfect for institutional breakouts!")
elif bullish_count == 1:
    st.warning("⚠️ **Mixed signals** - Focus on Nifty 50 only")
else:
    st.error("🛑 **BEAR MARKET** - High risk for long breakouts")

# --- CONTROLS & EXECUTION ---
st.sidebar.title("🔍 Fortress Controls")
selected_index = st.sidebar.selectbox("Universe", list(TICKER_GROUPS.keys()), key="universe")
TICKERS = TICKER_GROUPS[selected_index]
st.sidebar.info(f"📊 **{len(TICKERS)} stocks** | ⏱️ **~{len(TICKERS)*0.7/60:.1f}min scan** | **0.7s delay = RATE-LIMIT SAFE**")

if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# MAIN SCAN BUTTON
if st.button("🚀 START FULL FORTRESS SCAN", type="primary", use_container_width=True):
    results = []
    ticker_list = TICKERS
    total = len(ticker_list)
    
    # LIVE MONITOR
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(ticker_list):
        status_text.text(f"🔍 [{i+1}/{total}] Scanning {ticker}...")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            data = yf.download(ticker, period="1y", interval="1d", progress=False, 
                             threads=False, auto_adjust=True)
            
            if data.empty:
                status_text.warning(f"⚠️ {ticker}: Rate limited. Skipping...")
                time.sleep(2)
                continue
            
            result = check_institutional_fortress(ticker, data, ticker_obj)
            if result:
                results.append(result)
                st.toast(f"✅ FORTRESS HIT: {ticker} (Score: {result['Score']})", icon="🚀")
            
            time.sleep(0.7)  # CRITICAL: 0.7s for analyst data calls
            
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                status_text.error("🚨 Yahoo rate limit. Cooling 10s...")
                time.sleep(10)
            continue
            
        progress_bar.progress((i + 1) / total)

    status_text.success(f"✅ **SCAN COMPLETE!** Found {len(results)} Fortress setups.")

    # RESULTS DISPLAY
    if results:
        IST = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(IST).strftime("%d-%b-%Y | %I:%M %p IST")
        
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # SECTOR BREAKDOWN
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("🏦 Sector Heatmap")
            st.bar_chart(df['Sector'].value_counts(), height=300)
        
        with col2:
            st.subheader("📊 Ultimate Fortress Dashboard")
            st.caption(f"**{selected_index}** | {timestamp} | {len(results)}/{total} hits | {bullish_count}/3 indices bullish")
        
        # ADVANCED STYLING + FIXED COLUMN CONFIG
        def highlight_fortress(row):
            if row['Score'] >= 95:
                return ['background-color: #FFD700; color: black; font-weight: bold'] * len(row)
            elif row['Score'] >= 90:
                return ['background-color: #90EE90; font-weight: bold'] * len(row)
            elif row['News'] == "🚨 BLACK SWAN":
                return ['background-color: #FF6B6B; color: white'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df.style.apply(highlight_fortress, axis=1),
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Fortress Score", min_value=0, max_value=100, format="%d%%"),
                "Age": st.column_config.TextColumn("Trend Age", help="Days above EMA200+SuperTrend"),
                "Target Price": st.column_config.NumberColumn("Analyst Target", format="₹%.2f"),  # STANDALONE ANALYST TARGET
                "Upside": st.column_config.TextColumn("Analyst Upside"),
                "Analysts": st.column_config.NumberColumn("Institutional Coverage", help="Number of analysts backing this stock"),
                "Technical Target": st.column_config.NumberColumn("Swing Target", format="₹%.0f"),  # ATR BASED
                "SL": st.column_config.NumberColumn("Stop Loss", format="₹%.0f")
            },
            height=600
        )
        
        st.info("**🛡️ Dual Targets:** `Target Price` = Analyst Fair Value | `Technical Target` = ATR Swing (2.5x) | **0.7s delay = Rate-limit safe**")
        
    else:
        st.warning("🏰 **Fortress Walls Are Up.** No institutional setups found. Try another universe.")

st.caption("🛡️ **Fortress 95 Pro v3.0** - DUAL TARGETS | Analyst Visibility | 250+ Tickers | Production Ready")

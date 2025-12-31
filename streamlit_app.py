import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# --- HELPER FUNCTIONS ---
def clear_full_cache():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.toast("🧹 Cache cleared! Ready for a fresh scan.", icon="✅")

def reset_filters():
    st.session_state['use_analyst_filter'] = False
    st.session_state['min_analysts'] = 10
    st.session_state['max_age'] = 5
    st.sidebar.divider()
    st.toast("🔄 Filters reset to defaults.", icon="👍")

# 1. Page Config
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95: Professional Scanner")

# 2. YOUR Sector Mapping
SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT", "LTIM.NS": "IT", "TECHM.NS": "IT",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking", "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto", "BAJAJ-AUTO.NS": "Auto", "EICHERMOT.NS": "Auto",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma", "APOLLOHOSP.NS": "Healthcare",
    "LT.NS": "Construction", "ADANIENT.NS": "Conglomerate", "ADANIPORTS.NS": "Infrastructure",
    "BAJFINANCE.NS": "NBFC", "BAJAJFINSV.NS": "NBFC", "CHOLAFIN.NS": "NBFC", "SHRIRAMFIN.NS": "NBFC"
}

# YOUR TICKERS (unchanged)
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

# 2. ENHANCED Sidebar Controls with Keys
st.sidebar.title("🔍 Strategy Filters")

# Global Reset Button at the Top
if st.sidebar.button("🗑️ Reset All Filters"):
    reset_filters()
    st.rerun()

st.sidebar.divider()

# Keyed widgets for reset functionality
use_analyst_filter = st.sidebar.checkbox(
    "Filter by Analyst Support", 
    value=False, 
    key="use_analyst_filter"
)

min_analysts = st.sidebar.slider(
    "Min Analysts Required", 0, 50, 10, 
    key="min_analysts"
) if use_analyst_filter else 0

st.sidebar.divider()
st.sidebar.subheader("🕒 Entry Freshness")
max_age = st.sidebar.slider(
    "Max Trend Age (Days)", 1, 10, 5, 
    key="max_age"
)

# Capital Management
st.sidebar.divider()
st.sidebar.subheader("💰 Capital Management")
total_capital = st.sidebar.number_input(
    "Trading Capital (₹)", value=100000, 
    key="total_capital"
)
risk_per_trade = st.sidebar.slider(
    "Risk Per Trade (%)", 0.5, 3.0, 1.0, 
    key="risk_per_trade"
)

# Maintenance Section
st.sidebar.divider()
st.sidebar.subheader("⚙️ Maintenance")
if st.sidebar.button("🧹 Clear All Cache", help="Force-refreshes all stock data"):
    clear_full_cache()
    st.rerun()

# --- MARKET PULSE (NIFTY 50 CHECK) ---
st.subheader("🌐 Global Market Pulse")
try:
    nifty = yf.download("^NSEI", period="1y", interval="1d", progress=False, auto_adjust=True)
    nifty_price = nifty['Close'].iloc[-1]
    nifty_ema = ta.ema(nifty['Close'], length=200).iloc[-1]
    nifty_rsi = ta.rsi(nifty['Close'], length=14).iloc[-1]
    
    pulse_col1, pulse_col2, pulse_col3 = st.columns(3)
    
    if nifty_price > nifty_ema:
        market_status = "💹 BULLISH"
        pulse_col1.metric("Nifty 50 Trend", market_status, "Above EMA200")
    else:
        market_status = "📉 BEARISH"
        pulse_col1.metric("Nifty 50 Trend", market_status, "- Below EMA200", delta_color="inverse")
        
    pulse_col2.metric("Nifty RSI", f"{nifty_rsi:.1f}", "Neutral" if 40 < nifty_rsi < 60 else "Overextended")
    
    if market_status == "📉 BEARISH":
        st.error("⚠️ **Market Warning:** Nifty is below EMA200. High-probability setups may fail. Reduce position sizes!")
    else:
        st.success("✅ **Market Support:** Overall trend is Bullish. Institutional setups are favorable.")
except:
    st.write("Could not fetch Market Pulse. Proceed with caution.")

# 4. COMPLETE Logic Engine + FUNDAMENTALS (Updated to use sidebar vars)
@st.cache_data(ttl=600)
def check_institutional_fortress(ticker, total_capital, risk_per_trade, use_analyst_filter, min_analysts, max_age):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        if len(data) < 200: return None

        # --- TECHNICALS & AGE ---
        price = data['Close'].iloc[-1]
        data['EMA200'] = ta.ema(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        
        rsi = data['RSI'].iloc[-1]
        ema = data['EMA200'].iloc[-1]
        trend = st_df.iloc[:, 1].iloc[-1]
        sl_price = round(price * 0.96, 2)
        target_price = round(price + (data['ATR'].iloc[-1] * 2.5), 2)

        # Entry Age Logic
        days_in_trend = 0
        for i in range(1, 11):
            check_price = data['Close'].iloc[-i]
            check_ema = data['EMA200'].iloc[-i]
            check_trend = st_df.iloc[:, 1].iloc[-i]
            if (check_price > check_ema) and (check_trend == 1):
                days_in_trend += 1
            else: break
        
        if days_in_trend > max_age: return None

        # --- FUNDAMENTALS (NON-BLOCKING) ---
        info = ticker_obj.info
        pe = info.get('trailingPE', 0)
        pb = info.get('priceToBook', 0)
        valuation_label = "💎 Value" if (pe > 0 and pe < 25) else "🚀 Premium" if pe > 60 else "📊 Fair"
        
        # CORPORATE EVENTS (Hard Block for immediate results)
        calendar = ticker_obj.calendar
        event_warning = "✅ Clear"
        score = 0
        
        if calendar is not None and not calendar.empty:
            upcoming_date = calendar.iloc[0, 0]
            days_to_event = (upcoming_date.date() - datetime.now().date()).days
            
            if 0 <= days_to_event <= 2:
                return None  # HARD BLOCK
            elif 3 <= days_to_event <= 7:
                event_warning = f"⚠️ Results ({upcoming_date.strftime('%d-%b')})"
                score -= 20

        # Status Logic
        is_fresh_buy = (price > ema) and (45 <= rsi <= 65) and (trend == 1)
        is_trending_hold = (price > ema) and (65 < rsi < 75) and (trend == 1)
        if is_fresh_buy: status = "🚀 BUY"
        elif is_trending_hold: status = "📈 TRENDING"
        elif rsi >= 75: status = "✋ OVERBOUGHT"
        else: status = "🚫 AVOID"

        # Analyst Filter
        analyst_count = info.get('numberOfAnalystOpinions', 0)
        if use_analyst_filter and analyst_count < min_analysts: return None
        expert_target = info.get('targetMeanPrice', 0)

        # NEWS SENTINEL
        news_data = ticker_obj.news
        news_alert = "✅ Neutral"
        danger_keywords = ['fraud', 'investigation', 'default', 'bankruptcy', 'raid', 'resigns', 'scam', 'penalty', 'legal']
        
        if news_data:
            recent_titles = [n['title'].lower() for n in news_data[:5]]
            for title in recent_titles:
                if any(k in title for k in danger_keywords):
                    news_alert = "🚨 BLACK SWAN"
                    break
                elif any(k in title for k in ['growth', 'order', 'win', 'expansion', 'profit']):
                    news_alert = "🔥 Positive"

        news_link = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"

        # COMPLETE Conviction Score
        if trend == 1: score += 30
        if price > ema: score += 20
        if 48 <= rsi <= 58: score += 30
        elif 45 <= rsi <= 65: score += 15
        if days_in_trend >= 2: score += 10
        if expert_target and expert_target > price: score += 10
        if news_alert == "🔥 Positive": score += 10
        if news_alert == "🚨 BLACK SWAN": score -= 60
        if valuation_label == "💎 Value": score += 15

        # Momentum
        prev_rsi = data['RSI'].iloc[-2]
        if rsi > prev_rsi: momentum_icon = "🔼 Increasing"
        elif rsi < prev_rsi: momentum_icon = "🔽 Slowing"
        else: momentum_icon = "➡️ Stable"

        # RR + Position Sizing
        risk = price - sl_price
        reward = target_price - price
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        rupees_at_risk = total_capital * (risk_per_trade / 100)
        per_share_risk = price - sl_price
        if per_share_risk > 0:
            suggested_qty = int(rupees_at_risk / per_share_risk)
            total_investment = suggested_qty * price
        else:
            suggested_qty = 0
            total_investment = 0

        return {
            "Symbol": ticker,
            "News Status": news_alert,
            "Read News": news_link,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Valuation": valuation_label,
            "Event Risk": event_warning,
            "Age": f"{days_in_trend} Days",
            "Momentum": momentum_icon,
            "RR Ratio": rr_ratio,
            "Qty": suggested_qty,
            "Invest": round(total_investment, 0),
            "Conviction Score": score,
            "Status": status,
            "Price": round(price, 2),
            "2-Week (ATR) Target": target_price,
            "Analysts 👤": analyst_count,
            "Expert Target": round(expert_target, 2) if expert_target else "N/A",
            "RSI": round(rsi, 2),
            "P/E": round(pe, 1) if pe else "N/A",
            "SL": sl_price
        }
    except: return None

# 5. Execution + SMART FILTER Display
if st.button("🚀 Start Fortress Scan"):
    results = []
    with st.status("Scanning Nifty...", expanded=True):
        bar = st.progress(0)
        for i, t in enumerate(TICKERS):
            res = check_institutional_fortress(t, total_capital, risk_per_trade, use_analyst_filter, min_analysts, max_age)
            if res: results.append(res)
            bar.progress((i + 1) / len(TICKERS))

    if results:
        IST = pytz.timezone('Asia/Kolkata')
        timestamp_str = datetime.now(IST).strftime("%d-%b-%Y | %I:%M:%S %p")
        
        df = pd.DataFrame(results)
        df = df.sort_values(by="Conviction Score", ascending=False)

        # Sector Chart
        st.subheader("🏦 Sector Distribution")
        sector_sums = df.groupby('Sector')['Invest'].sum()
        st.bar_chart(sector_sums)

        # Enhanced Highlighting
        def highlight_rows(row):
            if row['News Status'] == "🚨 BLACK SWAN":
                return ['background-color: #9b1c1c; color: white; font-weight: bold'] * len(row)
            elif row['Conviction Score'] >= 90 and row['RR Ratio'] >= 1.5:
                return ['background-color: #FFD700; color: black; font-weight: bold'] * len(row)
            return [''] * len(row)

        st.subheader("📊 Fortress 95 Dashboard")
        st.caption(f"🕒 **Last Scan (IST):** {timestamp_str}")
        st.write("**UI Controls:** 🗑️ Reset Filters → 🚀 Run Scan → 🧹 Clear Cache (if stale)")

        st.dataframe(
            df.style.apply(highlight_rows, axis=1),
            use_container_width=True,
            column_config={
                "News Status": st.column_config.TextColumn("Sentiment", help="🚨 Black Swan = Exit Immediately"),
                "Read News": st.column_config.LinkColumn("Verify News 🔗"),
                "Valuation": st.column_config.TextColumn("Type", help="💎 Value = Safer | 🚀 Premium = Fast but Risky"),
                "Event Risk": st.column_config.TextColumn("Events", help="⚠️ Results soon = Higher volatility"),
                "Sector": st.column_config.TextColumn("Sector"),
                "Conviction Score": st.column_config.ProgressColumn("Confidence", help="Technicals+Fundamentals+News", min_value=0, max_value=100, format="%d%%"),
                "Momentum": st.column_config.TextColumn("Momentum"),
                "RR Ratio": st.column_config.NumberColumn("Risk:Reward", format="%.2fx"),
                "Qty": st.column_config.NumberColumn("Shares", format="%d"),
                "Invest": st.column_config.NumberColumn("Investment", format="₹%d"),
                "Status": st.column_config.TextColumn("Signal"),
                "P/E": st.column_config.NumberColumn("P/E Ratio", format="%.1f")
            }
        )
    else:
        st.warning("No fresh breakouts found.")

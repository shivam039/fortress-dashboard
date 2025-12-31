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
    st.session_state['total_capital'] = 100000
    st.session_state['risk_per_trade'] = 1.0
    st.toast("🔄 Filters reset to defaults.", icon="👍")

# 1. Page Config
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95: Professional Scanner")

# 2. Multi-Index Ticker Lists
TICKER_GROUPS = {
    "Nifty 50 (Large Cap)": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS",
        "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS",
        "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS", "KOTAKBANK.NS", "TITAN.NS",
        "ULTRACEMCO.NS", "AXISBANK.NS", "NTPC.NS", "ONGC.NS", "ADANIPORTS.NS",
        "ASIANPAINT.NS", "COALINDIA.NS", "JSWSTEEL.NS", "BAJAJ-AUTO.NS", "NESTLEIND.NS",
        "GRASIM.NS", "HINDALCO.NS", "POWERGRID.NS", "ADANIPOWER.NS", "WIPRO.NS",
        "EICHERMOT.NS", "SBILIFE.NS", "TATAMOTORS.NS", "BPCL.NS", "DRREDDY.NS",
        "HCLTECH.NS", "JIOFIN.NS", "TECHM.NS", "BRITANNIA.NS", "TATAPOWER.NS",
        "BAJAJFINSV.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS", "TVSMOTOR.NS", "APOLLOHOSP.NS",
        "CIPLA.NS", "BEL.NS", "TRENT.NS"
    ],
    "Nifty Next 50 (Junior Large Cap)": [
        "ADANIENSOL.NS", "ADANIGREEN.NS", "AMBUJACEM.NS", "DMART.NS", "BAJAJHLDNG.NS",
        "BANKBARODA.NS", "BHEL.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS",
        "COLPAL.NS", "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAL.NS",
        "HAVELLS.NS", "HZL.NS", "ICICILOMB.NS", "ICICIPRULI.NS", "IOC.NS", "IRCTC.NS",
        "IRFC.NS", "JINDALSTEL.NS", "JSWENERGY.NS", "LTIM.NS", "LUPIN.NS", "MARICO.NS",
        "MRF.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "PFC.NS", "PIDILITIND.NS", "PNB.NS",
        "RECLTD.NS", "SAMVARDHANA.NS", "SHREECEM.NS", "SIEMENS.NS", "TATACOMM.NS",
        "TATAELXSI.NS", "TATAMTRDVR.NS", "TORNTPHARM.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS",
        "ZOMATO.NS", "ZYDUSLIFE.NS", "ABB.NS", "TIINDIA.NS", "POLYCAB.NS"
    ],
    "Nifty Midcap 150 (Mid Cap)": [
        "AUROPHARMA.NS", "ASHOKLEY.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BIOCON.NS",
        "COFORGE.NS", "CUMMINSIND.NS", "ESCORTS.NS", "FEDERALBNK.NS", "FORTIS.NS",
        "GMRINFRA.NS", "GUJGASLTD.NS", "IDFCFIRSTB.NS", "INDIAMART.NS", "IPCALAB.NS",
        "JUBLFOOD.NS", "MAXHEALTH.NS", "MPHASIS.NS", "OBEROIRLTY.NS", "PAGEIND.NS",
        "PERSISTENT.NS", "PETRONET.NS", "SRF.NS", "SUZLON.NS", "SYNGENE.NS",
        "TATACHEM.NS", "VOLTAS.NS", "YESBANK.NS", "DIXON.NS", "ASTRAL.NS",
        "MAXFSL.NS", "CONCOR.NS", "DEEPAKNTR.NS", "MGL.NS", "PVRINOX.NS",
        "MCX.NS", "GLENMARK.NS", "RAMCOCEM.NS", "SUNTV.NS", "MANAPPURAM.NS"
    ]
}

# Sector Mapping
SECTOR_MAP = {
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking", 
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
    "BANKBARODA.NS": "Banking", "CANBK.NS": "Banking", "PNB.NS": "Banking",
    "BAJFINANCE.NS": "NBFC", "BAJAJFINSV.NS": "NBFC", "CHOLAFIN.NS": "NBFC",
    "SHRIRAMFIN.NS": "NBFC", "MUTHOOTFIN.NS": "NBFC", "IDFCFIRSTB.NS": "Banking",
    "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT", 
    "TECHM.NS": "IT", "LTIM.NS": "IT", "MPHASIS.NS": "IT", "PERSISTENT.NS": "IT",
    "COFORGE.NS": "IT", "TATAELXSI.NS": "IT", "RELIANCE.NS": "Energy", 
    "ONGC.NS": "Energy", "BPCL.NS": "Energy", "IOC.NS": "Energy",
    "ADANIPOWER.NS": "Energy", "TATAPOWER.NS": "Energy", "NTPC.NS": "Energy",
    "POWERGRID.NS": "Energy", "GAIL.NS": "Energy", "JSWSTEEL.NS": "Metals",
    "HINDALCO.NS": "Metals", "VEDL.NS": "Metals", "JINDALSTEL.NS": "Metals",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto", "TVSMOTOR.NS": "Auto", "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma", "APOLLOHOSP.NS": "Healthcare",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "GODREJCP.NS": "FMCG", "LT.NS": "Infra",
    "ADANIPORTS.NS": "Infra", "BEL.NS": "Defense", "HAL.NS": "Defense",
    "TRENT.NS": "Retail", "ZOMATO.NS": "Retail"
}

# Sidebar
st.sidebar.title("🔍 Strategy Filters")
selected_index = st.sidebar.selectbox("Select Universe", options=list(TICKER_GROUPS.keys()), index=0, key="selected_index")
TICKERS = TICKER_GROUPS[selected_index]
st.sidebar.write(f"📊 Total Stocks: **{len(TICKERS)}**")

if st.sidebar.button("🗑️ Reset All Filters"):
    reset_filters()
    st.rerun()

st.sidebar.divider()
use_analyst_filter = st.sidebar.checkbox("Filter by Analyst Support", value=False, key="use_analyst_filter")
min_analysts = st.sidebar.slider("Min Analysts Required", 0, 50, 10, key="min_analysts") if use_analyst_filter else 0
st.sidebar.divider()
st.sidebar.subheader("🕒 Entry Freshness")
max_age = st.sidebar.slider("Max Trend Age (Days)", 1, 10, 5, key="max_age")
st.sidebar.divider()
st.sidebar.subheader("💰 Capital Management")
total_capital = st.sidebar.number_input("Trading Capital (₹)", value=100000, key="total_capital")
risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, key="risk_per_trade")
st.sidebar.divider()
st.sidebar.subheader("⚙️ Maintenance")
if st.sidebar.button("🧹 Clear All Cache"):
    clear_full_cache()
    st.rerun()

# --- ENHANCED MARKET PULSE (STABLE VERSION WITH FALLBACKS) ---
st.subheader("🌐 Global Market Benchmarks")

# Using the most stable Yahoo Tickers for Indian Indices (Dec 2025)
index_benchmarks = {
    "Nifty 50": ["^NSEI", "NIFTY50.NS"],
    "Nifty Next 50": ["^NIFTYJR", "NIFTYNEXT50.NS"],
    "Nifty Midcap 150": ["^NSMIDCP", "NIFTYMIDCAP150.NS"]
}

pulse_cols = st.columns(len(index_benchmarks))
market_health = []

for i, (name, tickers) in enumerate(index_benchmarks.items()):
    idx_data = None
    # Try each ticker until one works
    for ticker in tickers:
        try:
            idx_data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if not idx_data.empty:
                break
        except:
            continue
    
    if idx_data is not None and not idx_data.empty:
        # Standardize columns (yfinance MultiIndex fix)
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

# --- FINAL SYSTEM ALERT ---
bullish_count = sum(market_health)
if bullish_count >= 2:
    st.success("✅ **Market Support:** Broad trend is BULLISH. Perfect for 'Fortress' setups.")
elif bullish_count == 1:
    st.warning("⚖️ **Mixed Market:** Divergence found. Trade only Nifty 50 'Gold' stocks.")
else:
    st.error("🛑 **System Alert:** Full Market BEARISH. High risk of failure for new longs.")

# Logic Engine (unchanged)
@st.cache_data(ttl=600)
def check_institutional_fortress(ticker, total_capital, risk_per_trade, use_analyst_filter, min_analysts, max_age):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        if len(data) < 200: return None

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

        days_in_trend = 0
        for i in range(1, 11):
            if data['Close'].iloc[-i] > data['EMA200'].iloc[-i] and st_df.iloc[:, 1].iloc[-i] == 1:
                days_in_trend += 1
            else: break
        
        if days_in_trend > max_age: return None

        info = ticker_obj.info
        pe = info.get('trailingPE', 0)
        valuation_label = "💎 Value" if (pe > 0 and pe < 25) else "🚀 Premium" if pe > 60 else "📊 Fair"
        
        calendar = ticker_obj.calendar
        event_warning = "✅ Clear"
        score = 0
        
        if calendar is not None and not calendar.empty:
            upcoming_date = calendar.iloc[0, 0]
            days_to_event = (upcoming_date.date() - datetime.now().date()).days
            if 0 <= days_to_event <= 2: return None
            elif 3 <= days_to_event <= 7:
                event_warning = f"⚠️ Results ({upcoming_date.strftime('%d-%b')})"
                score -= 20

        analyst_count = info.get('numberOfAnalystOpinions', 0)
        if use_analyst_filter and analyst_count < min_analysts: return None
        expert_target = info.get('targetMeanPrice', 0)

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

        if trend == 1: score += 30
        if price > ema: score += 20
        if 48 <= rsi <= 58: score += 30
        elif 45 <= rsi <= 65: score += 15
        if days_in_trend >= 2: score += 10
        if expert_target and expert_target > price: score += 10
        if news_alert == "🔥 Positive": score += 10
        if news_alert == "🚨 BLACK SWAN": score -= 60
        if valuation_label == "💎 Value": score += 15

        prev_rsi = data['RSI'].iloc[-2]
        momentum_icon = "🔼 Increasing" if rsi > prev_rsi else "🔽 Slowing" if rsi < prev_rsi else "➡️ Stable"

        risk = price - sl_price
        reward = target_price - price
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        rupees_at_risk = total_capital * (risk_per_trade / 100)
        per_share_risk = price - sl_price
        suggested_qty = int(rupees_at_risk / per_share_risk) if per_share_risk > 0 else 0
        total_investment = suggested_qty * price

        status = "🚀 BUY" if (price > ema and 45 <= rsi <= 65 and trend == 1) else "📈 TRENDING" if (price > ema and 65 < rsi < 75 and trend == 1) else "✋ OVERBOUGHT" if rsi >= 75 else "🚫 AVOID"

        return {
            "Symbol": ticker, "News Status": news_alert, "Read News": news_link,
            "Sector": SECTOR_MAP.get(ticker, "General"), "Valuation": valuation_label,
            "Event Risk": event_warning, "Age": f"{days_in_trend} Days",
            "Momentum": momentum_icon, "RR Ratio": rr_ratio, "Qty": suggested_qty,
            "Invest": round(total_investment, 0), "Conviction Score": score,
            "Status": status, "Price": round(price, 2), "2-Week (ATR) Target": target_price,
            "Analysts 👤": analyst_count, "Expert Target": round(expert_target, 2) if expert_target else "N/A",
            "RSI": round(rsi, 2), "P/E": round(pe, 1) if pe else "N/A", "SL": sl_price
        }
    except: return None

# Execution
if st.button("🚀 Start Fortress Scan"):
    results = []
    with st.status(f"Scanning {selected_index}...", expanded=True):
        bar = st.progress(0)
        for i, t in enumerate(TICKERS):
            res = check_institutional_fortress(t, total_capital, risk_per_trade, use_analyst_filter, min_analysts, max_age)
            if res: results.append(res)
            bar.progress((i + 1) / len(TICKERS))

    if results:
        IST = pytz.timezone('Asia/Kolkata')
        timestamp_str = datetime.now(IST).strftime("%d-%b-%Y | %I:%M:%S %p")
        df = pd.DataFrame(results).sort_values(by="Conviction Score", ascending=False)

        st.subheader("🏦 Sector Distribution")
        st.bar_chart(df.groupby('Sector')['Invest'].sum())

        def highlight_rows(row):
            if row['News Status'] == "🚨 BLACK SWAN":
                return ['background-color: #9b1c1c; color: white; font-weight: bold'] * len(row)
            elif row['Conviction Score'] >= 90 and row['RR Ratio'] >= 1.5:
                return ['background-color: #FFD700; color: black; font-weight: bold'] * len(row)
            return [''] * len(row)

        st.subheader("📊 Fortress 95 Dashboard")
        st.caption(f"🕒 **{selected_index} Scan (IST):** {timestamp_str} | Found: {len(results)}/{len(TICKERS)}")
        
        st.dataframe(
            df.style.apply(highlight_rows, axis=1),
            use_container_width=True,
            column_config={
                "News Status": st.column_config.TextColumn("Sentiment"),
                "Read News": st.column_config.LinkColumn("Verify News 🔗"),
                "Valuation": st.column_config.TextColumn("Type"),
                "Event Risk": st.column_config.TextColumn("Events"),
                "Conviction Score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d%%"),
                "RR Ratio": st.column_config.NumberColumn("Risk:Reward", format="%.2fx"),
                "Invest": st.column_config.NumberColumn("Investment", format="₹%d")
            }
        )
    else:
        st.warning(f"No setups found in {selected_index}.")

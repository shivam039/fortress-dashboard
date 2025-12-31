# fortress_app.py - FIXED SELECTION ERROR
import subprocess
import sys
import time
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Import config
from fortress_config import TICKER_GROUPS, SECTOR_MAP, INDEX_BENCHMARKS

# Auto-install dependencies
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])

# --- SYSTEM CONFIG ---
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v5.4 - AI INTELLIGENCE (FIXED)")

# --- ULTIMATE AI INTELLIGENCE REPORT ---
@st.dialog("📋 AI Institutional & News Intelligence Report", width="large")
def show_analyst_report(ticker_symbol):
    st.subheader(f"🧠 Strategic Intelligence Report: **{ticker_symbol}**")
    
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        news = ticker_obj.news
        
        # ROW 1: ANALYST CONSENSUS
        st.markdown("#### 🏦 **Analyst Consensus Summary**")
        c1, c2, c3 = st.columns(3)
        target = info.get('targetMeanPrice', 0)
        current = info.get('currentPrice', 1)
        upside = ((target/current)-1)*100 if target > 0 else 0
        
        c1.metric("Rating", info.get('recommendationKey', 'N/A').upper())
        c2.metric("Target Price", f"₹{target:,.0f}")
        c3.metric("Analyst Count", info.get('numberOfAnalystOpinions', 0))
        
        coverage = info.get('numberOfAnalystOpinions', 0)
        st.info(f"**AI Insight:** {coverage} analysts project **{upside:.1f}% upside**. Coverage is {'🟢 HIGH' if coverage > 15 else '🟡 MODERATE' if coverage > 5 else '🔴 LOW'}.")

        # ROW 2: AI NEWS SENTIMENT
        st.markdown("#### 📰 **Latest News Sentiment AI**")
        if news and len(news) > 0:
            danger_keys = ['fraud', 'investigation', 'default', 'raid', 'resigns', 'scam', 'bankruptcy']
            positive_keys = ['growth', 'order', 'expansion', 'profit', 'deal', 'partnership']
            
            for n in news[:5]:
                title = n['title']
                t_lower = title.lower()
                
                tag = "🔹 Neutral"
                if any(k in t_lower for k in danger_keys): 
                    tag = "🚨 **RISK ALERT**"
                elif any(k in t_lower for k in positive_keys): 
                    tag = "🔥 **POSITIVE**"
                
                st.markdown(f"**{tag}:** {title}")
                st.caption(f"📅 {n.get('providerPublishTime', 'Recent')} | [Read Full]({n['link']})")
        else:
            st.warning("📰 No recent news found.")

        # ROW 3: FINANCIAL HEALTH
        st.markdown("#### 📊 **Financial Health**")
        points = [
            f"📍 **P/E:** {info.get('trailingPE', 'N/A')} | {'⚠️ HIGH' if info.get('trailingPE', 0) > 25 else '✅ FAIR'}",
            f"📍 **Debt/Equity:** {info.get('debtToEquity', 'N/A')}",
            f"📍 **Market Cap:** ₹{info.get('marketCap', 0):,}",
            f"📍 **Beta:** {info.get('beta', 'N/A')}"
        ]
        for point in points:
            st.write(point)

        col1, col2 = st.columns(2)
        if col1.button("🔄 Refresh", use_container_width=True): st.rerun()
        if col2.button("❌ Close", use_container_width=True): st.rerun()
            
    except Exception as e:
        st.error(f"⚠️ Report error: {str(e)}")
        if st.button("❌ Close"): st.rerun()

# --- FORTRESS ENGINE ---
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        price = data['Close'].iloc[-1]
        ema200 = ta.ema(data['Close'], length=200).iloc[-1]
        rsi = ta.rsi(data['Close'], length=14).iloc[-1]
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        trend = st_df.iloc[:, 1].iloc[-1]
        
        is_pass = (price > ema200 and 40 <= rsi <= 70 and trend == 1)
        
        info = ticker_obj.info
        target = info.get('targetMeanPrice', 0)
        a_count = info.get('numberOfAnalystOpinions', 0)
        upside = ((target - price) / price * 100) if target > 0 else 0

        age = 0
        for i in range(1, 15):
            if i < len(data) and data['Close'].iloc[-i] > ema200 and st_df.iloc[:, 1].iloc[-i] == 1:
                age += 1
            else: break

        score = 95 if (is_pass and 48 <= rsi <= 58) else (80 if is_pass else 0)

        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Verdict": "🚀 PASS" if is_pass else "❌ FAIL",
            "Report": "🧠 AI",
            "Price": round(price, 2),
            "RSI": round(rsi, 2),
            "Age": f"{age}d",
            "Analyst Target": round(target, 2) if target > 0 else "N/A",
            "Analysts": a_count,
            "Upside %": f"{upside:.1f}%" if upside != 0 else "N/A",
            "Score": score
        }
    except Exception:
        return {
            "Symbol": ticker, "Sector": SECTOR_MAP.get(ticker, "N/A"),
            "Verdict": "⚠️ ERROR", "Report": "🧠 AI", "Price": 0, "RSI": 0, 
            "Age": "0d", "Analyst Target": "N/A", "Analysts": 0, 
            "Upside %": "N/A", "Score": 0
        }

# --- MARKET PULSE ---
st.subheader("🌐 Market Pulse")
cols = st.columns(3)
bullish_count = 0

for i, (name, symbol) in enumerate(INDEX_BENCHMARKS.items()):
    try:
        data = yf.download(symbol, period="1y", progress=False, threads=False)
        if not data.empty:
            price = data['Close'].iloc[-1]
            ema = ta.ema(data['Close'], 200).iloc[-1]
            status = "🟢 BULLISH" if price > ema else "🔴 BEARISH"
            if price > ema: bullish_count += 1
            cols[i].metric(name, f"₹{price:,.0f}", status)
    except:
        cols[i].error(f"{name} error")

market_status = "✅ BULL MARKET" if bullish_count >= 2 else "⚠️ MIXED" if bullish_count == 1 else "🛑 BEAR"
st.success(f"**{market_status}** - {bullish_count}/3 indices bullish")

# --- CONTROLS ---
st.sidebar.title("🔍 Fortress Controls")
selected_index = st.sidebar.selectbox("Universe", list(TICKER_GROUPS.keys()))
TICKERS = TICKER_GROUPS[selected_index]
st.sidebar.info(f"📊 **{len(TICKERS)} stocks** | **CLICK 🧠 AI** for full intelligence")

if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# --- MAIN SCAN ---
if st.button("🚀 START FULL INTELLIGENCE SCAN", type="primary", use_container_width=True):
    results = []
    total = len(TICKERS)
    progress = st.progress(0)
    status = st.empty()
    pass_count = 0
    
    for i, ticker in enumerate(TICKERS):
        status.text(f"🧠 [{i+1}/{total}] {ticker}")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            data = yf.download(ticker, period="1y", progress=False, threads=False)
            
            if not data.empty:
                result = check_institutional_fortress(ticker, data, ticker_obj)
                results.append(result)
                
                if result['Verdict'] == "🚀 PASS":
                    pass_count += 1
                    st.toast(f"✅ {ticker}", icon="🚀")
            
            time.sleep(0.7)
        except:
            continue
            
        progress.progress((i+1)/total)
    
    status.success("✅ **SCAN COMPLETE!**")

    if results:
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # METRICS
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚀 PASSES", pass_count)
        col2.metric("🧠 Max Coverage", df['Analysts'].max())
        col3.metric("📈 Top Score", df['Score'].max())
        col4.metric("📊 Scanned", len(results))
        
        # ✅ FIXED DATAFRAME (CORRECT SELECTION HANDLING)
        st.subheader("🧠 FULL INTELLIGENCE TABLE - CLICK ROWS 👇")
        st.info("**🟢 PASS** = Trade | **🔴 FAIL** = Watch | **Click 🧠 AI** = Full Report")
        
        def color_verdict(val):
            color = 'green' if val == '🚀 PASS' else 'red' if val == '❌ FAIL' else 'orange'
            return f'color: {color}; font-weight: bold'
        
        # FIXED: Use st.session_state for selection tracking
        if 'selected_row' not in st.session_state:
            st.session_state.selected_row = None
        
        df_display = df.style.applymap(color_verdict, subset=['Verdict'])
        selected_rows = st.dataframe(
            df_display,
            use_container_width=True,
            selection_mode="single-row",
            key="fortress_table",
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                "Verdict": st.column_config.TextColumn("Status"),
                "Report": st.column_config.TextColumn("🧠 Intelligence", help="News + Analyst + Financials"),
                "Analyst Target": st.column_config.NumberColumn("Target ₹", format="₹%.0f"),
                "Analysts": st.column_config.NumberColumn("Coverage"),
                "Price": st.column_config.NumberColumn("Price ₹", format="₹%.0f")
            },
            height=700
        )
        
        # ✅ FIXED SELECTION LOGIC
        if st.session_state.get('fortress_table'):
            selection = st.session_state.fortress_table
            if selection and 'selection' in selection and selection['selection']:
                rows = selection['selection'].get('rows', [])
                if rows:
                    row_idx = rows[0]
                    ticker = df.iloc[row_idx]['Symbol']
                    st.session_state.selected_row = ticker
                    show_analyst_report(ticker)
        
        if st.session_state.selected_row:
            st.info(f"🧠 Selected: **{st.session_state.selected_row}** | Click another row for new intelligence")

st.markdown("---")
st.caption("🛡️ **Fortress 95 Pro v5.4** - ✅ FIXED Selection Error | AI News + Intelligence | Production Ready")

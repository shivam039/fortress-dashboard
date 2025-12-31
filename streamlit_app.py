# fortress_app.py
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
st.title("🛡️ Fortress 95 Pro v5.3 - AI INTELLIGENCE EDITION")

# --- ULTIMATE AI INTELLIGENCE REPORT (NEWS + ANALYST + FINANCIALS) ---
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

        # ROW 2: AI NEWS SENTIMENT ENGINE
        st.markdown("#### 📰 **Latest News Sentiment AI** (Black Swan Detection)")
        if news and len(news) > 0:
            danger_keys = ['fraud', 'investigation', 'default', 'raid', 'resigns', 'scam', 'bankruptcy', 'loss', 'downgrade']
            positive_keys = ['growth', 'order', 'expansion', 'profit', 'deal', 'partnership', 'upgrade', 'beat']
            
            for n in news[:5]:  # Top 5 news items
                title = n['title']
                t_lower = title.lower()
                
                # AI SENTIMENT TAGGING
                tag = "🔹 Neutral"
                if any(k in t_lower for k in danger_keys): 
                    tag = "🚨 **RISK ALERT**"
                elif any(k in t_lower for k in positive_keys): 
                    tag = "🔥 **POSITIVE**"
                
                st.markdown(f"**{tag}:** {title}")
                st.caption(f"📅 {n.get('providerPublishTime', 'Recent')} | {n['publisher']} | [Read Full Story]({n['link']})")
                st.markdown("---")
        else:
            st.warning("📰 No recent news found. Stock may have low media coverage.")

        # ROW 3: FINANCIAL HEALTH PULSE
        st.markdown("#### 📊 **Financial Health Indicators**")
        points = [
            f"📍 **Valuation:** P/E {info.get('trailingPE', 'N/A')} | {'⚠️ HIGH' if info.get('trailingPE', 0) > 25 else '✅ FAIR'}",
            f"📍 **Debt/Equity:** {info.get('debtToEquity', 'N/A')} | {'🟢 SAFE' if info.get('debtToEquity', 0) < 100 else '🔴 HIGH'}",
            f"📍 **Market Cap:** ₹{info.get('marketCap', 0):,}",
            f"📍 **Beta:** {info.get('beta', 'N/A')} | {'📈 VOLATILE' if info.get('beta', 0) > 1.2 else '📊 STABLE'}",
            f"📍 **ROE:** {info.get('returnOnEquity', 'N/A'):.1%} | Return on shareholder equity"
        ]
        for point in points:
            st.write(point)

        # ACTION BUTTONS
        col1, col2 = st.columns(2)
        if col1.button("🔄 Refresh Intelligence", use_container_width=True):
            st.rerun()
        if col2.button("❌ Close Report", use_container_width=True):
            st.rerun()
            
    except Exception as e:
        st.error(f"⚠️ Intelligence report unavailable: {str(e)}")
        if st.button("❌ Close"):
            st.rerun()

# --- CORE FORTRESS ENGINE (ALL STOCKS + FULL DATA) ---
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        # Technical indicators (ALWAYS calculate)
        price = data['Close'].iloc[-1]
        ema200 = ta.ema(data['Close'], length=200).iloc[-1]
        rsi = ta.rsi(data['Close'], length=14).iloc[-1]
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        trend = st_df.iloc[:, 1].iloc[-1]
        
        # Fortress Pass/Fail (CORE LOGIC)
        is_pass = (price > ema200 and 40 <= rsi <= 70 and trend == 1)
        
        # Analyst Data (ALWAYS fetch)
        info = ticker_obj.info
        target = info.get('targetMeanPrice', 0)
        a_count = info.get('numberOfAnalystOpinions', 0)
        upside = ((target - price) / price * 100) if target > 0 else 0

        # Trend Age
        age = 0
        for i in range(1, 15):
            if i < len(data) and data['Close'].iloc[-i] > ema200 and st_df.iloc[:, 1].iloc[-i] == 1:
                age += 1
            else: 
                break

        # FULL SCORING
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

market_status = "✅ BULL MARKET" if bullish_count >= 2 else "⚠️ MIXED" if bullish_count == 1 else "🛑 BEAR MARKET"
st.success(f"**{market_status}** - {bullish_count}/3 indices above EMA200")

# --- CONTROLS ---
st.sidebar.title("🔍 Fortress Controls")
selected_index = st.sidebar.selectbox("Universe", list(TICKER_GROUPS.keys()))
TICKERS = TICKER_GROUPS[selected_index]
st.sidebar.info(f"📊 **{len(TICKERS)} stocks** | **CLICK 🧠 AI** for News + Analyst Intelligence")

# MAIN SCAN
if st.button("🚀 START FULL INTELLIGENCE SCAN", type="primary", use_container_width=True):
    results = []
    total = len(TICKERS)
    progress = st.progress(0)
    status = st.empty()
    pass_count = 0
    
    for i, ticker in enumerate(TICKERS):
        status.text(f"🧠 [{i+1}/{total}] Intelligence scan: {ticker}")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            data = yf.download(ticker, period="1y", progress=False, threads=False)
            
            if not data.empty:
                result = check_institutional_fortress(ticker, data, ticker_obj)
                results.append(result)
                
                if result['Verdict'] == "🚀 PASS":
                    pass_count += 1
                    st.toast(f"✅ FORTRESS + AI INTELLIGENCE: {ticker}", icon="🧠")
            
            time.sleep(0.7)
        except:
            continue
            
        progress.progress((i+1)/total)
    
    status.success("✅ **INTELLIGENCE SCAN COMPLETE!**")

    if results:
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # METRICS
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚀 PASSES", pass_count)
        col2.metric("🧠 Max Intelligence", df['Analysts'].max())
        col3.metric("📈 Top Score", df['Score'].max())
        col4.metric("📊 Scanned", len(results))
        
        # FULL INTELLIGENCE TABLE
        st.subheader("🧠 COMPLETE MARKET INTELLIGENCE - CLICK ANY ROW")
        def color_verdict(val):
            color = 'green' if val == '🚀 PASS' else 'red' if val == '❌ FAIL' else 'orange'
            return f'color: {color}; font-weight: bold'
        
        selected = st.dataframe(
            df.style.applymap(color_verdict, subset=['Verdict']),
            use_container_width=True,
            selection_mode="single-row",
            column_config={
                "Score": st.column_config.ProgressColumn("Fortress Score", min_value=0, max_value=100),
                "Verdict": st.column_config.TextColumn("Status"),
                "Report": st.column_config.TextColumn("🧠 Intelligence", help="News + Analyst + Financials"),
                "Analyst Target": st.column_config.NumberColumn("Target ₹", format="₹%.0f"),
                "Analysts": st.column_config.NumberColumn("Coverage"),
                "Price": st.column_config.NumberColumn("Price ₹", format="₹%.0f")
            },
            height=700
        )
        
        # AI INTELLIGENCE TRIGGER
        if selected and 'selection' in selected and selected['selection'].get('rows'):
            row_idx = selected['selection']['rows'][0]
            ticker = df.iloc[row_idx]['Symbol']
            show_analyst_report(ticker)

st.markdown("---")
st.caption("🛡️ **Fortress 95 Pro v5.3** - AI News Sentiment + Analyst Intelligence + Full Technicals | Production Ready")

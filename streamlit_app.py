# fortress_app.py - v5.9 FULL GUARDRAILS + NEWS + EARNINGS
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
st.title("🛡️ Fortress 95 Pro v5.9 - NEWS GUARDRAIL + EARNINGS BLOCKER")

# --- COMPLETE AI INTELLIGENCE REPORT ---
@st.dialog("📋 AI Intelligence + News Guardrail", width="large")
def show_analyst_report(ticker_symbol):
    st.markdown(f"### 🧠 **Strategic Intelligence: {ticker_symbol}**")
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        news = ticker_obj.news
        
        # ROW 1: FULL ANALYST CONSENSUS
        st.markdown("#### 🏦 **Analyst Consensus**")
        c1, c2, c3, c4 = st.columns(4)
        target = info.get('targetMeanPrice', 0)
        current = info.get('currentPrice', 1)
        upside = ((target/current)-1)*100 if target > 0 else 0
        
        c1.metric("Rating", info.get('recommendationKey', 'N/A').upper())
        c2.metric("Target ₹", f"{target:,.0f}")
        c3.metric("Analysts", info.get('numberOfAnalystOpinions', 0))
        c4.metric("Upside", f"{upside:.1f}%")

        # ROW 2: NEWS SENTIMENT + GUARDRAIL
        st.markdown("#### 📰 **News Guardrail Analysis**")
        danger_keys = ['fraud', 'investigation', 'default', 'scam', 'bankruptcy', 'legal']
        news_sentiment = "✅ Neutral"
        if news:
            titles = [n['title'].lower() for n in news[:5]]
            if any(any(k in t for k in danger_keys) for t in titles):
                news_sentiment = "🚨 BLACK SWAN DETECTED"
            
            for n in news[:5]:
                title = n['title']
                t_lower = title.lower()
                tag = "🔹"
                if any(k in t_lower for k in danger_keys): tag = "🚨 RISK"
                st.markdown(f"{tag} **{title}**")
                st.caption(f"*{n['publisher']}*")

        st.metric("News Risk", news_sentiment)

        # ROW 3: EARNINGS BLOCKER
        st.markdown("#### 📅 **Earnings Calendar**")
        event_risk = "✅ Safe"
        try:
            cal = ticker_obj.calendar
            if cal is not None and not cal.empty:
                days_to = (cal.iloc[0, 0].date() - datetime.now().date()).days
                if 0 <= days_to <= 7:
                    event_risk = f"🚨 EARNINGS {cal.iloc[0, 0].strftime('%d-%b')}"
        except: pass
        st.metric("Event Risk", event_risk)

        col1, col2 = st.columns(2)
        if col1.button("🔄 Refresh", use_container_width=True): st.rerun()
        if col2.button("❌ Close", use_container_width=True): st.rerun()
            
    except Exception as e:
        st.error(f"Report error: {str(e)}")
        if st.button("Close"): st.rerun()

# --- ULTIMATE FORTRESS ENGINE (ALL GUARDRAILS) ---
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        # Fix MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna()
        
        if len(data) < 200: 
            return {
                "Symbol": ticker, "Sector": SECTOR_MAP.get(ticker, "N/A"),
                "Verdict": "⚠️ DATA", "Report": "🧠", "Price": 0, "RSI": 0, 
                "Age": "0d", "Analyst Target": "N/A", "Analysts": 0, 
                "News Risk": "⚠️", "Earnings": "⚠️", "Upside %": "N/A", "Score": 0
            }
        
        # CORE TECHNICALS
        price = float(data['Close'].iloc[-1])
        ema200 = float(ta.ema(data['Close'], length=200).iloc[-1])
        rsi = float(ta.rsi(data['Close'], length=14).iloc[-1])
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        trend = float(st_df['SUPERT_10_3.0'].iloc[-1]) if 'SUPERT_10_3.0' in st_df.columns else 1
        
        # TECHNICAL PASS
        technical_pass = (price > ema200 and 40 <= rsi <= 70 and trend <= 1)
        
        # 1. NEWS SENTIMENT GUARDRAIL
        news_sentiment = "✅ Neutral"
        danger_keys = ['fraud', 'investigation', 'default', 'scam', 'bankruptcy', 'legal']
        try:
            news = ticker_obj.news
            if news:
                titles = [n['title'].lower() for n in news[:5]]
                if any(any(k in t for k in danger_keys) for t in titles):
                    news_sentiment = "🚨 BLACK SWAN"
        except: pass

        # 2. EARNINGS BLOCKER
        event_risk = "✅ Safe"
        try:
            cal = ticker_obj.calendar
            if cal is not None and not cal.empty:
                days_to = (cal.iloc[0, 0].date() - datetime.now().date()).days
                if 0 <= days_to <= 7:
                    event_risk = f"🚨 EARNINGS ({cal.iloc[0, 0].strftime('%d-%b')})"
        except: pass

        # 3. FINAL VERDICT (ALL 3 CONDITIONS)
        is_pass = (technical_pass and news_sentiment == "✅ Neutral" and "EARNINGS" not in event_risk)
        
        # ANALYST DATA
        info = ticker_obj.info
        target = info.get('targetMeanPrice', 0)
        analysts = info.get('numberOfAnalystOpinions', 0)
        upside = ((target - price) / price * 100) if target > 0 else 0
        
        # TREND AGE
        age = 0
        for i in range(1, 15):
            if i < len(data) and data['Close'].iloc[-i] > ema200 and st_df['SUPERT_10_3.0'].iloc[-i] <= 1:
                age += 1
            else: break
        
        # SCORING
        score = 95 if (is_pass and 48 <= rsi <= 58) else (80 if is_pass else 0)

        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Verdict": "🚀 PASS" if is_pass else "❌ FAIL",
            "Report": "🧠 AI",
            "Price": round(price, 2),
            "RSI": round(rsi, 1),
            "Age": f"{age}d",
            "Analyst Target": round(target, 0) if target > 0 else "N/A",
            "Analysts": int(analysts),
            "News Risk": news_sentiment,
            "Earnings": event_risk,
            "Upside %": f"{upside:.1f}%" if upside != 0 else "N/A",
            "Score": score
        }
    except:
        return {
            "Symbol": ticker, "Sector": SECTOR_MAP.get(ticker, "N/A"),
            "Verdict": "⚠️ ERROR", "Report": "🧠 AI", "Price": 0, "RSI": 0, 
            "Age": "0d", "Analyst Target": "N/A", "Analysts": 0,
            "News Risk": "⚠️", "Earnings": "⚠️", "Upside %": "N/A", "Score": 0
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
            status = "🟢 BULL" if price > ema else "🔴 BEAR"
            if price > ema: bullish_count += 1
            cols[i].metric(name, f"₹{price:,.0f}", status)
    except:
        cols[i].error("Error")

market_status = "✅ BULL" if bullish_count >= 2 else "⚠️ MIXED" if bullish_count == 1 else "🛑 BEAR"
st.success(f"**{market_status} MARKET** - {bullish_count}/3 bullish")

# --- CONTROLS ---
st.sidebar.title("🔍 Fortress Controls")
selected_index = st.sidebar.selectbox("Universe", list(TICKER_GROUPS.keys()))
TICKERS = TICKER_GROUPS[selected_index]
st.sidebar.info(f"📊 **{len(TICKERS)} stocks** | **News + Earnings Guardrails ACTIVE**")

if st.sidebar.button("🧹 Clear Cache"): st.rerun()

# --- MAIN SCAN ---
if st.button("🚀 FULL FORTRESS SCAN w/ GUARDRAILS", type="primary", use_container_width=True):
    results = []
    total = len(TICKERS)
    progress = st.progress(0)
    status = st.empty()
    pass_count = 0
    
    for i, ticker in enumerate(TICKERS):
        status.text(f"🔍 [{i+1}/{total}] {ticker} (News+Earnings check)")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            data = yf.download(ticker, period="1y", progress=False, threads=False)
            
            if not data.empty:
                result = check_institutional_fortress(ticker, data, ticker_obj)
                results.append(result)
                
                if result['Verdict'] == "🚀 PASS":
                    pass_count += 1
                    st.toast(f"✅ {ticker} PASSED all guardrails!", icon="🚀")
            
            time.sleep(0.7)
        except Exception as e:
            if "429" in str(e):
                status.error("🚨 Rate limit - waiting...")
                time.sleep(10)
            continue
        
        progress.progress((i+1)/total)
    
    status.success(f"✅ **COMPLETE!** {pass_count}/{total} PASSES (Guardrails Active)")

    if results:
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # SUMMARY METRICS
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🚀 PASSES", pass_count)
        col2.metric("📈 Top Score", df['Score'].max())
        col3.metric("🏦 Max Analysts", int(df['Analysts'].max()))
        col4.metric("🚨 Black Swans", len(df[df['News Risk'] == '🚨 BLACK SWAN']))
        col5.metric("📊 Scanned", len(results))
        
        # TABLE w/ GUARDRAIL COLUMNS
        st.subheader("📊 **FULL RESULTS w/ GUARDRAILS**")
        st.info("🚀 PASS = Technical + No Black Swan + No Earnings | Click buttons 👇")
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%d"),
                "Verdict": st.column_config.TextColumn("Status"),
                "News Risk": st.column_config.TextColumn("News"),
                "Earnings": st.column_config.TextColumn("Events"),
                "Analyst Target": st.column_config.NumberColumn("Target ₹", format="₹%.0f"),
                "Analysts": st.column_config.NumberColumn("Coverage"),
                "Price": st.column_config.NumberColumn("Price ₹", format="₹%.0f")
            },
            height=600
        )
        
        # QUICK INTELLIGENCE BUTTONS
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        if col1.button("🟢 #1 SAFE PASS", use_container_width=True):
            safe_pass = df[(df['Verdict'] == '🚀 PASS') & (df['News Risk'] == '✅ Neutral')]
            if not safe_pass.empty: show_analyst_report(safe_pass.iloc[0]['Symbol'])
        
        if col2.button("🚨 SHOW RISKS", use_container_width=True):
            risks = df[df['News Risk'] == '🚨 BLACK SWAN']
            if not risks.empty: show_analyst_report(risks.iloc[0]['Symbol'])
        
        if col3.button("⭐ #1 OVERALL", use_container_width=True):
            show_analyst_report(df.iloc[0]['Symbol'])

st.markdown("---")
st.caption("🛡️ **Fortress 95 Pro v5.9** - ✅ NEWS GUARDRAIL + EARNINGS BLOCKER + FULL LOGIC")

# fortress_app.py - v5.13 FULL FEATURES + ARROW SAFE
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
st.title("🛡️ Fortress 95 Pro v5.13 - FULL FEATURES RESTORED")

# --- COMPLETE AI INTELLIGENCE REPORT (ALL ORIGINAL) ---
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
        st.info(f"**AI Insight:** {coverage} analysts project **{upside:.1f}% upside**. Coverage: {'🟢 HIGH' if coverage > 15 else '🟡 MODERATE' if coverage > 5 else '🔴 LOW'}.")

        # ROW 2: AI NEWS SENTIMENT
        st.markdown("#### 📰 **Latest News Sentiment AI** (Black Swan Detection)")
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
                st.caption(f"📅 Recent | [Read Full Story]({n['link']}) | {n['publisher']}")
                st.markdown("---")
        else:
            st.warning("📰 No recent news found.")

        # ROW 3: FINANCIAL HEALTH
        st.markdown("#### 📊 **Financial Health Indicators**")
        points = [
            f"📍 **P/E Ratio:** {info.get('trailingPE', 'N/A')} | {'⚠️ HIGH' if info.get('trailingPE', 0) > 25 else '✅ FAIR'}",
            f"📍 **Debt/Equity:** {info.get('debtToEquity', 'N/A')}",
            f"📍 **Market Cap:** ₹{info.get('marketCap', 0):,}",
            f"📍 **Beta:** {info.get('beta', 'N/A')} | {'📈 VOLATILE' if info.get('beta', 0) > 1.2 else '📊 STABLE'}"
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
        st.error(f"⚠️ Report unavailable: {str(e)}")
        if st.button("❌ Close"):
            st.rerun()

# --- BULLETPROOF FORTRESS ENGINE (ARROW SAFE) ---
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        # Fix data columns first
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if len(data) < 200:
            return {
                "Symbol": ticker,
                "Sector": SECTOR_MAP.get(ticker, "General"),
                "Verdict": "⚠️ DATA",
                "Report": "🧠 AI",
                "Price": 0.0,
                "RSI": 0.0,
                "Age": "0d",
                "Analyst_Target": 0.0,
                "Analysts": 0,
                "Upside_Percent": 0.0,
                "Score": 0,
                "News_Risk": "⚠️ DATA",
                "Earnings": "⚠️ DATA"
            }
        
        price = float(data['Close'].iloc[-1])
        ema200 = float(ta.ema(data['Close'], length=200).iloc[-1])
        rsi = float(ta.rsi(data['Close'], length=14).iloc[-1])
        
        # SuperTrend fix
        try:
            st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
            trend = float(st_df['SUPERT_10_3.0'].iloc[-1])
        except:
            trend = 1
        
        # 1. FIXED EARNINGS LOGIC
        event_risk = "✅ No Data"
        try:
            cal = ticker_obj.calendar
            if cal is not None and isinstance(cal, pd.DataFrame) and not cal.empty:
                next_date = cal.iloc[0, 0]
                days_to = (next_date.date() - datetime.now().date()).days
                if 0 <= days_to <= 7:
                    event_risk = f"🚨 EARNINGS ({next_date.strftime('%d-%b')})"
                else:
                    event_risk = "✅ Safe"
            else:
                event_risk = "✅ No Data"
        except:
            event_risk = "✅ No Data"

        # 2. NEWS SENTIMENT GUARDRAIL
        news_sentiment = "✅ Neutral"
        danger_keys = ['fraud', 'investigation', 'default', 'scam', 'bankruptcy', 'legal']
        try:
            news = ticker_obj.news
            if news:
                titles = [n['title'].lower() for n in news[:5]]
                if any(any(k in t for k in danger_keys) for t in titles):
                    news_sentiment = "🚨 BLACK SWAN"
        except: pass

        # 3. TECHNICAL PASS (LENIENT RSI <= 75)
        tech_pass = (price > ema200 and 40 <= rsi <= 75 and trend <= 1)
        
        # 4. FINAL VERDICT
        is_pass = (tech_pass and news_sentiment != "🚨 BLACK SWAN" and "🚨" not in event_risk)
        
        info = ticker_obj.info
        target = info.get('targetMeanPrice', 0) or 0.0
        analysts = info.get('numberOfAnalystOpinions', 0) or 0
        upside = ((target - price) / price * 100) if target > 0 and price > 0 else 0.0

        # TREND AGE (ORIGINAL LOGIC)
        age = 0
        for i in range(1, 15):
            if i < len(data) and data['Close'].iloc[-i] > ema200 and st_df['SUPERT_10_3.0'].iloc[-i] <= 1:
                age += 1
            else: 
                break

        # SCORING (GOLDEN RSI = 95pts)
        score = 95 if (is_pass and 48 <= rsi <= 58) else (80 if is_pass else 0)

        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Verdict": "🚀 PASS" if is_pass else "❌ FAIL",
            "Report": "🧠 AI",
            "Price": round(price, 2),
            "RSI": round(rsi, 1),
            "Age": f"{age}d",
            "Analyst_Target": round(target, 0),
            "Analysts": int(analysts),
            "Upside_Percent": round(upside, 1),
            "Score": score,
            "News_Risk": news_sentiment,
            "Earnings": event_risk
        }
    except Exception:
        return {
            "Symbol": ticker, "Sector": "ERROR",
            "Verdict": "⚠️ ERROR", "Report": "🧠 AI", 
            "Price": 0.0, "RSI": 0.0, "Age": "0d",
            "Analyst_Target": 0.0, "Analysts": 0, 
            "Upside_Percent": 0.0, "Score": 0,
            "News_Risk": "⚠️", "Earnings": "⚠️"
        }

# --- MARKET PULSE (ALL 3 INDICES) ---
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
st.sidebar.info(f"📊 **{len(TICKERS)} stocks** | **ALL FEATURES ACTIVE**")

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
                    st.toast(f"✅ FORTRESS PASS: {ticker}", icon="🚀")
            
            time.sleep(0.7)
        except:
            continue
            
        progress.progress((i+1)/total)
    
    status.success("✅ **SCAN COMPLETE!** Full intelligence below 👇")

    if results:
        # ARROW-SAFE DataFrame processing
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # Force numeric safety
        numeric_cols = ['Price', 'RSI', 'Analyst_Target', 'Analysts', 'Upside_Percent', 'Score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # SUMMARY METRICS
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚀 PASSES", pass_count)
        col2.metric("🧠 Max Coverage", df['Analysts'].max())
        col3.metric("📈 Top Score", df['Score'].max())
        col4.metric("📊 Scanned", len(results))
        
        # ✅ ARROW-SAFE TABLE (SIMPLE + SAFE)
        st.subheader("🧠 FULL INTELLIGENCE DASHBOARD")
        st.info("**🟢 PASS** = Trade Now | **🔴 FAIL** = Watchlist | **Use Buttons Below** 👇")
        
        st.dataframe(
            df,
            use_container_width=True,
            height=600
        )
        
        # ✅ BULLETPROOF INTELLIGENCE BUTTONS
        st.markdown("---")
        st.subheader("🚀 **QUICK AI INTELLIGENCE BUTTONS**")
        
        col1, col2, col3 = st.columns(3)
        
        # BUTTON 1: TOP PASS
        if col1.button("🟢 TOP FORTRESS PASS", use_container_width=True):
            top_pass = df[df['Verdict'] == '🚀 PASS']
            if not top_pass.empty:
                ticker = top_pass.iloc[0]['Symbol']
                show_analyst_report(ticker)
            else:
                st.warning("No PASS stocks found!")
        
        # BUTTON 2: HIGHEST ANALYST COVERAGE
        if col2.button("🏦 TOP ANALYST COVERAGE", use_container_width=True):
            top_analyst = df.loc[df['Analysts'].idxmax()]['Symbol']
            show_analyst_report(top_analyst)
        
        # BUTTON 3: HIGHEST SCORE
        if col3.button("⭐ HIGHEST SCORE", use_container_width=True):
            top_score = df.iloc[0]['Symbol']
            show_analyst_report(top_score)

st.markdown("---")
st.caption("🛡️ **Fortress 95 Pro v5.13** - ✅ ALL CODE RESTORED | FULL AI REPORTS | ARROW SAFE")

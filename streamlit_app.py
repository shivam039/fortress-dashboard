# fortress_app.py - v5.11 ARROW SERIALIZATION FIXED
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
st.title("🛡️ Fortress 95 Pro v5.11 - ✅ ARROW FIXED")

# --- AI INTELLIGENCE REPORT ---
@st.dialog("📋 AI Intelligence + Guardrails", width="large")
def show_analyst_report(ticker_symbol):
    st.markdown(f"### 🧠 **Strategic Intelligence: {ticker_symbol}**")
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        news = ticker_obj.news
        
        # Analyst Consensus
        st.markdown("#### 🏦 **Analyst Consensus**")
        c1, c2, c3, c4 = st.columns(4)
        target = info.get('targetMeanPrice', 0) or 0
        current = info.get('currentPrice', 1) or 1
        upside = ((target/current)-1)*100 if target > 0 else 0
        
        c1.metric("Rating", info.get('recommendationKey', 'N/A').upper())
        c2.metric("Target ₹", f"{target:,.0f}")
        c3.metric("Analysts", info.get('numberOfAnalystOpinions', 0))
        c4.metric("Upside", f"{upside:.1f}%")

        # News Guardrail
        st.markdown("#### 📰 **News Guardrail**")
        danger_keys = ['fraud', 'investigation', 'default', 'scam', 'bankruptcy', 'legal']
        news_sentiment = "✅ Neutral"
        if news:
            titles = [n['title'].lower() for n in news[:5]]
            if any(any(k in t for k in danger_keys) for t in titles):
                news_sentiment = "🚨 BLACK SWAN"
            for n in news[:5]:
                title = n['title']
                t_lower = title.lower()
                tag = "🔹"
                if any(k in t_lower for k in danger_keys): tag = "🚨 RISK"
                st.markdown(f"{tag} **{title}**")
                st.caption(f"*{n['publisher']}*")
        st.metric("News Risk", news_sentiment)

        # Earnings Calendar
        st.markdown("#### 📅 **Earnings Calendar**")
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
        st.metric("Event Risk", event_risk)

        col1, col2 = st.columns(2)
        if col1.button("🔄 Refresh", use_container_width=True): st.rerun()
        if col2.button("❌ Close", use_container_width=True): st.rerun()
            
    except Exception as e:
        st.error(f"Report error: {str(e)}")
        if st.button("Close"): st.rerun()

# --- BULLETPROOF FORTRESS ENGINE (ARROW SAFE) ---
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        # Fix MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna()
        
        if len(data) < 200: 
            return {
                "Symbol": ticker,
                "Sector": SECTOR_MAP.get(ticker, "N/A"),
                "Verdict": "⚠️ DATA",
                "Report": "🧠",
                "Price": 0.0,
                "RSI": 0.0,
                "Age": "0d",
                "Analyst_Target": 0.0,  # ✅ ARROW SAFE: All numeric
                "Analysts": 0,
                "News_Risk": "⚠️",
                "Earnings": "⚠️",
                "Upside": 0.0,  # ✅ ARROW SAFE: All numeric
                "Score": 0
            }
        
        # Technical indicators
        price = float(data['Close'].iloc[-1])
        ema200 = float(ta.ema(data['Close'], length=200).iloc[-1])
        rsi = float(ta.rsi(data['Close'], length=14).iloc[-1])
        st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
        trend = float(st_df['SUPERT_10_3.0'].iloc[-1]) if 'SUPERT_10_3.0' in st_df.columns else 1
        
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

        # Analyst Data
        info = ticker_obj.info
        target = info.get('targetMeanPrice', 0) or 0.0
        analysts = info.get('numberOfAnalystOpinions', 0) or 0
        upside = ((target - price) / price * 100) if target > 0 and price > 0 else 0.0
        
        # Trend Age
        age = 0
        for i in range(1, 15):
            if i < len(data) and data['Close'].iloc[-i] > ema200 and st_df['SUPERT_10_3.0'].iloc[-i] <= 1:
                age += 1
            else: break
        
        # Scoring
        score = 95 if (is_pass and 48 <= rsi <= 58) else (80 if is_pass else 0)

        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Verdict": "🚀 PASS" if is_pass else "❌ FAIL",
            "Report": "🧠 AI",
            "Price": round(price, 2),
            "RSI": round(rsi, 1),
            "Age": f"{age}d",
            "Analyst_Target": round(target, 0),  # ✅ NUMERIC ONLY
            "Analysts": int(analysts),
            "News_Risk": news_sentiment,
            "Earnings": event_risk,
            "Upside": round(upside, 1),  # ✅ NUMERIC ONLY
            "Score": score
        }
    except:
        return {
            "Symbol": ticker,
            "Sector": "ERROR",
            "Verdict": "⚠️ ERROR",
            "Report": "🧠 AI",
            "Price": 0.0,
            "RSI": 0.0,
            "Age": "0d",
            "Analyst_Target": 0.0,  # ✅ NUMERIC
            "Analysts": 0,
            "News_Risk": "⚠️",
            "Earnings": "⚠️",
            "Upside": 0.0,  # ✅ NUMERIC
            "Score": 0
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
st.sidebar.info(f"📊 **{len(TICKERS)} stocks** | **ARROW COMPATIBLE**")

if st.sidebar.button("🧹 Clear Cache"): st.rerun()

# --- MAIN SCAN ---
if st.button("🚀 FULL FORTRESS SCAN", type="primary", use_container_width=True):
    results = []
    total = len(TICKERS)
    progress = st.progress(0)
    status = st.empty()
    pass_count = 0
    
    for i, ticker in enumerate(TICKERS):
        status.text(f"🔍 [{i+1}/{total}] {ticker}")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            data = yf.download(ticker, period="1y", progress=False, threads=False)
            
            if not data.empty:
                result = check_institutional_fortress(ticker, data, ticker_obj)
                results.append(result)
                
                if result['Verdict'] == "🚀 PASS":
                    pass_count += 1
                    st.toast(f"✅ {ticker} PASSED!", icon="🚀")
            
            time.sleep(0.7)
        except Exception as e:
            if "429" in str(e):
                status.error("🚨 Rate limit - waiting...")
                time.sleep(10)
            continue
        
        progress.progress((i+1)/total)
    
    status.success(f"✅ **COMPLETE!** {pass_count}/{total} PASSES")

    if results:
        # ✅ ARROW SAFE: Convert to clean DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('Score', ascending=False).reset_index(drop=True)
        
        # Force numeric columns for Arrow compatibility
        numeric_cols = ['Price', 'RSI', 'Analyst_Target', 'Analysts', 'Upside', 'Score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # SUMMARY METRICS
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🚀 PASSES", pass_count)
        col2.metric("📈 Top Score", int(df['Score'].max()))
        col3.metric("🏦 Max Analysts", int(df['Analysts'].max()))
        col4.metric("🚨 Black Swans", len(df[df['News_Risk'] == '🚨 BLACK SWAN']))
        col5.metric("📊 Scanned", len(results))
        
        # ✅ ARROW PERFECT TABLE
        st.subheader("📊 **COMPLETE RESULTS**")
        st.info("🚀 PASS = Tech(RSI≤75) + No Black Swan + No Earnings Risk")
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%d"),
                "Verdict": st.column_config.TextColumn("Status"),
                "News_Risk": st.column_config.TextColumn("News"),
                "Earnings": st.column_config.TextColumn("Events"),
                "Analyst_Target": st.column_config.NumberColumn("Target ₹", format="₹%.0f"),
                "Analysts": st.column_config.NumberColumn("Coverage"),
                "Price": st.column_config.NumberColumn("Price ₹", format="₹%.0f"),
                "RSI": st.column_config.NumberColumn("RSI", help="40-75 allowed")
            },
            height=600
        )
        
        # INTELLIGENCE BUTTONS
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        if col1.button("🟢 #1 SAFE PASS", use_container_width=True):
            safe_pass = df[df['Verdict'] == '🚀 PASS']
            if not safe_pass.empty: show_analyst_report(safe_pass.iloc[0]['Symbol'])
        
        if col2.button("🚨 SHOW RISKS", use_container_width=True):
            risks = df[df['News_Risk'] == '🚨 BLACK SWAN']
            if not risks.empty: show_analyst_report(risks.iloc[0]['Symbol'])
        
        if col3.button("⭐ #1 SCORE", use_container_width=True):
            show_analyst_report(df.iloc[0]['Symbol'])

st.markdown("---")
st.caption("🛡️ **Fortress 95 Pro v5.11** - ✅ ARROW FIXED | No 'N/A' in numeric columns")

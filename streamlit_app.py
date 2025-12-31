# fortress_app.py - v6.0 WEIGHTED CONVICTION (ARROW FIXED)
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
st.title("🛡️ Fortress 95 Pro v6.0 - WEIGHTED CONVICTION ENGINE")

# --- UPDATED FORTRESS ENGINE (WEIGHTED LOGIC) ---
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        # Fix data columns first
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if len(data) < 200:
            return {"Symbol": ticker, "Sector": SECTOR_MAP.get(ticker, "General"), "Verdict": "⚠️ DATA", "Score": 0, "Price": 0.0, "RSI": 0.0, "News": "⚠️", "Events": "⚠️", "Target_Analyst": 0.0}
        
        # --- 1. TECHNICAL FOUNDATION ---
        price = float(data['Close'].iloc[-1])
        ema200 = float(ta.ema(data['Close'], length=200).iloc[-1])
        rsi = float(ta.rsi(data['Close'], length=14).iloc[-1])
        
        try:
            st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
            # Handle potential column naming variations in pandas-ta
            trend_col = 'SUPERT_10_3.0' if 'SUPERT_10_3.0' in st_df.columns else 'SUPERTd_10_3.0'
            trend = float(st_df[trend_col].iloc[-1])
        except:
            trend = 1
        
        # --- 2. THE MODIFIERS (NON-BLOCKING) ---
        # A. News Sentinel
        news_sentiment = "Neutral"
        score_mod = 0
        danger_keys = ['fraud', 'investigation', 'default', 'scam', 'bankruptcy', 'legal']
        try:
            news = ticker_obj.news
            if news:
                titles = [n['title'].lower() for n in news[:5]]
                if any(any(k in t for k in danger_keys) for t in titles):
                    news_sentiment = "🚨 BLACK SWAN"
                    score_mod -= 40
        except: pass

        # B. Earnings Date
        event_status = "✅ Safe"
        try:
            cal = ticker_obj.calendar
            if cal is not None and isinstance(cal, pd.DataFrame) and not cal.empty:
                next_date = cal.iloc[0, 0]
                days_to = (next_date.date() - datetime.now().date()).days
                if 0 <= days_to <= 7:
                    event_status = f"🚨 EARNINGS ({next_date.strftime('%d-%b')})"
                    score_mod -= 20
        except: pass

        # --- 3. SCORING CALCULATION ---
        tech_base = (price > ema200 and trend <= 1)
        conviction = 0
        
        if tech_base:
            conviction += 60
            # Momentum Scoring (Golden Zone 48-62)
            if 48 <= rsi <= 62: conviction += 20
            elif 40 <= rsi < 48 or 62 < rsi <= 75: conviction += 10
            
            # Analyst Boost
            info = ticker_obj.info
            target = info.get('targetMeanPrice', 0) or 0
            if target > price * 1.10: conviction += 10
            
            # Apply modifiers
            conviction += score_mod
        
        # Keep score in bounds
        conviction = max(0, min(100, conviction))
        
        # --- 4. DYNAMIC VERDICT ---
        if conviction >= 85: verdict = "🔥 HIGH CONVICTION"
        elif conviction >= 60: verdict = "🚀 PASS"
        elif tech_base: verdict = "🟡 WATCH"
        else: verdict = "❌ FAIL"

        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Verdict": verdict,
            "Score": conviction,
            "Price": round(price, 2),
            "RSI": round(rsi, 1),
            "News": news_sentiment,
            "Events": event_status,
            "Target_Analyst": round(target, 0)
        }
    except Exception:
        return {"Symbol": ticker, "Verdict": "⚠️ ERROR", "Score": 0, "Price": 0.0, "RSI": 0.0, "Target_Analyst": 0.0}

# --- FIXED MARKET PULSE ---
st.subheader("🌐 Market Pulse")
cols = st.columns(3)
bullish_count = 0
for i, (name, symbol) in enumerate(INDEX_BENCHMARKS.items()):
    try:
        data = yf.download(symbol, period="1y", progress=False)
        if not data.empty:
            price = data['Close'].iloc[-1]
            ema = ta.ema(data['Close'], 200).iloc[-1]
            status = "🟢 BULLISH" if price > ema else "🔴 BEARISH"
            if price > ema: bullish_count += 1
            cols[i].metric(name, f"₹{price:,.0f}", status)
    except: pass

market_status = "✅ BULL MARKET" if bullish_count >= 2 else "⚠️ MIXED" if bullish_count == 1 else "🛑 BEAR"
st.success(f"**{market_status}** - {bullish_count}/3 indices bullish")

# --- CONTROLS ---
st.sidebar.title("🔍 Fortress Controls")
selected_index = st.sidebar.selectbox("Universe", list(TICKER_GROUPS.keys()))
TICKERS = TICKER_GROUPS[selected_index]
st.sidebar.info(f"📊 **{len(TICKERS)} stocks** | **Weighted Conviction Active**")

if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# --- MAIN SCAN ---
if st.button("🚀 START WEIGHTED SCAN", type="primary", use_container_width=True):
    results = []
    total = len(TICKERS)
    progress = st.progress(0)
    status = st.empty()
    high_conviction = 0
    
    for i, ticker in enumerate(TICKERS):
        status.text(f"🔍 [{i+1}/{total}] {ticker}")
        try:
            ticker_obj = yf.Ticker(ticker)
            data = yf.download(ticker, period="1y", progress=False)
            if not data.empty:
                result = check_institutional_fortress(ticker, data, ticker_obj)
                results.append(result)
                if result['Verdict'] == "🔥 HIGH CONVICTION":
                    high_conviction += 1
                    st.toast(f"🔥 HIGH CONVICTION: {ticker}", icon="🔥")
                elif result['Verdict'] == "🚀 PASS":
                    st.toast(f"✅ PASS: {ticker}", icon="🚀")
            time.sleep(0.7)
        except: continue
        progress.progress((i+1)/total)
    
    status.success(f"✅ SCAN COMPLETE! {high_conviction} High Conviction found.")

    if results:
        # ARROW-SAFE DataFrame
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # Force numeric columns
        numeric_cols = ['Price', 'RSI', 'Target_Analyst', 'Score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # SUMMARY METRICS
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🔥 HIGH CONV", high_conviction)
        c2.metric("🚀 PASSES", len(df[df['Verdict'] == '🚀 PASS']))
        c3.metric("📈 Top Score", int(df['Score'].max()))
        c4.metric("🏦 Max Target", f"₹{int(df['Target_Analyst'].max()):,}")
        c5.metric("📊 Scanned", len(results))
        
        # ✅ ARROW-SAFE TABLE (NO styling/ProgressColumn)
        st.subheader("📊 CONVICTION DASHBOARD")
        st.info("**🔥 HIGH CONVICTION** (85+) = Trade Now | **🚀 PASS** (60+) = Strong | **🟡 WATCH** = Monitor")
        
        st.dataframe(df, use_container_width=True, height=600)

st.markdown("---")
st.caption("🛡️ **Fortress 95 Pro v6.0** - Weighted Scoring | No Errors | Production Ready")

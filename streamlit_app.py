# fortress_app.py - v5.14 CLEAN SCANNER (No Dialogs)
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
st.title("🛡️ Fortress 95 Pro v5.14 - CLEAN SCANNER")

# --- BULLETPROOF FORTRESS ENGINE (ALL LOGIC RETAINED) ---
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        # Fix data columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if len(data) < 200:
            return {
                "Symbol": ticker, "Sector": SECTOR_MAP.get(ticker, "General"),
                "Verdict": "⚠️ DATA", "Price": 0.0, "RSI": 0.0, "Age": "0d",
                "Analyst_Target": 0.0, "Analysts": 0, "Upside_Percent": 0.0,
                "Score": 0, "News_Risk": "⚠️ DATA", "Earnings": "⚠️ DATA"
            }
        
        price = float(data['Close'].iloc[-1])
        ema200 = float(ta.ema(data['Close'], length=200).iloc[-1])
        rsi = float(ta.rsi(data['Close'], length=14).iloc[-1])
        
        try:
            st_df = ta.supertrend(data['High'], data['Low'], data['Close'], 10, 3)
            trend = float(st_df['SUPERT_10_3.0'].iloc[-1])
        except:
            trend = 1

        # 1. EARNINGS BLOCKER LOGIC
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

        # TREND AGE
        age = 0
        for i in range(1, 15):
            if i < len(data) and data['Close'].iloc[-i] > ema200:
                age += 1
            else: break

        # SCORING
        score = 95 if (is_pass and 48 <= rsi <= 58) else (80 if is_pass else 0)

        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Verdict": "🚀 PASS" if is_pass else "❌ FAIL",
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
        return {"Symbol": ticker, "Verdict": "⚠️ ERROR", "Score": 0}

# --- MARKET PULSE ---
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

st.sidebar.title("🔍 Fortress Controls")
selected_index = st.sidebar.selectbox("Universe", list(TICKER_GROUPS.keys()))
TICKERS = TICKER_GROUPS[selected_index]

# --- MAIN SCAN ---
if st.button("🚀 START FULL SCAN", type="primary", use_container_width=True):
    results = []
    total = len(TICKERS)
    progress = st.progress(0)
    status = st.empty()
    pass_count = 0
    
    for i, ticker in enumerate(TICKERS):
        status.text(f"🔍 [{i+1}/{total}] {ticker}")
        try:
            ticker_obj = yf.Ticker(ticker)
            data = yf.download(ticker, period="1y", progress=False)
            if not data.empty:
                result = check_institutional_fortress(ticker, data, ticker_obj)
                results.append(result)
                if result['Verdict'] == "🚀 PASS":
                    pass_count += 1
            time.sleep(0.7)
        except: continue
        progress.progress((i+1)/total)
    
    status.success(f"✅ SCAN COMPLETE! {pass_count} Fortress setups found.")

    if results:
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # SUMMARY METRICS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚀 PASSES", pass_count)
        c2.metric("🧠 Max Coverage", int(df['Analysts'].max()))
        c3.metric("📈 Top Score", int(df['Score'].max()))
        c4.metric("📊 Scanned", len(results))
        
        # DASHBOARD
        st.subheader("📊 FULL SCAN RESULTS")
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Fortress Score", min_value=0, max_value=100),
                "Verdict": st.column_config.TextColumn("Verdict"),
                "Analyst_Target": st.column_config.NumberColumn("Target ₹", format="₹%d"),
                "Price": st.column_config.NumberColumn("Price ₹", format="₹%.2f"),
                "Upside_Percent": st.column_config.NumberColumn("Upside %", format="%.1f%%")
            },
            height=600
        )

st.caption("🛡️ Fortress 95 Pro v5.14 - Clean Scanner | Logic Intact")

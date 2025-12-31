# fortress_app.py - v7.1 INSTITUTIONAL CONVICTION SCREENER

import subprocess, sys, time
from datetime import datetime

import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np

from fortress_config import (
    TICKER_GROUPS,
    SECTOR_MAP,
    INDEX_BENCHMARKS,
    NIFTY_SYMBOL
)

# ---------------- DEPENDENCIES ----------------
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

# ---------------- UI ----------------
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v7.1 — Institutional Conviction Screener")

# ======================================================
def check_institutional_fortress(ticker, data, ticker_obj):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if len(data) < 210:
            return {
                "Symbol": ticker,
                "Sector": SECTOR_MAP.get(ticker, "General"),
                "Verdict": "⚠️ DATA",
                "Score": 0,
                "Price": float(data["Close"].iloc[-1]),
                "RSI": 0.0,
                "News": "⚠️",
                "Events": "⚠️",
                "Target_Analyst": 0
            }

        close, high, low = data["Close"], data["High"], data["Low"]

        ema200 = ta.ema(close, 200)
        rsi_series = ta.rsi(close, 14)

        if ema200.isna().iloc[-1] or rsi_series.isna().iloc[-1]:
            return None

        price = float(close.iloc[-1])
        rsi = float(rsi_series.iloc[-1])

        st_df = ta.supertrend(high, low, close, 10, 3)
        trend_col = [c for c in st_df.columns if c.startswith("SUPERTd")][0]
        trend_dir = int(st_df[trend_col].iloc[-1])

        tech_base = price > ema200.iloc[-1] and trend_dir == 1

        conviction = 0
        score_mod = 0
        news_sentiment = "Neutral"
        event_status = "✅ Safe"
        target = 0

        # -------- NEWS CHECK --------
        try:
            news = ticker_obj.news or []
            titles = " ".join(n.get("title", "").lower() for n in news[:5])
            danger = ["fraud", "investigation", "default", "bankruptcy", "scam", "legal"]
            if any(k in titles for k in danger):
                news_sentiment = "🚨 BLACK SWAN"
                score_mod -= 40
        except:
            pass

        # -------- EARNINGS CHECK --------
        try:
            cal = ticker_obj.calendar
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                next_date = pd.to_datetime(cal.iloc[0, 0]).date()
                days_to = (next_date - datetime.now().date()).days
                if 0 <= days_to <= 7:
                    event_status = f"🚨 EARNINGS ({next_date.strftime('%d-%b')})"
                    score_mod -= 20
        except:
            pass

        # -------- ANALYST TARGET --------
        try:
            info = ticker_obj.info or {}
            target = info.get("targetMeanPrice", 0) or 0
        except:
            pass

        # -------- SCORING --------
        if tech_base:
            conviction += 60

            if 48 <= rsi <= 62:
                conviction += 20
            elif 40 <= rsi < 48 or 62 < rsi <= 72:
                conviction += 10

            if target > price * 1.1:
                conviction += 10

            conviction += score_mod

        conviction = max(0, min(100, conviction))

        verdict = (
            "🔥 HIGH CONVICTION" if conviction >= 85 else
            "🚀 PASS" if conviction >= 60 else
            "🟡 WATCH" if tech_base else
            "❌ FAIL"
        )

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

    except:
        return None

# ================= MARKET PULSE =================
st.subheader("🌐 Market Pulse")
cols = st.columns(3)
bullish = 0

for i, (name, symbol) in enumerate(INDEX_BENCHMARKS.items()):
    try:
        idx = yf.download(symbol, period="1y", progress=False)
        if not idx.empty:
            p = idx["Close"].iloc[-1]
            e = ta.ema(idx["Close"], 200).iloc[-1]
            status = "🟢 BULLISH" if p > e else "🔴 BEARISH"
            if p > e: bullish += 1
            cols[i].metric(name, f"{p:,.0f}", status)
    except:
        pass

# ================= CONTROLS =================
st.sidebar.title("🔍 Fortress Controls")
selected_index = st.sidebar.selectbox("Universe", list(TICKER_GROUPS.keys()))
TICKERS = TICKER_GROUPS[selected_index]

# ================= MAIN SCAN =================
if st.button("🚀 START FULL SCAN", type="primary", use_container_width=True):

    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(TICKERS):
        status.text(f"🔍 [{i+1}/{len(TICKERS)}] {ticker}")
        try:
            tkr = yf.Ticker(ticker)
            data = yf.download(ticker, period="2y", progress=False)
            if not data.empty:
                row = check_institutional_fortress(ticker, data, tkr)
                if row:
                    results.append(row)
            time.sleep(0.7)
        except:
            pass

        progress.progress((i + 1) / len(TICKERS))

    if results:
        df = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.subheader("📊 FULL SCAN RESULTS")
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.warning("No valid data returned.")

st.caption("🛡️ Fortress 95 Pro — Clean Institutional Scanner | Logic Preserved")

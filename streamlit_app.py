# fortress_app.py - v7.0 INSTITUTIONAL SIGNAL ENGINE
import subprocess
import sys
import time
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------- CONFIG ----------------
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
st.title("🛡️ Fortress 95 Pro v7.0 — Institutional Swing Signals")

# =========================================================
# 🔥 CORE INSTITUTIONAL ENGINE (SIGNAL-ONLY)
# =========================================================
def check_institutional_fortress(
    ticker: str,
    data: pd.DataFrame,
    nifty_data: pd.DataFrame,
    sector_data: pd.DataFrame | None = None
):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if len(data) < 210 or len(nifty_data) < 60:
            return None

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        price = float(close.iloc[-1])

        # -------- INDICATORS --------
        ema200 = ta.ema(close, 200).iloc[-1]
        rsi = ta.rsi(close, 14).iloc[-1]
        atr = ta.atr(high, low, close, 14).iloc[-1]

        if np.isnan(ema200) or np.isnan(rsi) or np.isnan(atr):
            return None

        # -------- SUPERTREND DIRECTION --------
        st_df = ta.supertrend(high, low, close, 10, 3)
        trend_dir = int(
            st_df[[c for c in st_df.columns if c.startswith("SUPERTd")][0]].iloc[-1]
        )

        # -------- BASE TREND FILTER --------
        if not (price > ema200 and trend_dir == 1):
            return None

        conviction = 60

        # -------- RSI QUALITY --------
        if 48 <= rsi <= 62:
            conviction += 20
        elif 40 <= rsi < 48 or 62 < rsi <= 72:
            conviction += 10

        # =====================================================
        # 🔥 RELATIVE STRENGTH vs NIFTY (50-DAY)
        # =====================================================
        stock_ret = (close.iloc[-1] / close.iloc[-50]) - 1
        nifty_ret = (
            nifty_data["Close"].iloc[-1] / nifty_data["Close"].iloc[-50]
        ) - 1

        rs_alpha = stock_ret - nifty_ret

        if rs_alpha > 0.05:
            conviction += 15
        elif rs_alpha > 0:
            conviction += 10

        # =====================================================
        # 🏭 SECTOR ROTATION WEIGHTING
        # =====================================================
        if sector_data is not None and len(sector_data) > 60:
            s_close = sector_data["Close"]
            s_ema50 = ta.ema(s_close, 50).iloc[-1]
            s_ret = (s_close.iloc[-1] / s_close.iloc[-50]) - 1

            if s_close.iloc[-1] > s_ema50 and s_ret > nifty_ret:
                conviction += 10

        conviction = min(conviction, 100)

        # =====================================================
        # 🎯 10-DAY PROJECTED TARGET
        # =====================================================
        target_10d = price + (atr * 1.8)

        # =====================================================
        # 🚨 SIGNAL-ONLY MODE
        # =====================================================
        if conviction >= 85:
            verdict = "🔥 HIGH CONVICTION"
        elif conviction >= 70:
            verdict = "🚀 PASS"
        else:
            return None

        return {
            "Symbol": ticker,
            "Sector": SECTOR_MAP.get(ticker, "General"),
            "Verdict": verdict,
            "Score": conviction,
            "Price": round(price, 2),
            "RSI": round(rsi, 1),
            "ATR": round(atr, 2),
            "Target_10D": round(target_10d, 2),
            "RS_vs_NIFTY_%": round(rs_alpha * 100, 1)
        }

    except Exception:
        return None


# =========================================================
# 🌐 MARKET REGIME (FILTER ONLY)
# =========================================================
st.subheader("🌐 Market Regime")
cols = st.columns(3)
bullish_count = 0

for i, (name, symbol) in enumerate(INDEX_BENCHMARKS.items()):
    try:
        idx = yf.download(symbol, period="1y", progress=False)
        if not idx.empty:
            p = idx["Close"].iloc[-1]
            e = ta.ema(idx["Close"], 200).iloc[-1]
            status = "🟢 BULLISH" if p > e else "🔴 BEARISH"
            if p > e:
                bullish_count += 1
            cols[i].metric(name, f"{p:,.0f}", status)
    except:
        pass

market_state = (
    "✅ BULL MARKET" if bullish_count >= 2
    else "⚠️ MIXED"
    if bullish_count == 1
    else "🛑 BEAR"
)
st.success(f"{market_state} — {bullish_count}/3 indices bullish")

# =========================================================
# 🎛 CONTROLS
# =========================================================
st.sidebar.title("🔍 Fortress Controls")
selected_index = st.sidebar.selectbox("Universe", list(TICKER_GROUPS.keys()))
TICKERS = TICKER_GROUPS[selected_index]
st.sidebar.info(f"📊 {len(TICKERS)} stocks | Signal-Only Mode")

# =========================================================
# 🚀 MAIN SCAN
# =========================================================
if st.button("🚀 START SIGNAL SCAN", type="primary", use_container_width=True):

    nifty_data = yf.download(NIFTY_SYMBOL, period="1y", progress=False)

    signals = []
    total = len(TICKERS)
    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(TICKERS):
        status.text(f"🔍 [{i+1}/{total}] {ticker}")

        try:
            data = yf.download(ticker, period="1y", progress=False)
            if data.empty:
                continue

            sector_symbol = SECTOR_MAP.get(ticker)
            sector_data = None
            if sector_symbol:
                sector_data = yf.download(sector_symbol, period="1y", progress=False)

            signal = check_institutional_fortress(
                ticker,
                data,
                nifty_data,
                sector_data
            )

            if signal:
                signals.append(signal)
                st.toast(f"{signal['Verdict']}: {ticker}", icon="🔥")

            time.sleep(0.4)

        except:
            pass

        progress.progress((i + 1) / total)

    # =====================================================
    # 📊 OUTPUT — SIGNALS ONLY
    # =====================================================
    if signals:
        df = pd.DataFrame(signals).sort_values("Score", ascending=False)

        st.subheader("🎯 INSTITUTIONAL SWING SIGNALS (10-Day Horizon)")
        st.info("Only stocks with **institutional alignment + alpha + sector support**")

        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.warning("No institutional-grade signals today.")

st.markdown("---")
st.caption("🛡️ Fortress 95 Pro v7.0 — Institutional Logic | Signal-Only | No Noise")

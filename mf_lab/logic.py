import pandas as pd
import numpy as np
import yfinance as yf
import requests
import streamlit as st

@st.cache_data(ttl="1d")
def get_benchmark_data(ticker="^NSEI"):
    """Fetches Benchmark data with multi-index handling for True Alpha"""
    try:
        nifty = yf.download(ticker, period="5y", interval="1d", progress=False)
        if nifty.empty: return pd.DataFrame()
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty['ret'] = nifty['Close'].pct_change()
        return nifty[['ret']].dropna()
    except Exception as e:
        return pd.DataFrame()

def detect_integrity_issues(fund_df, benchmark_df, category):
    """Calculates Alpha, Beta, Tracking Error, Sortino, Drawdown, and Capture Ratios."""
    try:
        combined = pd.merge(fund_df[['date', 'ret']], benchmark_df[['ret']],
                             left_on='date', right_index=True, suffixes=('_f', '_b'))
        if len(combined) < 200: return None

        f_ret, b_ret = combined['ret_f'], combined['ret_b']

        # 1. Capture Ratios (Consistency in Bull/Bear cycles)
        up_mkt = b_ret > 0
        down_mkt = b_ret < 0
        upside_cap = (f_ret[up_mkt].mean() / b_ret[up_mkt].mean()) * 100 if b_ret[up_mkt].mean() != 0 else 100
        downside_cap = (f_ret[down_mkt].mean() / b_ret[down_mkt].mean()) * 100 if b_ret[down_mkt].mean() != 0 else 100

        # 2. Risk Metrics (Beta & Tracking Error)
        covariance = np.cov(f_ret, b_ret)[0][1]
        variance = np.var(b_ret)
        beta = covariance / variance if variance != 0 else 1.0
        tracking_error = (f_ret - b_ret).std() * np.sqrt(252) * 100

        # 3. Rolling Alpha (60-day smoothed)
        alpha = (f_ret - b_ret).rolling(60).mean().mean() * 252 * 100

        # 4. Sortino Ratio (Risk-Free Rate = 6%)
        RFR = 0.06
        excess_ret = (f_ret.mean() * 252) - RFR
        neg_ret = f_ret[f_ret < 0]
        downside_dev = neg_ret.std() * np.sqrt(252)
        sortino = excess_ret / downside_dev if downside_dev > 0 else 0

        # 5. Max Drawdown & Win Rate
        cum = (1 + f_ret).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
        win_rate = (f_ret > b_ret).rolling(252).mean().mean() * 100

        # 6. Style Drift Detection
        drift = "✅ Stable"
        if category == "Large Cap" and beta > 1.15: drift = "🚨 Beta Drift (Aggressive)"
        if tracking_error > 8.0: drift = "🚨 Strategy Drift"

        return {
            "alpha": alpha, "beta": beta, "te": tracking_error, "sortino": sortino,
            "max_dd": max_dd, "win_rate": win_rate, "drift": drift,
            "upside": upside_cap, "downside": downside_cap
        }
    except: return None

@st.cache_data(ttl="7d")
def discover_funds(limit=250):
    """Auto-discovers Direct Growth Equity funds from mfapi.in"""
    try:
        url = "https://api.mfapi.in/mf"
        schemes = requests.get(url).json()
        keywords = ["flexi", "large", "mid", "small", "focused", "growth", "direct"]
        exclusions = ["regular", "idcw", "etf"]

        candidates = []
        for s in schemes:
            name = s['schemeName'].lower()
            if all(k in name for k in ["direct", "growth"]) and \
               any(k in name for k in keywords) and \
               not any(ex in name for ex in exclusions):
                candidates.append(s)
        return candidates[:limit]
    except: return []

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

@st.cache_data(ttl="1d")
def get_benchmark_data(ticker="^NSEI"):
    """Fetches Benchmark data to calculate true Alpha"""
    try:
        nifty = yf.download(ticker, period="5y", interval="1d", progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty['ret'] = nifty['Close'].pct_change()
        return nifty.dropna()
    except Exception as e:
        print(f"Benchmark Fetch Error: {e}")
        return pd.DataFrame()

def detect_integrity_issues(fund_df, benchmark_df, category):
    """
    Calculates Alpha, Beta, Tracking Error, and Style Drift.
    Tracking Error = Std Dev of (Fund Return - Benchmark Return)
    """
    try:
        # Align dates for precise comparison
        combined = pd.merge(fund_df[['date', 'ret']], benchmark_df[['ret']],
                             left_on='date', right_index=True, suffixes=('_f', '_b'))

        if len(combined) < 100: return None

        # Ratios
        f_ret = combined['ret_f']
        b_ret = combined['ret_b']

        # Beta calculation
        covariance = np.cov(f_ret, b_ret)[0][1]
        variance = np.var(b_ret)
        beta = covariance / variance if variance != 0 else 0

        # Tracking Error & Alpha
        tracking_diff = f_ret - b_ret
        tracking_error = tracking_diff.std() * np.sqrt(252) * 100
        alpha = (f_ret.mean() - b_ret.mean()) * 252 * 100

        # Drift Detection
        drift = "✅ Stable"
        if category == "Large Cap" and beta > 1.15: drift = "🚨 Beta Drift (Aggressive)"
        if tracking_error > 8.0: drift = "🚨 Tracking Drift (Ghost Fund?)"

        return {"alpha": alpha, "beta": beta, "te": tracking_error, "drift": drift}
    except: return None

@st.cache_data(ttl="7d")
def discover_funds(limit=250):
    try:
        url = "https://api.mfapi.in/mf"
        schemes = requests.get(url).json()
        keywords = ["flexi", "large", "mid", "small", "focused", "growth", "direct"]

        # Filter for Direct Growth Equity
        candidates = [s for s in schemes if all(k in s['schemeName'].lower() for k in ["direct", "growth"])
                      and any(k in s['schemeName'].lower() for k in keywords)]
        return candidates[:limit]
    except: return []

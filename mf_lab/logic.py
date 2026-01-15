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
    Calculates Alpha, Beta, Tracking Error, Sortino, Max Drawdown, and Win Rate.
    """
    try:
        # Align dates for precise comparison
        combined = pd.merge(fund_df[['date', 'ret']], benchmark_df[['ret']],
                             left_on='date', right_index=True, suffixes=('_f', '_b'))

        if len(combined) < 100: return None

        # Ratios
        f_ret = combined['ret_f']
        b_ret = combined['ret_b']

        # 1. Beta calculation
        covariance = np.cov(f_ret, b_ret)[0][1]
        variance = np.var(b_ret)
        beta = covariance / variance if variance != 0 else 0

        # 2. Tracking Error
        tracking_diff = f_ret - b_ret
        tracking_error = tracking_diff.std() * np.sqrt(252) * 100

        # 3. Rolling Alpha (Smoothed)
        # alpha = (f_ret.mean() - b_ret.mean()) * 252 * 100  <-- Old
        rolling_alpha = (f_ret - b_ret).rolling(60).mean()
        alpha = rolling_alpha.mean() * 252 * 100

        # 4. Sortino Ratio (Risk-Free Rate = 6%)
        RFR = 0.06
        # Annualized excess return
        excess_ret = (f_ret.mean() * 252) - RFR
        # Downside deviation
        neg_ret = f_ret[f_ret < 0]
        downside_dev = neg_ret.std() * np.sqrt(252)

        if downside_dev > 0:
            sortino = excess_ret / downside_dev
        else:
            sortino = 0 # Or some high number if no negative returns, but 0 is safe

        # 5. Max Drawdown
        cum = (1 + f_ret).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min() * 100 # Returns negative percentage, e.g. -20.5

        # 6. Win Rate (Consistency)
        # Ratio of rolling 1-year (252 days) periods with positive returns?
        # User prompt: win_rate = (df['ret'] > 0).rolling(252).mean().mean()*100
        # This calculates the average % of positive days in a rolling 252 window?
        # No, (df['ret'] > 0) is a boolean series of daily up/down.
        # rolling(252).mean() gives the % of positive days in that year.
        # .mean() gives the average of that over time.
        # So it's "Average Annual Percentage of Positive Days" - a bit specific but I will follow the formula.
        win_rate = (f_ret > 0).rolling(252).mean().mean() * 100

        # Drift Detection
        drift = "✅ Stable"
        if category == "Large Cap" and beta > 1.15: drift = "🚨 Beta Drift (Aggressive)"
        if tracking_error > 8.0: drift = "🚨 Tracking Drift (Ghost Fund?)"

        return {
            "alpha": alpha,
            "beta": beta,
            "te": tracking_error,
            "sortino": sortino,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "drift": drift
        }
    except Exception as e:
        # print(f"Error in calculation: {e}")
        return None

@st.cache_data(ttl="7d")
def discover_funds(limit=250):
    try:
        url = "https://api.mfapi.in/mf"
        schemes = requests.get(url).json()
        keywords = ["flexi", "large", "mid", "small", "focused", "growth", "direct"]

        # Exclusions
        exclusions = ["regular", "idcw", "etf"]

        # Filter for Direct Growth Equity and strict exclusions
        candidates = []
        for s in schemes:
            name = s['schemeName'].lower()
            if all(k in name for k in ["direct", "growth"]) and \
               any(k in name for k in keywords) and \
               not any(ex in name for ex in exclusions):
                candidates.append(s)

        return candidates[:limit]
    except: return []

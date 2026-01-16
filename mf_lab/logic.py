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
        # Keep Close for rolling calculations if needed, but ret is sufficient for now
        return nifty[['ret']].dropna()
    except Exception as e:
        return pd.DataFrame()

def detect_integrity_issues(fund_df, benchmark_df, category):
    """Calculates Alpha, Beta, Tracking Error, Sortino, Drawdown, and Capture Ratios."""
    try:
        combined = pd.merge(fund_df[['date', 'ret']], benchmark_df[['ret']],
                             left_on='date', right_index=True, suffixes=('_f', '_b'))

        # Require substantial history for accurate "Fortress" analysis
        if len(combined) < 200: return None

        f_ret, b_ret = combined['ret_f'], combined['ret_b']

        # 1. Capture Ratios (Consistency in Bull/Bear cycles)
        # Maintained as Daily Ratio method as per requirements
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
        # Annualized rolling 60-day Alpha
        alpha = (f_ret - b_ret).rolling(60).mean().mean() * 252 * 100

        # 4. Sortino Ratio (Risk-Free Rate = 6%)
        # Numerator uses CAGR over the period, not annualized mean
        RFR = 0.06
        days = len(f_ret)
        years = days / 252.0
        total_ret_f = (1 + f_ret).prod() - 1
        cagr_f = (1 + total_ret_f) ** (1/years) - 1 if years > 0 else 0

        excess_ret = cagr_f - RFR
        neg_ret = f_ret[f_ret < 0]
        downside_dev = neg_ret.std() * np.sqrt(252)
        sortino = excess_ret / downside_dev if downside_dev > 0 else 0

        # 5. Max Drawdown & Win Rate
        cum = (1 + f_ret).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100

        # Win Rate: Percentage of rolling 252-day periods where Fund CAGR > Benchmark CAGR
        # Calculate rolling 1-year returns
        rolling_f = (1 + f_ret).rolling(252).apply(np.prod, raw=True) - 1
        rolling_b = (1 + b_ret).rolling(252).apply(np.prod, raw=True) - 1

        # Compare and calculate percentage
        wins = (rolling_f > rolling_b).dropna()
        win_rate = wins.mean() * 100 if not wins.empty else 0.0

        # 6. Style Drift Detection (Tiered Beta Thresholds)
        drift_reasons = []
        if tracking_error > 8.0: drift_reasons.append("High TE")

        beta_drift = False
        if category == "Large Cap" and beta > 1.15: beta_drift = True
        elif category == "Flexi/Other" and beta > 1.20: beta_drift = True
        elif category == "Mid Cap" and beta > 1.25: beta_drift = True
        elif category == "Small Cap" and beta > 1.40: beta_drift = True

        if beta_drift: drift_reasons.append("Beta Drift")

        if not drift_reasons:
            drift = "✅ Stable"
        else:
            drift = f"🚨 {' & '.join(drift_reasons)}"

        return {
            "alpha": alpha, "beta": beta, "te": tracking_error, "sortino": sortino,
            "max_dd": max_dd, "win_rate": win_rate, "drift": drift,
            "upside": upside_cap, "downside": downside_cap
        }
    except Exception as e:
        # print(f"Error in detect_integrity_issues: {e}")
        return None

@st.cache_data(ttl="7d")
def discover_funds(limit=None):
    """
    Auto-discovers Direct Growth Equity funds from mfapi.in.
    Filters: Direct, Growth, Keywords (Flexi, Large, Mid, Small, Focused, Value, Contra).
    Exclusions: Regular, IDCW, ETF.
    Limit: Defaults to None (fetch all), can be set for testing.
    """
    try:
        url = "https://api.mfapi.in/mf"
        schemes = requests.get(url).json()
        keywords = ["flexi", "large", "mid", "small", "focused", "value", "contra"]
        required = ["direct", "growth"]
        exclusions = ["regular", "idcw", "etf"]

        candidates = []
        for s in schemes:
            name = s['schemeName'].lower()

            # Must have ALL required terms
            if not all(req in name for req in required):
                continue

            # Must have AT LEAST ONE keyword
            if not any(k in name for k in keywords):
                continue

            # Must NOT have ANY exclusions
            if any(ex in name for ex in exclusions):
                continue

            candidates.append(s)

        if limit:
            return candidates[:limit]
        return candidates
    except: return []

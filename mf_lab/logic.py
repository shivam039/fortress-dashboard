import pandas as pd
import numpy as np
import yfinance as yf
import requests
import streamlit as st
from fortress_config import TICKER_GROUPS

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

def get_category(scheme_name):
    name = scheme_name.lower()

    # Priority 1: Specific Styles
    if "focused" in name: return "Focused"
    if "value" in name: return "Value"
    if "contra" in name: return "Contra"
    if "elss" in name: return "ELSS"

    # Priority 2: Market Cap (Handle exclusions)
    if "mid" in name and "small" in name: return "Small Cap"

    if "small" in name: return "Small Cap"
    if "mid" in name: return "Mid Cap"
    if "large" in name: return "Large Cap"

    # Priority 3: Flexi/Other (Fallback)
    return "Flexi/Other"

def calculate_fortress_score(cat_df):
    """
    Calculates the 0-100 normalized Fortress Score for a category dataframe.
    """
    if cat_df.empty: return cat_df

    df = cat_df.copy()
    if len(df) > 1:
        c_min, c_max = df['Score'].min(), df['Score'].max()
        if c_max != c_min:
            # Min-Max Scaling
            df['Fortress Score'] = ((df['Score'] - c_min) / (c_max - c_min)) * 100
        else:
            df['Fortress Score'] = 50.0
    else:
        df['Fortress Score'] = 100.0

    df['Fortress Score'] = df['Fortress Score'].round(1)
    return df

def calculate_drift_status(row):
    """
    Determines Integrity Badge and Status based on thresholds.
    Returns: (Badge, Status, Drift Score, Message)
    """
    try:
        # Defaults if columns missing
        beta = row.get('beta', row.get('Beta', 0))
        te = row.get('te', row.get('Tracking Error', 0))
        cat = row.get('Category', 'Flexi/Other')

        # 1. Tiered Beta Thresholds
        beta_limit = 1.15
        if cat == "Flexi/Other": beta_limit = 1.20
        elif cat == "Mid Cap": beta_limit = 1.25
        elif cat == "Small Cap": beta_limit = 1.45 # Updated per requirements

        # 2. Tracking Error Threshold
        te_limit = 8.0

        # Scoring
        drift_score = 0
        issues = []

        if beta > beta_limit:
            drift_score += 50
            issues.append(f"Beta {beta:.2f} > {beta_limit}")

        if te > te_limit:
            drift_score += 50
            issues.append(f"TE {te:.1f} > {te_limit}")

        # Classification
        if drift_score >= 100:
            return "🚨", "Critical", drift_score, f"Critical Drift: {' & '.join(issues)}"
        elif drift_score >= 50:
            return "⚠️", "Moderate", drift_score, f"Moderate Drift: {' & '.join(issues)}"
        else:
            return "✅", "Stable", 0, "Stable"

    except Exception:
        return "❓", "Unknown", 0, "Data Missing"

def apply_drift_status(df):
    """Applies drift calculation to a dataframe."""
    if df.empty: return df

    results = df.apply(calculate_drift_status, axis=1)
    df['Integrity'] = [r[0] for r in results]
    df['Drift Status'] = [r[1] for r in results]
    df['Drift Score'] = [r[2] for r in results]
    df['Drift Message'] = [r[3] for r in results]
    return df

def fetch_market_data():
    """
    Fetches Nifty 50 close, Sector Returns (1M), and Market Breadth (A/D).
    Returns a dictionary of metrics.
    """
    data = {}

    # 1. Nifty 50 Status
    try:
        nifty = yf.download("^NSEI", period="1d", progress=False)
        if not nifty.empty:
            close_val = nifty['Close'].iloc[-1]
            if isinstance(close_val, pd.Series): close_val = close_val.iloc[0] # Handle multi-col mismatch
            data['nifty_level'] = f"{close_val:,.2f}"
        else:
            data['nifty_level'] = "N/A"
    except:
        data['nifty_level'] = "Unavailable"

    # 2. Sector Rotation (1 Month Return)
    sectors = {
        "Banking": "^NSEBANK",
        "IT": "^CNXIT",
        "Auto": "^CNXAUTO",
        "FMCG": "^CNXFMCG",
        "Infra": "^CNXINFRA"
    }

    sector_perf = {}
    for name, ticker in sectors.items():
        try:
            sec_data = yf.download(ticker, period="1mo", progress=False)
            if not sec_data.empty and len(sec_data) > 1:
                start = sec_data['Close'].iloc[0]
                end = sec_data['Close'].iloc[-1]
                # scalar check
                if hasattr(start, 'iloc'): start = start.iloc[0]
                if hasattr(end, 'iloc'): end = end.iloc[0]

                ret = ((end - start) / start) * 100
                sector_perf[name] = ret
            else:
                sector_perf[name] = None
        except:
            sector_perf[name] = None

    data['sectors'] = sector_perf

    # 3. Market Breadth (Nifty 50 A/D)
    # Using TICKER_GROUPS from config, already imported
    try:
        nifty_tickers = TICKER_GROUPS.get("Nifty 50", [])
        if nifty_tickers:
            # Batch download 2 days
            breadth_df = yf.download(nifty_tickers, period="2d", group_by='ticker', progress=False)

            advances = 0
            declines = 0

            for t in nifty_tickers:
                try:
                    # Check if ticker column exists (top level)
                    if t in breadth_df.columns:
                        closes = breadth_df[t]['Close']
                        if len(closes) >= 2:
                            prev = closes.iloc[-2]
                            curr = closes.iloc[-1]
                            if curr > prev: advances += 1
                            elif curr < prev: declines += 1
                except: continue

            data['advances'] = advances
            data['declines'] = declines
            data['ad_ratio'] = f"{advances}/{declines}" if declines > 0 else "Positive"
        else:
            data['ad_ratio'] = "N/A"
    except:
        data['ad_ratio'] = "Unavailable"

    return data

def generate_health_check_report(current_df, previous_df=None):
    """
    Generates the Markdown report for the Health Check.
    """

    # --- 1. PREP DATA ---
    # Apply Categorization & Drift to Current
    if 'Category' not in current_df.columns:
        current_df['Category'] = current_df['Symbol'].apply(get_category)
    current_df = apply_drift_status(current_df)

    # Previous Data Prep (for Hidden Gem)
    if previous_df is not None and not previous_df.empty:
        if 'Category' not in previous_df.columns:
            previous_df['Category'] = previous_df['Symbol'].apply(get_category)
        # We need raw Score to calculate Fortress Score correctly per category *at that time*?
        # Actually, let's just assume we want to compare the final Fortress Score.
        # But Fortress Score is relative. So we should recalculate it for the previous scan to be fair?
        # Or just use raw Score delta? The prompt says "Delta in its Fortress Score".
        # So I need to calculate Fortress Score for both datasets first.
        pass

    # --- 2. CALCULATE SCORES ---
    # Categories to iterate
    categories = ["Large Cap", "Mid Cap", "Small Cap", "Flexi/Other", "Focused", "Value", "Contra", "ELSS"]
    # Wait, get_category produces: Focused, Value, Contra, ELSS, Small Cap, Mid Cap, Large Cap, Flexi/Other.
    # I should use the unique values found or a fixed list.

    current_scored = pd.DataFrame()
    for cat in current_df['Category'].unique():
        sub = current_df[current_df['Category'] == cat]
        sub = calculate_fortress_score(sub)
        current_scored = pd.concat([current_scored, sub])

    previous_scored = pd.DataFrame()
    if previous_df is not None and not previous_df.empty:
        for cat in previous_df['Category'].unique():
            sub = previous_df[previous_df['Category'] == cat]
            sub = calculate_fortress_score(sub)
            previous_scored = pd.concat([previous_scored, sub])

    # --- 3. MARKET DATA ---
    mkt = fetch_market_data()

    # Sector Rotation Logic
    # Find Top performing and Bottom performing sector
    secs = mkt.get('sectors', {})
    valid_secs = {k:v for k,v in secs.items() if v is not None}

    sector_summary = "Sector Data Unavailable"
    if valid_secs:
        sorted_secs = sorted(valid_secs.items(), key=lambda x: x[1], reverse=True)
        top_sec = sorted_secs[0]
        bot_sec = sorted_secs[-1]
        sector_summary = f"Capital is rotating into **{top_sec[0]}** (+{top_sec[1]:.2f}%) and out of **{bot_sec[0]}** ({bot_sec[1]:.2f}%) over the last 30 days."

    # --- 4. CATEGORY LEADERBOARDS ---
    # We want Top Fund from each category
    leaderboard_md = ""
    # Defined order for report
    report_cats = ["Large Cap", "Mid Cap", "Small Cap", "Flexi/Other", "Value", "Contra", "Focused", "ELSS"]

    for cat in report_cats:
        cat_data = current_scored[current_scored['Category'] == cat]
        if not cat_data.empty:
            leader = cat_data.sort_values("Fortress Score", ascending=False).iloc[0]
            # Identify specific strength based on metrics?
            # Random "Tag" based on logic:
            tag = "High Performance"
            if leader['Sortino'] > 2: tag = "Consistency: High"
            elif leader['Upside Cap'] > 110: tag = "Upside Potential: Elite"
            elif leader['Downside Cap'] < 85: tag = "Downside Shield: Strong"

            leaderboard_md += f"🏆 **{cat} Leader**: {leader['Symbol']} (Score: {leader['Fortress Score']}) | {tag}\n\n"

    # --- 5. STRATEGY WATCHDOG ---
    stable_count = len(current_scored[current_scored['Drift Status'] == 'Stable'])
    breach_count = len(current_scored[current_scored['Drift Status'] != 'Stable'])
    total_audited = len(current_scored)

    stable_pct = (stable_count / total_audited * 100) if total_audited > 0 else 0

    watchdog_md = ""
    # List top critical/moderate
    drifters = current_scored[current_scored['Drift Status'] != 'Stable'].sort_values("Drift Score", ascending=False).head(3)
    if not drifters.empty:
        for _, row in drifters.iterrows():
            icon = "🚨" if row['Drift Status'] == "Critical" else "⚠️"
            watchdog_md += f"{icon} **{row['Drift Status']}**: {{{row['Symbol']}}} — {row['Drift Message']}\n\n"

    watchdog_md += f"✅ **Stable**: {stable_pct:.0f}% of audited funds are maintaining their mandate integrity."

    # --- 6. DISCOVERY OPPORTUNITY ---
    gem_md = "New Audit Cycle: Benchmarking in progress."

    if not previous_scored.empty:
        # Merge current and previous on Symbol
        merged = pd.merge(current_scored[['Symbol', 'Fortress Score']],
                          previous_scored[['Symbol', 'Fortress Score']],
                          on='Symbol', suffixes=('_curr', '_prev'))

        merged['Delta'] = merged['Fortress Score_curr'] - merged['Fortress Score_prev']
        merged = merged.sort_values("Delta", ascending=False)

        if not merged.empty:
            top_gem = merged.iloc[0]
            if top_gem['Delta'] > 0:
                gem_md = f"**Hidden Gem**: {{{top_gem['Symbol']}}} has jumped **+{top_gem['Delta']:.1f} points** in Fortress Score this month."

    # --- 7. COMPILE REPORT ---
    report = f"""
🛡️ **Fortress Monthly Health Check: {pd.Timestamp.now().strftime('%B %Y')} Template**
Subject: 🛡️ Fortress Audit: Monthly Portfolio Integrity & Discovery Report

**1. Executive Market Summary (The Heartbeat)**

*   **Nifty 50 Status**: {mkt['nifty_level']}
*   **Sector Rotation**: {sector_summary}
*   **Market Breadth**: {mkt.get('advances', 0)} Advances / {mkt.get('declines', 0)} Declines (A/D Ratio: {mkt.get('ad_ratio', 'N/A')})

**2. Category Leaderboards (Top Fortress Scores)**

{leaderboard_md}
**3. Strategy Watchdog (Integrity Alerts)**

{watchdog_md}

**4. The "Discovery" Opportunity**

{gem_md}
"""
    return report

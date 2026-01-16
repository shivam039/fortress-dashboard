import pandas as pd
import requests
import time
import sys
import os
import yfinance as yf
from datetime import datetime

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

# Import Logic (Re-using existing logic to ensure consistency)
# We will use the functions, but might need to bypass the @st.cache_data decorators if they cause issues
# in a non-Streamlit environment. However, usually they just work or can be bypassed.
from mf_lab.logic import detect_integrity_issues, discover_funds, get_category
from utils.db import log_scan_results

def fetch_benchmark_data_headless(ticker):
    """Fetches Benchmark data without Streamlit caching for the background script"""
    try:
        # Using 5y history to cover the required analysis window
        nifty = yf.download(ticker, period="5y", interval="1d", progress=False)
        if nifty.empty: return pd.DataFrame()
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty['ret'] = nifty['Close'].pct_change()
        return nifty[['ret']].dropna()
    except Exception as e:
        print(f"Error fetching benchmark {ticker}: {e}")
        return pd.DataFrame()

def run_audit(limit=None):
    print(f"[{datetime.now()}] Starting Fortress Discovery Audit...")

    # 1. Fetch Benchmarks
    print("Fetching Benchmarks...")
    benchmarks = {
        'Large Cap': fetch_benchmark_data_headless("^NSEI"),
        'Mid Cap': fetch_benchmark_data_headless("^NSEMDCP50"),
        'Small Cap': fetch_benchmark_data_headless("^CNXSC"),
        'Flexi/Other': fetch_benchmark_data_headless("^NSEI")
    }

    # 2. Discover Funds
    print("Discovering Funds...")
    candidates = discover_funds(limit=limit)
    print(f"Found {len(candidates)} candidates for audit.")

    results = []

    # 3. Audit Loop
    for i, c in enumerate(candidates):
        try:
            scheme_code = c['schemeCode']
            scheme_name = c['schemeName']

            # Rate Limiting
            time.sleep(0.4)

            # Fetch NAV
            url = f"https://api.mfapi.in/mf/{scheme_code}"
            resp = requests.get(url)
            if resp.status_code != 200:
                continue

            data = resp.json()
            if not data.get('data'):
                continue

            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = df['nav'].astype(float)
            df = df.sort_values('date')
            df['ret'] = df['nav'].pct_change()

            # Survival Filter: Skip if < 750 days of history
            # (Roughly 3 years of trading days ~ 750)
            if len(df) < 750:
                continue

            cat = get_category(scheme_name)
            bench = benchmarks.get(cat)
            if bench is None or bench.empty:
                continue

            # Calculate Metrics
            metrics = detect_integrity_issues(df, bench, cat)

            if metrics:
                # Fortress Pro Scoring Formula
                # Score = (Alpha * 0.4) + (Sortino * 0.3) + ((100 - Downside) * 0.3)
                raw_score = (metrics['alpha'] * 0.4) + \
                            (metrics['sortino'] * 0.3) + \
                            ((100 - metrics['downside']) * 0.3)

                results.append({
                    "Symbol": scheme_name[:100], # Truncate for DB safety
                    "Scheme Code": scheme_code,
                    "Category": cat,
                    "Score": raw_score,
                    "Alpha (True)": metrics['alpha'],
                    "Sortino": metrics['sortino'],
                    "Upside Cap": metrics['upside'],
                    "Downside Cap": metrics['downside'],
                    "Max Drawdown": metrics['max_dd'],
                    "Win Rate": metrics['win_rate'],
                    "Verdict": metrics['drift'],
                    "Price": df['nav'].iloc[-1],
                    "Beta": metrics['beta'],
                    "Tracking Error": metrics['te']
                })

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(candidates)}...")

        except Exception as e:
            # print(f"Error processing {c.get('schemeName', 'Unknown')}: {e}")
            continue

    # 4. Persistence
    if results:
        final_df = pd.DataFrame(results)
        print(f"Audit Complete. Saving {len(final_df)} records to database...")

        # We explicitly write to 'scan_mf' table
        log_scan_results(final_df, "scan_mf")
        print("Database update successful.")
    else:
        print("No results found or all filtered out.")

if __name__ == "__main__":
    # If run directly, check for arguments or run default
    # Example: python cron_mf_audit.py --limit 10
    limit = None
    if len(sys.argv) > 1:
        try:
            # Simple arg parsing for limit
            arg = sys.argv[1]
            if arg.startswith("--limit="):
                limit = int(arg.split("=")[1])
        except:
            pass

    run_audit(limit)

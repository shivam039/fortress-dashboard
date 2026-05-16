import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import from streamlit_app instead
from streamlit_app import _run_scan_directly
import pandas as pd
import fortress_config

payload = {
    "universe": "Nifty 50",
    "portfolio_val": 1000000.0,
    "risk_pct": 0.01,
    "weights": {
        "technical": 0.5,
        "fundamental": 0.25,
        "sentiment": 0.15,
        "context": 0.1,
    },
    "enable_regime": True,
    "liquidity_cr_min": 8.0,
    "market_cap_cr_min": 1500.0,
    "price_min": 80.0,
    "broker": "Zerodha",
}

print("Running diagnostic scan for RELIANCE...")
original_tickers = fortress_config.TICKER_GROUPS["Nifty 50"]
fortress_config.TICKER_GROUPS["Nifty 50"] = ["RELIANCE.NS"]

try:
    results = _run_scan_directly(payload)
    df = pd.DataFrame(results)
    
    if not df.empty:
        print("\nRELIANCE Results:")
        for col in df.columns:
            print(f"{col}: {df[col].iloc[0]}")
    else:
        print("\nNo results returned for RELIANCE.")

finally:
    # Restore original tickers
    fortress_config.TICKER_GROUPS["Nifty 50"] = original_tickers

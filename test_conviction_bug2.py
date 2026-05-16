#!/usr/bin/env python3
"""
Test with zero values to see if that's the issue
"""
import sys
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ENGINE_DIR = ROOT_DIR / "engine"

for path in (str(ENGINE_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from stock_scanner.logic import apply_advanced_scoring, DEFAULT_SCORING_CONFIG

# Create sample data with zero conviction
sample_data = [
    {
        "Symbol": "STOCK1",
        "Score": 0,  # Zero conviction
        "Price": 100.0,
        "RSI": 50.0,
        "News": "Neutral",
        "Events": "✅ Safe",
        "Sector": "IT",
        "Avg_Value_20D_Cr": 500.0,
        "Market_Cap_Cr": 800000.0,
        "Debt_To_Equity": 0.1,
        "Technical_Raw": 0.0,  # Zero technical
        "Fundamental_Raw": 30.0,
        "Sentiment_Raw": 50.0,
        "Context_Raw": 30.0,
        "RS_Score": 0.0,
        "RS_Composite": 0.9,
        "Vol_Surge_Ratio": 1.0,
        "Dist_52W_High_Pct": 20.0,
        "Extension_Pct": 0.0,
        "Is_Coiling": False,
        "Black_Swan_Flag": 0,
        "Regime_Multiplier": 1.0
    },
    {
        "Symbol": "STOCK2",
        "Score": 50,  # Good conviction
        "Price": 150.0,
        "RSI": 60.0,
        "News": "Neutral",
        "Events": "✅ Safe",
        "Sector": "Finance",
        "Avg_Value_20D_Cr": 500.0,
        "Market_Cap_Cr": 800000.0,
        "Debt_To_Equity": 0.1,
        "Technical_Raw": 50.0,  # Good technical
        "Fundamental_Raw": 45.0,
        "Sentiment_Raw": 50.0,
        "Context_Raw": 50.0,
        "RS_Score": 2.0,
        "RS_Composite": 1.2,
        "Vol_Surge_Ratio": 1.5,
        "Dist_52W_High_Pct": 5.0,
        "Extension_Pct": 2.0,
        "Is_Coiling": False,
        "Black_Swan_Flag": 0,
        "Regime_Multiplier": 1.0
    }
]

df = pd.DataFrame(sample_data)
print("BEFORE apply_advanced_scoring:")
for idx, row in df.iterrows():
    print(f"  {row['Symbol']}: Score={row['Score']}, Technical_Raw={row['Technical_Raw']}, Context_Raw={row['Context_Raw']}")

scored_df = apply_advanced_scoring(df, DEFAULT_SCORING_CONFIG)

print("\nAFTER apply_advanced_scoring:")
for idx, row in scored_df.iterrows():
    print(f"  {row['Symbol']}: Score={row['Score']}, Technical_Score={row['Technical_Score']}, Context_Score={row['Context_Score']}, Quality_Gate_Pass={row['Quality_Gate_Pass']}, Verdict={row['Verdict']}")

#!/usr/bin/env python3
"""
Test case where all stocks have identical raw scores
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

# Create sample data where ALL stocks have identical scores
sample_data = [
    {
        "Symbol": "STOCK1",
        "Score": 30,  # All same score
        "Price": 100.0,
        "RSI": 50.0,
        "News": "Neutral",
        "Events": "✅ Safe",
        "Sector": "IT",
        "Avg_Value_20D_Cr": 500.0,
        "Market_Cap_Cr": 800000.0,
        "Debt_To_Equity": 0.1,
        "Technical_Raw": 10.0,  # All same raw
        "Fundamental_Raw": 30.0,
        "Sentiment_Raw": 50.0,
        "Context_Raw": 20.0,
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
        "Score": 30,  # All same score
        "Price": 150.0,
        "RSI": 50.0,
        "News": "Neutral",
        "Events": "✅ Safe",
        "Sector": "Finance",
        "Avg_Value_20D_Cr": 500.0,
        "Market_Cap_Cr": 800000.0,
        "Debt_To_Equity": 0.1,
        "Technical_Raw": 10.0,  # All same raw
        "Fundamental_Raw": 30.0,
        "Sentiment_Raw": 50.0,
        "Context_Raw": 20.0,
        "RS_Score": 0.0,
        "RS_Composite": 0.9,
        "Vol_Surge_Ratio": 1.0,
        "Dist_52W_High_Pct": 20.0,
        "Extension_Pct": 0.0,
        "Is_Coiling": False,
        "Black_Swan_Flag": 0,
        "Regime_Multiplier": 1.0
    }
]

df = pd.DataFrame(sample_data)
scored_df = apply_advanced_scoring(df, DEFAULT_SCORING_CONFIG)

print("Test: All stocks have identical raw scores\n")
for idx, row in scored_df.iterrows():
    print(f"{row['Symbol']}:")
    print(f"  Original Score: {df.iloc[idx]['Score']}")
    print(f"  Technical_Raw: {row['Technical_Raw']} -> Technical_Score: {row['Technical_Score']}")
    print(f"  Fundamental_Raw: {row['Fundamental_Raw']} -> Fundamental_Score: {row['Fundamental_Score']}")
    print(f"  Sentiment_Raw: {row['Sentiment_Raw']} -> Sentiment_Score: {row['Sentiment_Score']}")
    print(f"  Context_Raw: {row['Context_Raw']} -> Context_Score: {row['Context_Score']}")
    print(f"  Final Score: {row['Score']}")
    print()

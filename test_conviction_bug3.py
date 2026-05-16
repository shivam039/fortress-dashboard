#!/usr/bin/env python3
"""
Test with zero values - show all scores
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
scored_df = apply_advanced_scoring(df, DEFAULT_SCORING_CONFIG)

print("Detailed Score Breakdown:")
for idx, row in scored_df.iterrows():
    print(f"\n{row['Symbol']}:")
    print(f"  Original Score: {df.iloc[idx]['Score']}")
    print(f"  Technical_Raw: {row['Technical_Raw']} -> Technical_Score: {row['Technical_Score']}")
    print(f"  Fundamental_Raw: {row['Fundamental_Raw']} -> Fundamental_Score: {row['Fundamental_Score']}")
    print(f"  Sentiment_Raw: {row['Sentiment_Raw']} -> Sentiment_Score: {row['Sentiment_Score']}")
    print(f"  Context_Raw: {row['Context_Raw']} -> Context_Score: {row['Context_Score']}")
    print(f"  Score_Pre_Regime: {row.get('Score_Pre_Regime', 'N/A')}")
    print(f"  Regime_Multiplier: {row['Regime_Multiplier']}")
    print(f"  Final Score: {row['Score']}")
    print(f"  Verdict: {row['Verdict']}")

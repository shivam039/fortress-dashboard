#!/usr/bin/env python3
"""
Quick test to diagnose the conviction score always showing 0 issue
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

# Create a sample dataframe that would come from check_institutional_fortress
sample_data = [
    {
        "Symbol": "INFY",
        "Score": 65,  # Raw conviction from Phase 1
        "Price": 1800.0,
        "RSI": 55.0,
        "News": "Neutral",
        "Events": "✅ Safe",
        "Sector": "IT",
        "Avg_Value_20D_Cr": 500.0,
        "Market_Cap_Cr": 800000.0,
        "Debt_To_Equity": 0.1,
        "Technical_Raw": 45.0,
        "Fundamental_Raw": 50.0,
        "Sentiment_Raw": 50.0,
        "Context_Raw": 40.0,
        "RS_Score": 2.0,
        "RS_Composite": 1.1,
        "Vol_Surge_Ratio": 1.2,
        "Dist_52W_High_Pct": 5.0,
        "Extension_Pct": 3.0,
        "Is_Coiling": False,
        "Black_Swan_Flag": 0,
        "Regime_Multiplier": 1.0
    }
]

df = pd.DataFrame(sample_data)
print("BEFORE apply_advanced_scoring:")
print(f"Score: {df['Score'].iloc[0]}")
print(f"Technical_Raw: {df['Technical_Raw'].iloc[0]}")
print(f"Context_Raw: {df['Context_Raw'].iloc[0]}")

scored_df = apply_advanced_scoring(df, DEFAULT_SCORING_CONFIG)

print("\nAFTER apply_advanced_scoring:")
print(f"Score: {scored_df['Score'].iloc[0]}")
print(f"Score_Pre_Regime: {scored_df.get('Score_Pre_Regime', 'N/A').iloc[0] if 'Score_Pre_Regime' in scored_df else 'N/A'}")
print(f"Technical_Score: {scored_df.get('Technical_Score', 'N/A').iloc[0] if 'Technical_Score' in scored_df else 'N/A'}")
print(f"Context_Score: {scored_df.get('Context_Score', 'N/A').iloc[0] if 'Context_Score' in scored_df else 'N/A'}")
print(f"Quality_Gate_Pass: {scored_df.get('Quality_Gate_Pass', 'N/A').iloc[0] if 'Quality_Gate_Pass' in scored_df else 'N/A'}")
print(f"Verdict: {scored_df.get('Verdict', 'N/A').iloc[0] if 'Verdict' in scored_df else 'N/A'}")

print("\n\nAll columns in output:")
for col in scored_df.columns:
    print(f"  {col}: {scored_df[col].iloc[0]}")

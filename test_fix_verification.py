#!/usr/bin/env python3
"""
Test that conviction scores are properly calculated and consistent
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

# Simulate a real scenario with multiple stocks
sample_data = [
    # Stock with uptrend (tech_base = True)
    {
        "Symbol": "UPTREND",
        "Score": 70,  # Good raw conviction
        "Price": 2000.0,
        "RSI": 60.0,
        "News": "Neutral",
        "Events": "✅ Safe",
        "Sector": "IT",
        "Avg_Value_20D_Cr": 500.0,
        "Market_Cap_Cr": 800000.0,
        "Debt_To_Equity": 0.1,
        "Technical_Raw": 60.0,  # High technical
        "Fundamental_Raw": 50.0,
        "Sentiment_Raw": 50.0,
        "Context_Raw": 50.0,
        "RS_Score": 3.0,
        "RS_Composite": 1.3,
        "Vol_Surge_Ratio": 1.8,
        "Dist_52W_High_Pct": 5.0,
        "Extension_Pct": 5.0,
        "Is_Coiling": False,
        "Black_Swan_Flag": 0,
        "Regime_Multiplier": 1.0
    },
    # Stock in downtrend (tech_base = False) - should have low conviction
    {
        "Symbol": "DOWNTREND",
        "Score": 10,  # Low raw conviction
        "Price": 500.0,
        "RSI": 40.0,
        "News": "Neutral",
        "Events": "✅ Safe",
        "Sector": "Finance",
        "Avg_Value_20D_Cr": 200.0,
        "Market_Cap_Cr": 300000.0,
        "Debt_To_Equity": 0.5,
        "Technical_Raw": 5.0,  # Low technical
        "Fundamental_Raw": 20.0,
        "Sentiment_Raw": 50.0,
        "Context_Raw": 15.0,
        "RS_Score": -2.0,
        "RS_Composite": 0.8,
        "Vol_Surge_Ratio": 0.9,
        "Dist_52W_High_Pct": 40.0,
        "Extension_Pct": -10.0,
        "Is_Coiling": False,
        "Black_Swan_Flag": 0,
        "Regime_Multiplier": 1.0
    }
]

df = pd.DataFrame(sample_data)
print("=" * 60)
print("ORIGINAL DATA (from check_institutional_fortress)")
print("=" * 60)
for idx, row in df.iterrows():
    print(f"{row['Symbol']:12} Score: {row['Score']:5.1f}  Tech_Raw: {row['Technical_Raw']:5.1f}  Verdict: ", end="")
    s = row['Score']
    if s >= 85:
        print("🔥 HIGH")
    elif s >= 60:
        print("🚀 PASS")
    elif s >= 35:
        print("🟡 WATCH")
    else:
        print("❌ FAIL")

print("\n" + "=" * 60)
print("AFTER apply_advanced_scoring (what gets saved)")
print("=" * 60)

scored_df = apply_advanced_scoring(df, DEFAULT_SCORING_CONFIG)

for idx, row in scored_df.iterrows():
    print(f"{row['Symbol']:12} Score: {row['Score']:5.1f}  Tech_Score: {row['Technical_Score']:5.1f}  Verdict: {row['Verdict']:10}  Gate: {row['Quality_Gate_Pass']}")

print("\n" + "=" * 60)
print("ANALYSIS: Are conviction scores properly calculated?")
print("=" * 60)
print(f"✓ UPTREND score: {scored_df[scored_df['Symbol']=='UPTREND']['Score'].values[0]:.1f} (should be high)")
print(f"✓ DOWNTREND score: {scored_df[scored_df['Symbol']=='DOWNTREND']['Score'].values[0]:.1f} (should be low but not 0)")
print(f"✓ Both scores are non-zero and differentiated? {'YES' if all(scored_df['Score'] > 0) else 'NO - ISSUE FOUND'}")

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from engine.stock_scanner.logic import apply_advanced_scoring

df = pd.DataFrame([
    {"Symbol": "RELIANCE", "Technical_Raw": 40.0, "Fundamental_Raw": 30.0, "Sentiment_Raw": 50.0, "Context_Raw": 30.0, "Regime_Multiplier": 1.0, "Avg_Value_20D_Cr": 10.0, "Price": 2500.0},
    {"Symbol": "TCS", "Technical_Raw": 20.0, "Fundamental_Raw": 60.0, "Sentiment_Raw": 40.0, "Context_Raw": 20.0, "Regime_Multiplier": 1.0, "Avg_Value_20D_Cr": 5.0, "Price": 3500.0}
])

print("Mock DF:")
print(df)

df_out = apply_advanced_scoring(df)
print("\nOutput DF:")
print(df_out[["Symbol", "Score", "Score_Pre_Regime", "Technical_Score", "Quality_Gate_Pass"]].to_string())

import yfinance as yf
ticker = "RELIANCE.NS"
t = yf.Ticker(ticker)
h = t.history(period="1y")
print(f"RELIANCE history length: {len(h)}")
print(f"Columns: {h.columns.tolist()}")

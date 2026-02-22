print("Loading commodities.logic ...")
import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf

from fortress_config import COMMODITIES_TICKERS

logger = logging.getLogger(__name__)

COMMODITY_MAP = {
    "Gold": {"global": "GC=F", "local": "GOLDBEES.NS", "unit_adj": 1.0},
    "Silver": {"global": "SI=F", "local": "SILVERBEES.NS", "unit_adj": 1.0},
    "Crude": {"global": "CL=F", "local": "OIL.NS", "unit_adj": 1.0},
    "Copper": {"global": "HG=F", "local": "HINDCOPPER.NS", "unit_adj": 1.0},
}


def _retry(operation, module_name: str, retries: int = 3, base_delay: float = 1.0):
    last_error = None
    for attempt in range(retries):
        try:
            return operation()
        except Exception as e:
            last_error = e
            logger.error(f"{module_name} error: {e}")
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"{module_name} failed after retries") from last_error


def fetch_price_series(symbol: str, period: str = "6mo") -> pd.DataFrame:
    data = _retry(lambda: yf.download(symbol, period=period, progress=False, auto_adjust=True), f"commodities_{symbol}")
    if data.empty:
        return pd.DataFrame()
    out = data[["Close", "High", "Low"]].copy()
    out.columns = ["close", "high", "low"]
    return out.dropna()


def compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    if df.empty:
        return 0.0
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.rolling(window).mean().iloc[-1]) if len(tr) > window else float(tr.mean())


def build_commodities_frame(selection: str | None = None) -> pd.DataFrame:
    usdinr = fetch_price_series("INR=X", period="1mo")
    fx = float(usdinr["close"].iloc[-1]) if not usdinr.empty else 84.0

    rows = []
    for name, cfg in COMMODITY_MAP.items():
        if selection and name != selection:
            continue
        global_df = fetch_price_series(cfg["global"])
        local_df = fetch_price_series(cfg["local"])
        if global_df.empty or local_df.empty:
            continue

        g_price = float(global_df["close"].iloc[-1])
        l_price = float(local_df["close"].iloc[-1])
        parity = g_price * fx * cfg["unit_adj"]
        spread_pct = ((l_price - parity) / (parity + 1e-9)) * 100
        atr = compute_atr(local_df)
        vol = local_df["close"].pct_change().std() * np.sqrt(252) * 100
        score = np.clip(50 + (abs(spread_pct) * 8) - (vol * 0.3), 0, 100)

        rows.append(
            {
                "Commodity": name,
                "Global Symbol": cfg["global"],
                "Local Symbol": cfg["local"],
                "Price": l_price,
                "ATR": atr,
                "Spread %": spread_pct,
                "USDINR": fx,
                "Score": score,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.fillna(0).sort_values("Score", ascending=False)

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from fortress_config import INDEX_BENCHMARKS, MF_SCHEMES

try:
    from mftool import Mftool
except Exception:  # optional dependency at runtime
    Mftool = None

logger = logging.getLogger(__name__)


def _retry(operation, module_name: str, retries: int = 3, base_delay: float = 1.0):
    last_error = None
    for attempt in range(retries):
        try:
            return operation()
        except Exception as e:
            last_error = e
            logger.error(f"{module_name} error: {e}")
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"{module_name} failed after {retries} retries") from last_error


@st.cache_data(ttl=3600)
def fetch_nav_history(scheme_code: str) -> pd.DataFrame:
    def _load():
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna(subset=["date", "nav"]).sort_values("date").set_index("date")
        df = df.asfreq("B").ffill()
        df["ret"] = df["nav"].pct_change()
        return df.dropna()

    return _retry(_load, f"mf_nav_{scheme_code}")


def _extract_name_and_nav(mf_client, scheme_code: str):
    details = mf_client.get_scheme_quote(scheme_code)
    if not isinstance(details, dict):
        return f"Scheme {scheme_code}", np.nan
    scheme_name = details.get("scheme_name") or details.get("fund_house", "") or f"Scheme {scheme_code}"
    nav_value = pd.to_numeric(details.get("nav"), errors="coerce")
    return scheme_name, nav_value


@st.cache_data(ttl=1800)
def fetch_mf_snapshot(scheme_codes: list[str]) -> pd.DataFrame:
    mf_client = Mftool() if Mftool else None
    rows = []

    for code in scheme_codes:
        try:
            history = fetch_nav_history(str(code))
            if history.empty:
                continue

            nav = float(history["nav"].iloc[-1])
            scheme_name = f"Scheme {code}"
            if mf_client:
                scheme_name, nav_from_mftool = _retry(lambda: _extract_name_and_nav(mf_client, str(code)), "mf_quote")
                # Defensive delay to avoid AMFI (mfapi.in) rate-limit blocks
                time.sleep(0.5)
                if pd.notna(nav_from_mftool):
                    nav = float(nav_from_mftool)

            # --- Dynamic Benchmark Realignment ---
            bench_ticker = INDEX_BENCHMARKS.get("Nifty 50", "^NSEI")
            nm_lower = scheme_name.lower()
            if "small cap" in nm_lower or "smallcap" in nm_lower:
                bench_ticker = INDEX_BENCHMARKS.get("Nifty Smallcap 250", "^CNXSC")
            elif "mid cap" in nm_lower or "midcap" in nm_lower:
                bench_ticker = INDEX_BENCHMARKS.get("Nifty Midcap 150", "^NSMIDCP")

            bench_ret_series = fetch_benchmark_returns(bench_ticker)
            ret = history["ret"].dropna()

            # Alpha/Beta Calculation
            alpha, beta = np.nan, np.nan
            if not bench_ret_series.empty and not ret.empty:
                combined = pd.concat([ret, bench_ret_series], axis=1, join="inner").dropna()
                if len(combined) > 60:
                    cov_matrix = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])
                    cov = cov_matrix[0, 1]
                    var = np.var(combined.iloc[:, 1])
                    beta = cov / var if var != 0 else 1.0

                    # Annualized Alpha
                    # Alpha = (Rp - Rf) - Beta * (Rm - Rf)
                    # Simplified: Alpha = Rp - Beta * Rm (assuming Rf is negligible or handled in excess returns)
                    # Using simple mean difference annualized:
                    rp = combined.iloc[:, 0].mean() * 252
                    rm = combined.iloc[:, 1].mean() * 252
                    alpha = (rp - 0.06) - beta * (rm - 0.06) # Assuming Rf=6%
                    alpha = alpha * 100 # Convert to percentage

            ret_1y = history["nav"].pct_change(252).iloc[-1] * 100 if len(history) > 252 else np.nan
            ret_3y = ((history["nav"].iloc[-1] / history["nav"].iloc[-min(756, len(history))]) ** (252 / min(756, len(history))) - 1) * 100 if len(history) > 252 else np.nan
            ret_5y = ((history["nav"].iloc[-1] / history["nav"].iloc[-min(1260, len(history))]) ** (252 / min(1260, len(history))) - 1) * 100 if len(history) > 252 else np.nan

            vol = ret.std() * np.sqrt(252) * 100
            downside = ret[ret < 0].std() * np.sqrt(252) * 100
            sharpe = ((ret.mean() * 252) - 0.06) / (ret.std() * np.sqrt(252) + 1e-9)
            sortino = ((ret.mean() * 252) - 0.06) / ((ret[ret < 0].std() * np.sqrt(252)) + 1e-9)
            rolling_std = ret.rolling(21).std().mean() * np.sqrt(252) * 100

            rows.append(
                {
                    "Scheme Code": str(code),
                    "Scheme": scheme_name,
                    "NAV": nav,
                    "1Y Return": ret_1y,
                    "3Y Return": ret_3y,
                    "5Y Return": ret_5y,
                    "Volatility": vol,
                    "Downside Deviation": downside,
                    "Sharpe": sharpe,
                    "Sortino": sortino,
                    "Rolling Std": rolling_std,
                    "Alpha": alpha,
                    "Beta": beta,
                    "Benchmark": bench_ticker
                }
            )
            # Defensive sleep to respect mfapi.in rate limits
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"mf_lab error: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.fillna(0)
    vol_penalty = (df["Volatility"] + df["Downside Deviation"] + df["Rolling Std"]).clip(lower=0)
    raw = (df["Sharpe"] + df["Sortino"] - (vol_penalty / 100)).fillna(0)
    min_raw, max_raw = raw.min(), raw.max()
    if max_raw == min_raw:
        df["Consistency Score"] = 50.0
    else:
        df["Consistency Score"] = ((raw - min_raw) / (max_raw - min_raw) * 100).clip(0, 100)

    return df.sort_values("Consistency Score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800)
def fetch_benchmark_returns(ticker: str = INDEX_BENCHMARKS.get("Nifty 50", "^NSEI")) -> pd.Series:
    def _load():
        data = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            return pd.Series(dtype=float)
        close_col = data["Close"] if "Close" in data else data.iloc[:, 0]
        return close_col.pct_change().dropna()

    return _retry(_load, "mf_benchmark")


def backtest_vs_benchmark(scheme_code: str) -> pd.DataFrame:
    fund = fetch_nav_history(scheme_code)
    bench_ret = fetch_benchmark_returns()
    if fund.empty or bench_ret.empty:
        return pd.DataFrame()

    merged = pd.DataFrame({"fund": fund["ret"], "bench": bench_ret}).dropna()
    if merged.empty:
        return pd.DataFrame()

    out = (1 + merged).cumprod()
    out.columns = ["Fund", "Nifty 50"]
    return out


def classify_category(name: str) -> str:
    nm = (name or "").lower()
    if any(k in nm for k in ["liquid", "bond", "gilt", "debt", "duration"]):
        return "Debt"
    if any(k in nm for k in ["hybrid", "balanced", "asset allocation"]):
        return "Hybrid"
    return "Equity"


DEFAULT_SCHEMES = MF_SCHEMES

import pandas as pd
import pandas_ta as ta
import datetime
import yfinance as yf
from fortress_config import SECTOR_MAP, NIFTY_SYMBOL
from datetime import datetime

_BENCHMARK_CACHE = {}

DEFAULT_SCORING_CONFIG = {
    "weights": {"technical": 0.50, "fundamental": 0.25, "sentiment": 0.15, "context": 0.10},
    "liquidity_cr_min": 8.0,
    "market_cap_cr_min": 1500.0,
    "price_min": 80.0,
    "max_debt_to_equity": 2.0,
    "min_interest_coverage": 3.0,
    "enable_regime": True,
}


def _safe_float(value, default=0.0):
    try:
        val = float(value)
        return default if pd.isna(val) else val
    except:
        return default


def _get_benchmark_series(symbol):
    cached = _BENCHMARK_CACHE.get(symbol)
    if cached is not None and len(cached) > 0:
        return cached

    try:
        bench = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = bench.columns.get_level_values(0)
        close = bench.get("Close", pd.Series(dtype=float)).dropna()
        _BENCHMARK_CACHE[symbol] = close
        return close
    except:
        return pd.Series(dtype=float)


def _return_ratio(series, periods):
    if len(series) <= periods:
        return 1.0
    base = _safe_float(series.iloc[-(periods + 1)], default=0.0)
    now = _safe_float(series.iloc[-1], default=0.0)
    if base <= 0:
        return 1.0
    return now / base


def _safe_info_float(info, key, default=0.0):
    if not isinstance(info, dict):
        return default
    return _safe_float(info.get(key, default), default=default)


def _extract_sector(symbol):
    return SECTOR_MAP.get(symbol, "General")


def _normalize_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(50.0, index=series.index)
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        clipped = numeric.clip(lower=low, upper=high)
    else:
        clipped = numeric
    min_v = clipped.min()
    max_v = clipped.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(50.0, index=series.index)
    return ((clipped - min_v) / (max_v - min_v) * 100).fillna(50.0)


def detect_market_regime():
    try:
        nifty = yf.download(NIFTY_SYMBOL, period="1y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty_close = nifty["Close"].dropna()
        nifty_now = _safe_float(nifty_close.iloc[-1]) if not nifty_close.empty else 0.0
        nifty_ema200 = _safe_float(ta.ema(nifty_close, 200).iloc[-1]) if len(nifty_close) >= 200 else 0.0

        vix = yf.download("^INDIAVIX", period="6mo", interval="1d", progress=False, auto_adjust=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix_value = _safe_float(vix.get("Close", pd.Series(dtype=float)).dropna().iloc[-1], default=20.0)

        if nifty_now > nifty_ema200 and vix_value < 18:
            return {"Regime": "Bull", "Regime_Multiplier": 1.15, "VIX": vix_value}
        if nifty_now < nifty_ema200 or vix_value > 25:
            return {"Regime": "Bear", "Regime_Multiplier": 0.80, "VIX": vix_value}
        return {"Regime": "Neutral", "Regime_Multiplier": 1.00, "VIX": vix_value}
    except:
        return {"Regime": "Neutral", "Regime_Multiplier": 1.00, "VIX": 20.0}


def apply_advanced_scoring(df, scoring_config=None):
    if df is None or df.empty:
        return df

    cfg = DEFAULT_SCORING_CONFIG.copy()
    if scoring_config:
        cfg.update({k: v for k, v in scoring_config.items() if k != "weights"})
        if "weights" in scoring_config:
            cfg["weights"] = scoring_config["weights"]

    df = df.copy()

    # Normalize category sub-scores within scan universe
    df["Technical_Score"] = _normalize_series(df.get("Technical_Raw", 0)).round(2)
    df["Fundamental_Score"] = _normalize_series(df.get("Fundamental_Raw", 0)).round(2)
    df["Sentiment_Score"] = _normalize_series(df.get("Sentiment_Raw", 0)).round(2)
    df["Context_Score"] = _normalize_series(df.get("Context_Raw", 0)).round(2)

    # RS ranking and top quartile bonus
    df["RS_Rank"] = pd.to_numeric(df.get("RS_Composite", 0), errors="coerce").rank(method="average", pct=True) * 100
    rs_gate = (pd.to_numeric(df.get("RS_Composite", 0), errors="coerce") > 1.0) | (df["RS_Rank"] >= 75)
    df.loc[rs_gate.fillna(False), "Context_Score"] = (df.loc[rs_gate.fillna(False), "Context_Score"] + 20).clip(upper=100)

    # Regime handling
    regime = detect_market_regime() if cfg.get("enable_regime", True) else {"Regime": "Neutral", "Regime_Multiplier": 1.0, "VIX": 20.0}
    df["Regime"] = regime["Regime"]
    df["Regime_Multiplier"] = regime["Regime_Multiplier"]
    df["India_VIX"] = round(regime["VIX"], 2)

    if regime["Regime"] == "Bull":
        df["Technical_Score"] = (df["Technical_Score"] * 1.10).clip(upper=100)
        df["Context_Score"] = (df["Context_Score"] * 1.15).clip(upper=100)
    elif regime["Regime"] == "Bear":
        df["Technical_Score"] = (df["Technical_Score"] * 0.90).clip(lower=0)
        df["Fundamental_Score"] = (df["Fundamental_Score"] * 1.10).clip(upper=100)

    w = cfg["weights"]
    df["Score_Pre_Regime"] = (
        df["Technical_Score"] * w["technical"]
        + df["Fundamental_Score"] * w["fundamental"]
        + df["Sentiment_Score"] * w["sentiment"]
        + df["Context_Score"] * w["context"]
    )
    df["Score"] = (df["Score_Pre_Regime"] * df["Regime_Multiplier"]).clip(lower=0, upper=100).round(2)

    # Hard quality gates / avoid-list penalties
    df["Quality_Gate_Failures"] = ""
    gates = [
        (pd.to_numeric(df.get("Avg_Value_20D_Cr", 0), errors="coerce") <= cfg["liquidity_cr_min"], f"Liquidity<{cfg['liquidity_cr_min']}Cr"),
        (pd.to_numeric(df.get("Market_Cap_Cr", 0), errors="coerce") <= cfg["market_cap_cr_min"], f"MCap<{cfg['market_cap_cr_min']}Cr"),
        (pd.to_numeric(df.get("Price", 0), errors="coerce") <= cfg["price_min"], f"Price<{cfg['price_min']}"),
    ]

    debt = pd.to_numeric(df.get("Debt_To_Equity", 0), errors="coerce")
    icr = pd.to_numeric(df.get("Interest_Coverage", 0), errors="coerce")
    bad_balance_sheet = (debt >= cfg["max_debt_to_equity"]) & (icr <= cfg["min_interest_coverage"])
    gates.append((bad_balance_sheet.fillna(False), "WeakBalanceSheet"))
    gates.append((df.get("Negative_Earnings_Surprise", False) == True, "NegEarningsSurprise"))

    for cond, label in gates:
        idx = cond.fillna(False)
        df.loc[idx, "Quality_Gate_Failures"] = df.loc[idx, "Quality_Gate_Failures"].apply(lambda x: f"{x}|{label}" if x else label)

    fail_mask = df["Quality_Gate_Failures"].str.len() > 0
    df.loc[fail_mask, "Score"] = (df.loc[fail_mask, "Score"] - 60).clip(lower=0)

    avoid_mask = (df.get("Black_Swan_Flag", 0).astype(float) > 0) | (df.get("News") == "🚨 BLACK SWAN")
    df.loc[avoid_mask.fillna(False), "Score"] = (df.loc[avoid_mask.fillna(False), "Score"] - 50).clip(lower=0)
    df["Avoid_Flag"] = avoid_mask.fillna(False)

    # Keep verdict semantics backward-compatible
    df["Verdict"] = df["Score"].apply(lambda x: "🔥 HIGH" if x >= 85 else "🚀 PASS" if x >= 60 else "🟡 WATCH")
    df.loc[fail_mask, "Verdict"] = "❌ FAIL"
    df.loc[df["Avoid_Flag"], "Verdict"] = "🚨 AVOID"
    return df

def check_institutional_fortress(ticker, data, ticker_obj, portfolio_value, risk_per_trade):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data)<210: return None

        close, high, low, open_price = data["Close"], data["High"], data["Low"], data["Open"]
        volume = data.get("Volume", pd.Series(0, index=data.index, dtype=float)).fillna(0)

        ema200 = _safe_float(ta.ema(close,200).iloc[-1])
        ema50 = _safe_float(ta.ema(close,50).iloc[-1])
        rsi = _safe_float(ta.rsi(close,14).iloc[-1])
        atr = _safe_float(ta.atr(high,low,close,14).iloc[-1])
        atr100 = _safe_float(ta.atr(high, low, close, 100).iloc[-1])
        st_df = ta.supertrend(high,low,close,10,3)
        trend_col = [c for c in st_df.columns if c.startswith("SUPERTd")][0]
        trend_dir = int(_safe_float(st_df[trend_col].iloc[-1]))
        price = _safe_float(close.iloc[-1])
        prev_close = _safe_float(close.iloc[-2])
        curr_open = _safe_float(open_price.iloc[-1])
        curr_low = _safe_float(low.iloc[-1])
        current_volume = _safe_float(volume.iloc[-1])
        avg_volume_20 = _safe_float(volume.tail(20).mean())
        vol_surge_ratio = (current_volume / avg_volume_20) if avg_volume_20 > 0 else 0.0
        vol_surge = vol_surge_ratio > 1.5

        weekly_close = close.resample("W-FRI").last().dropna() if isinstance(close.index, pd.DatetimeIndex) else pd.Series(dtype=float)
        weekly_ema30 = _safe_float(ta.ema(weekly_close, 30).iloc[-1]) if len(weekly_close) >= 30 else 0.0

        tech_base = price>ema200 and trend_dir==1
        mtf_aligned = price > weekly_ema30 if weekly_ema30 > 0 else False

        sl_distance = atr*1.5
        sl_price = round(price-sl_distance,2)
        target_10d = round(price + atr*1.8,2)
        risk_amount = portfolio_value*risk_per_trade
        pos_size = int(risk_amount / sl_distance) if sl_distance>0 else 0

        conviction = 0
        score_mod = 0
        news_sentiment = "Neutral"
        event_status = "✅ Safe"
        black_swan_flag = 0

        # --- Resilience & Gap Logic ---
        # War/News Resilience
        drop = prev_close - curr_low
        resilience_label = "✅ Safe"
        if drop > (2.0 * atr):
            if price > ema200:
                resilience_label = "🛡️ HOLD (Shakeout)"
            else:
                resilience_label = "💀 FAIL (Breakdown)"
                score_mod -= 40 # Automatic penalty

        # Gap Integrity
        gap_integrity = "N/A"
        if curr_open < prev_close:
            gap_size = prev_close - curr_open
            # "Integral" if Open > EMA200 AND gap < 1.5 ATR
            if curr_open > ema200 and gap_size < (1.5 * atr):
                gap_integrity = "✅ Integral"
            else:
                gap_integrity = "⚠️ Gap Risk"

        try:
            news = ticker_obj.news or []
            titles = " ".join(n.get("title","").lower() for n in news[:5])
            if any(k in titles for k in ["fraud","investigation","default","bankruptcy","scam","legal"]):
                news_sentiment = "🚨 BLACK SWAN"
                score_mod -= 40
                black_swan_flag = 1
        except: pass
        try:
            cal = ticker_obj.calendar
            if isinstance(cal,pd.DataFrame) and not cal.empty:
                next_date = pd.to_datetime(cal.iloc[0,0]).date()
                days_to = (next_date - datetime.now().date()).days
                if 0<=days_to<=7:
                    event_status = f"🚨 EARNINGS ({next_date.strftime('%d-%b')})"
                    score_mod -= 20
        except: pass

        analyst_count = target_high = target_low = target_median = target_mean = 0
        market_cap_cr = debt_to_equity = interest_coverage = 0.0
        earnings_ts = None
        earnings_surprise = 0.0
        negative_earnings_surprise = False
        try:
            info = ticker_obj.info or {}
            analyst_count = info.get("numberOfAnalystOpinions",0)
            target_high = info.get("targetHighPrice",0)
            target_low = info.get("targetLowPrice",0)
            target_median = info.get("targetMedianPrice",0)
            target_mean = info.get("targetMeanPrice",0)
            market_cap_cr = _safe_info_float(info, "marketCap", 0.0) / 1e7
            debt_to_equity = _safe_info_float(info, "debtToEquity", 0.0)
            interest_coverage = _safe_info_float(info, "interestCoverage", 0.0)
        except: pass

        try:
            earnings = ticker_obj.earnings_dates
            if isinstance(earnings, pd.DataFrame) and not earnings.empty:
                latest = earnings.sort_index(ascending=False).iloc[0]
                earnings_ts = earnings.sort_index(ascending=False).index[0]
                earnings_surprise = _safe_float(latest.get("Surprise(%)", 0.0), default=0.0)
                negative_earnings_surprise = earnings_surprise < 0
        except:
            pass

        if tech_base:
            conviction += 60
            if 48<=rsi<=62: conviction+=20
            elif 40<=rsi<=72: conviction+=10
            conviction += score_mod

        # Relative Strength vs Nifty 50
        benchmark_close = _get_benchmark_series(NIFTY_SYMBOL)
        rs_score = 0.0
        try:
            stock_ret_30d = ((price / _safe_float(close.iloc[-31], default=price)) - 1) * 100 if len(close) > 30 else 0.0
            nifty_ret_30d = 0.0
            if len(benchmark_close) > 30:
                bench_now = _safe_float(benchmark_close.iloc[-1])
                bench_30 = _safe_float(benchmark_close.iloc[-31], default=bench_now)
                nifty_ret_30d = ((bench_now / bench_30) - 1) * 100 if bench_30 > 0 else 0.0
            rs_score = stock_ret_30d - nifty_ret_30d
        except:
            rs_score = 0.0

        if rs_score > 0:
            conviction += 15

        # Multi-horizon RS
        benchmark_close = _get_benchmark_series(NIFTY_SYMBOL)
        rs_3m = _return_ratio(close, 63) / max(_return_ratio(benchmark_close, 63), 1e-6)
        rs_6m = _return_ratio(close, 126) / max(_return_ratio(benchmark_close, 126), 1e-6)
        rs_12m = _return_ratio(close, 252) / max(_return_ratio(benchmark_close, 252), 1e-6)
        rs_composite = (rs_3m * 0.5) + (rs_6m * 0.3) + (rs_12m * 0.2)

        # Volume confirmation
        breakout = False
        if len(close) > 20:
            breakout_level = _safe_float(high.iloc[-21:-1].max(), default=price)
            breakout = price > breakout_level
        if vol_surge:
            conviction += 10
        if breakout and current_volume < avg_volume_20:
            conviction -= 10

        # Volatility contraction (VCP-like)
        is_coiling = atr > 0 and atr100 > 0 and atr < (atr100 * 0.8)
        if is_coiling:
            conviction += 10

        # Mean reversion / over-extension guard
        extension_pct = ((price - ema50) / ema50) * 100 if ema50 > 0 else 0.0
        extension_ema200_pct = ((price - ema200) / ema200) * 100 if ema200 > 0 else 0.0
        overextended = extension_pct > 15
        if overextended:
            conviction -= 20
        if extension_ema200_pct > 40:
            conviction -= 20

        dispersion_pct = ((target_high-target_low)/price)*100 if price>0 else 0
        dispersion_alert = "⚠️ High Dispersion" if dispersion_pct>30 else "✅"
        if dispersion_pct>30: conviction -= 10

        # Sub-score raw components
        technical_raw = 0.0
        technical_raw += 35 if tech_base else 0
        if 45 <= rsi <= 65:
            technical_raw += 15
        elif (40 <= rsi < 45) or (65 < rsi <= 72):
            technical_raw += 8
        technical_raw += 10 if vol_surge_ratio > 1.8 else 0
        technical_raw += 8 if is_coiling else 0
        technical_raw -= 20 if extension_ema200_pct > 40 else 0
        technical_raw = max(0, technical_raw)

        fundamental_raw = 30.0
        if analyst_count > 0:
            upside_pct = ((_safe_float(target_mean, price) - price) / price) * 100 if price > 0 else 0
            fundamental_raw += min(max(upside_pct, -20), 25)
        fundamental_raw += 10 if market_cap_cr > 1500 else 0
        if dispersion_pct > 25:
            fundamental_raw *= 0.7

        sentiment_raw = 50.0
        if news_sentiment == "🚨 BLACK SWAN":
            sentiment_raw -= 50
        half_life_days = 5.0
        decay = 1.0
        if earnings_ts is not None:
            days_ago = max((datetime.now().date() - earnings_ts.date()).days, 0)
            decay = 0.5 ** (days_ago / half_life_days)
        sentiment_raw += 10 * decay
        sentiment_raw -= 15 if negative_earnings_surprise else 0

        context_raw = 30.0
        context_raw += 20 if mtf_aligned else 0
        context_raw += 20 if rs_composite > 1.0 else 0
        ret_6m = (_return_ratio(close, 126) - 1) * 100
        vol_adj_mom = ret_6m / atr if atr > 0 else 0
        context_raw += min(max(vol_adj_mom, -10), 20)

        conviction = max(0,min(100,conviction))
        verdict = "🔥 HIGH" if conviction>=85 and mtf_aligned else "🚀 PASS" if conviction>=60 else "🟡 WATCH" if tech_base else "❌ FAIL"
        if overextended:
            verdict = "⚠️ OVEREXTENDED"

        # Backtest returns (7, 30, 60, 90 days)
        current_date = close.index[-1]
        returns = {}
        for days in [7, 30, 60, 90]:
            try:
                target_date = current_date - pd.Timedelta(days=days)
                # Find nearest index
                idx = close.index.get_indexer([target_date], method='nearest')[0]
                past_price = float(close.iloc[idx])
                pct_change = ((price - past_price) / past_price) * 100
                returns[f"Ret_{days}D"] = pct_change
            except:
                returns[f"Ret_{days}D"] = 0.0

        # --- Velocity & Strategy ---
        ret_7d = returns.get("Ret_7D", 0.0)
        ret_30d = returns.get("Ret_30D", 0.0)
        velocity = ret_7d - ret_30d

        strategy = "Neutral"
        if price > ema50 and 55 <= rsi <= 70:
            strategy = "Momentum Pick"
        elif price > ema200 and dispersion_pct <= 30:
            strategy = "Long-Term Pick"

        buy_zone_high = price + (0.5 * atr)
        buy_zone = f"₹{price:.2f} - ₹{buy_zone_high:.2f}"

        steam_left = target_10d - price
        rsi_vel_factor = rsi / 50.0
        days_to_target = 0
        if rsi_vel_factor > 0 and atr > 0:
             days_to_target = steam_left / (atr * rsi_vel_factor)

        return {
            "Symbol": ticker,
            "Verdict": verdict,
            "Score": conviction,
            "Price": round(price,2),
            "RSI": round(rsi,1),
            "News": news_sentiment,
            "Events": event_status,
            "Sector": SECTOR_MAP.get(ticker,"General"),
            "Position_Qty": pos_size,
            "Stop_Loss": sl_price,
            "Target_10D": target_10d,
            "Analysts": analyst_count,
            "Tgt_High": target_high,
            "Tgt_Median": target_median,
            "Tgt_Low": target_low,
            "Tgt_Mean": target_mean,
            "Dispersion_Alert": dispersion_alert,
            "Ret_30D": returns.get("Ret_30D"),
            "Ret_60D": returns.get("Ret_60D"),
            "Ret_90D": returns.get("Ret_90D"),
            # New Metrics
            "Ret_7D": ret_7d,
            "Velocity": velocity,
            "Strategy": strategy,
            "Buy_Zone": buy_zone,
            "Steam_Left": steam_left,
            "Days_To_Target": days_to_target,
            "Resilience": resilience_label,
            "Gap_Integrity": gap_integrity,
            "Above_EMA200": price > ema200,
            "RS_Score": round(rs_score, 2),
            "Vol_Surge_Ratio": round(vol_surge_ratio, 2),
            "Extension_Pct": round(extension_pct, 2),
            "Is_Coiling": is_coiling,
            "Avg_Volume_20D": round(avg_volume_20, 0),
            "Avg_Value_20D_Cr": round((avg_volume_20 * price) / 1e7, 2),
            "Market_Cap_Cr": round(market_cap_cr, 2),
            "Debt_To_Equity": round(debt_to_equity, 2),
            "Interest_Coverage": round(interest_coverage, 2),
            "Negative_Earnings_Surprise": bool(negative_earnings_surprise),
            "Black_Swan_Flag": black_swan_flag,
            "RS_3M": round(rs_3m, 3),
            "RS_6M": round(rs_6m, 3),
            "RS_12M": round(rs_12m, 3),
            "RS_Composite": round(rs_composite, 3),
            "Vol_Adj_Mom": round(vol_adj_mom, 2),
            "EMA200_Extension_Pct": round(extension_ema200_pct, 2),
            "Technical_Raw": round(technical_raw, 2),
            "Fundamental_Raw": round(fundamental_raw, 2),
            "Sentiment_Raw": round(sentiment_raw, 2),
            "Context_Raw": round(context_raw, 2)
        }
    except: return None

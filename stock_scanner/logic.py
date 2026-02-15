import pandas as pd
import pandas_ta as ta
import datetime
import yfinance as yf
from fortress_config import SECTOR_MAP, NIFTY_SYMBOL
from datetime import datetime

_BENCHMARK_CACHE = {}


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
        try:
            info = ticker_obj.info or {}
            analyst_count = info.get("numberOfAnalystOpinions",0)
            target_high = info.get("targetHighPrice",0)
            target_low = info.get("targetLowPrice",0)
            target_median = info.get("targetMedianPrice",0)
            target_mean = info.get("targetMeanPrice",0)
        except: pass

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
        overextended = extension_pct > 15
        if overextended:
            conviction -= 20

        dispersion_pct = ((target_high-target_low)/price)*100 if price>0 else 0
        dispersion_alert = "⚠️ High Dispersion" if dispersion_pct>30 else "✅"
        if dispersion_pct>30: conviction -= 10

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
            "Is_Coiling": is_coiling
        }
    except: return None

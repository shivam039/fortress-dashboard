import pandas as pd
import pandas_ta as ta
import datetime
from fortress_config import SECTOR_MAP
from datetime import datetime

def check_institutional_fortress(ticker, data, ticker_obj, portfolio_value, risk_per_trade):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data)<210: return None

        close, high, low, open_price = data["Close"], data["High"], data["Low"], data["Open"]

        ema200 = ta.ema(close,200).iloc[-1]
        ema50 = ta.ema(close,50).iloc[-1]
        rsi = ta.rsi(close,14).iloc[-1]
        atr = ta.atr(high,low,close,14).iloc[-1]
        st_df = ta.supertrend(high,low,close,10,3)
        trend_col = [c for c in st_df.columns if c.startswith("SUPERTd")][0]
        trend_dir = int(st_df[trend_col].iloc[-1])
        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        curr_open = float(open_price.iloc[-1])
        curr_low = float(low.iloc[-1])

        tech_base = price>ema200 and trend_dir==1

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

        dispersion_pct = ((target_high-target_low)/price)*100 if price>0 else 0
        dispersion_alert = "⚠️ High Dispersion" if dispersion_pct>30 else "✅"
        if dispersion_pct>30: conviction -= 10

        conviction = max(0,min(100,conviction))
        verdict = "🔥 HIGH" if conviction>=85 else "🚀 PASS" if conviction>=60 else "🟡 WATCH" if tech_base else "❌ FAIL"

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
            "Above_EMA200": price > ema200
        }
    except: return None

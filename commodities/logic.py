import yfinance as yf
import pandas as pd
import logging
from fortress_config import COMMODITY_TICKERS, COMMODITY_CONSTANTS

logger = logging.getLogger(__name__)

def fetch_market_data():
    """
    Fetches live data for commodities and currency.
    Returns a dictionary of latest prices.
    """
    tickers = []
    # Add Currency
    tickers.append("INR=X")

    # Add Commodities
    for name, cfg in COMMODITY_TICKERS.items():
        tickers.append(cfg['global'])
        tickers.append(cfg['local'])

    # Deduplicate
    tickers = list(set(tickers))

    try:
        # Fetch all in one go for efficiency
        # Using period='5d' to ensure we have last close if market is closed today
        # group_by='ticker' ensures we get a structure we can parse easily even with 1 ticker
        data = yf.download(tickers, period="5d", progress=False, group_by='ticker')

        latest_prices = {}

        # Iterate and extract Close
        for t in tickers:
            try:
                # yfinance returns a DataFrame for each ticker if group_by='ticker'
                # or a MultiIndex if not. With group_by='ticker', data[t] is a DF.
                if len(tickers) == 1:
                     df = data
                else:
                     df = data[t]

                # Get last valid Close
                last_price = df['Close'].ffill().iloc[-1]
                # If it's a series (single ticker), it's a scalar. If not, handle.
                if isinstance(last_price, pd.Series):
                    last_price = last_price.iloc[0] # Should not happen with specific ticker selection

                latest_prices[t] = float(last_price)
            except Exception as e:
                # logger.warning(f"Could not extract data for {t}: {e}")
                pass

        return latest_prices
    except Exception as e:
        logger.error(f"Error fetching commodity data: {e}")
        return {}

def analyze_arbitrage():
    """
    Performs the full arbitrage analysis.
    Returns a DataFrame containing the opportunities.
    """
    prices = fetch_market_data()
    if not prices:
        return pd.DataFrame()

    usd_inr = prices.get("INR=X", 84.0) # Default fallback if missing

    results = []

    for name, cfg in COMMODITY_TICKERS.items():
        glob_sym = cfg['global']
        loc_sym = cfg['local']

        p_glob = prices.get(glob_sym, 0.0)
        p_loc = prices.get(loc_sym, 0.0)

        # Skip if missing data
        if pd.isna(p_glob) or pd.isna(p_loc) or p_glob == 0 or p_loc == 0:
            continue

        # --- PARITY CALCULATION ---
        # Formula: (Global * USDINR * Conversion * (1 + Duty)) + Warehousing

        base_parity_inr = (p_glob * usd_inr * cfg['conversion_factor'])

        # Duty Cost
        duty_amt = base_parity_inr * cfg['import_duty']

        # Warehousing Cost (Monthly) - Applied to the base value
        # User: "fixed percentage model... of the contract value"
        # We calculate cost for 1 month holding
        warehousing_amt = base_parity_inr * COMMODITY_CONSTANTS["WAREHOUSING_COST_PCT_MONTHLY"]

        # Total Parity Price
        parity_price = base_parity_inr + duty_amt + warehousing_amt

        # Spread (Local - Parity)
        spread = p_loc - parity_price

        # Annualized Yield % (Theoretical)
        # Yield = (Spread / Parity) * 100
        # If we assume this gap closes or is captured.
        # User asked for "Gross Arbitrage Yield %".
        arb_yield = (spread / parity_price) * 100

        # --- ACTION LOGIC ---
        action = "WAIT"
        threshold = COMMODITY_CONSTANTS.get("ARB_YIELD_THRESHOLD", 10.0)

        # Note: Yield here is flat %. Annualized would depend on duration.
        # User asked for "Gross Arbitrage Yield %" without explicit time factor in formula prompt,
        # but threshold is >10% annualized.
        # If this is a spot-future arb, the spread is for the duration.
        # Assuming near month (approx 30 days).
        # Annualized = Flat * (365/30) ~ Flat * 12.
        # Let's display Flat Yield and maybe check threshold against Annualized?
        # "Validation: Only enable execution... if Gross Arbitrage Yield % exceeds threshold... (e.g. >10% annualized)"
        # I'll calculate annualized for the check.

        annualized_yield = arb_yield * 12 # approx

        trade_type = ""

        if annualized_yield > threshold:
            action = "🔥 SHORT MCX / LONG GLOBAL"
            trade_type = "SELL" # We Sell the expensive Local
        elif annualized_yield < -threshold:
             action = "❄️ LONG MCX / SHORT GLOBAL"
             trade_type = "BUY" # We Buy the cheap Local

        results.append({
            "Commodity": name,
            "Symbol (Local)": loc_sym,
            "Symbol (Global)": glob_sym,
            "Global Price ($)": p_glob,
            "USD/INR": usd_inr,
            "Parity Price (₹)": parity_price,
            "MCX Price (₹)": p_loc,
            "Spread (₹)": spread,
            "Yield (%)": arb_yield,
            "Ann. Yield (%)": annualized_yield,
            "Action": action,
            "Trade_Type": trade_type,
            "Duty Paid": duty_amt,
            "Warehousing": warehousing_amt
        })

    return pd.DataFrame(results)

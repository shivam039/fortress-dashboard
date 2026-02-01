import yfinance as yf
import pandas as pd
import logging
import sys
import os
import importlib

# Ensure root is in path for config import
# This handles the case where the script is run directly from the 'commodities' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import fortress_config
    COMMODITY_TICKERS = fortress_config.COMMODITY_TICKERS
    COMMODITY_SPECS = fortress_config.COMMODITY_SPECS
    COMMODITY_CONSTANTS = fortress_config.COMMODITY_CONSTANTS
except (ImportError, AttributeError):
    try:
        # Fallback to relative import if running as a package
        from .. import fortress_config
        COMMODITY_TICKERS = fortress_config.COMMODITY_TICKERS
        COMMODITY_SPECS = fortress_config.COMMODITY_SPECS
        COMMODITY_CONSTANTS = fortress_config.COMMODITY_CONSTANTS
    except (ImportError, ValueError):
        # Final fallback if both fail (e.g., config missing or bad path)
        print("Error importing fortress_config. Using empty defaults.")
        COMMODITY_TICKERS = []
        COMMODITY_SPECS = {}
        COMMODITY_CONSTANTS = {
            "WAREHOUSING_COST_PCT_MONTHLY": 0.001,
            "ARB_YIELD_THRESHOLD": 10.0,
            "DEFAULT_WINDOW": 20,
            "CORRELATION_THRESHOLD": 0.8,
            "VOLATILITY_LOOKBACK": 14
        }

logger = logging.getLogger(__name__)

def fetch_market_data():
    """
    Fetches live data for commodities and currency.
    Returns a dictionary of latest prices.
    """
    tickers = []
    # Add Currency
    tickers.append("INR=X")

    # Add Commodities from simple list
    tickers.extend(COMMODITY_TICKERS)

    # Add Commodities from specs if not in list (for local tickers which might not be in the global list)
    for name, cfg in COMMODITY_SPECS.items():
        if cfg.get('local') and cfg['local'] not in tickers:
            tickers.append(cfg['local'])
        # Ensure global from specs is also included if missing from list
        if cfg.get('global') and cfg['global'] not in tickers:
             tickers.append(cfg['global'])

    # Deduplicate and filter empty/None
    tickers = list(set([t for t in tickers if t]))

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

    # Iterate over COMMODITY_SPECS for arbitrage details
    for name, cfg in COMMODITY_SPECS.items():
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

def check_correlations(prices=None):
    """
    Checks for sector-wide impacts based on commodity prices.
    Returns a list of warnings/alerts.
    """
    if prices is None:
        prices = fetch_market_data()

    alerts = []

    # Crude Oil Impact
    # Crude Up -> Paints Down (Raw material), Aviation Down (Fuel)
    # Crude Down -> Paints Up, Aviation Up
    # Use COMMODITY_SPECS to lookup Crude ticker
    crude_sym = COMMODITY_SPECS.get("Crude Oil", {}).get("global")
    if crude_sym:
        # We need "Change" to determine direction. fetch_market_data only returns latest price.
        # We need historical data to compute change.
        try:
             # Quick fetch for trend
             hist = yf.Ticker(crude_sym).history(period="5d")
             if len(hist) >= 2:
                 pct_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100

                 if pct_change > 1.0:
                     alerts.append({
                         "Source": "Crude Oil",
                         "Change": f"Diff +{pct_change:.1f}%",
                         "Impact": "Negative",
                         "Sectors": "Paints (ASIANPAINT, BERGEPAINT), Aviation (INDIGO)",
                         "Thesis": "Rising input costs squeeze margins."
                     })
                 elif pct_change < -1.0:
                     alerts.append({
                         "Source": "Crude Oil",
                         "Change": f"Diff {pct_change:.1f}%",
                         "Impact": "Positive",
                         "Sectors": "Paints, Aviation, Tyres",
                         "Thesis": "Lower input costs boost margins."
                     })
        except: pass

    # Gold Impact
    # Gold Up -> Gold Loan Up (Collateral value), Jewelry mixed (Inventory gain vs Demand drop)
    gold_sym = COMMODITY_SPECS.get("Gold", {}).get("global")
    if gold_sym:
        try:
             hist = yf.Ticker(gold_sym).history(period="5d")
             if len(hist) >= 2:
                 pct_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100

                 if pct_change > 0.5:
                     alerts.append({
                         "Source": "Gold",
                         "Change": f"Diff +{pct_change:.1f}%",
                         "Impact": "Positive",
                         "Sectors": "Gold Loans (MUTHOOTFIN, MANAPPURAM)",
                         "Thesis": "Higher collateral value reduces LTV risk."
                     })
        except: pass

    return alerts

import math
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Black-Scholes Math
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

def calculate_greeks(S, K, T, r, sigma, option_type="CE"):
    """
    S: Spot Price
    K: Strike Price
    T: Time to Expiry (years)
    r: Risk-free rate (decimal)
    sigma: Volatility (decimal)
    option_type: "CE" or "PE"
    """
    if T <= 0 or sigma <= 0:
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    N_d1 = norm_cdf(d1)
    N_d2 = norm_cdf(d2)
    N_neg_d1 = norm_cdf(-d1)
    N_neg_d2 = norm_cdf(-d2)
    n_d1 = norm_pdf(d1)

    if option_type == "CE":
        delta = N_d1
        theta = (- (S * sigma * n_d1) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N_d2) / 365.0
    else:
        delta = N_d1 - 1
        theta = (- (S * sigma * n_d1) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * N_neg_d2) / 365.0

    gamma = n_d1 / (S * sigma * math.sqrt(T))
    vega = (S * math.sqrt(T) * n_d1) / 100.0 # Per 1% change in vol

    return {
        "Delta": round(delta, 3),
        "Gamma": round(gamma, 4),
        "Theta": round(theta, 3),
        "Vega": round(vega, 3)
    }

def fetch_option_chain(symbol, expiry_index=0):
    try:
        tkr = yf.Ticker(symbol)
        expirations = tkr.options
        if not expirations:
            return None, None, 0

        expiry_date_str = expirations[expiry_index] if len(expirations) > expiry_index else expirations[0]
        chain = tkr.option_chain(expiry_date_str)

        # Calculate T (Time to Expiry)
        expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
        now = datetime.now()
        T = (expiry_date - now).days / 365.0
        if T < 0.002: T = 0.002 # Min 1 day

        # Get Underlying Spot
        # Try fast info
        spot = 0
        try:
             spot = tkr.fast_info['last_price']
        except:
             hist = tkr.history(period='1d')
             if not hist.empty:
                 spot = hist['Close'].iloc[-1]

        return chain, T, spot
    except Exception as e:
        print(f"Error fetching options: {e}")
        return None, None, 0

def resolve_strategy_legs(template, symbol, expiry_idx=0):
    # Fetch chain
    chain, T, spot = fetch_option_chain(symbol, expiry_idx)
    if chain is None or spot == 0:
        return []

    r = 0.06 # Risk free rate
    legs = []

    strikes = sorted(list(set(chain.calls['strike'].tolist() + chain.puts['strike'].tolist())))
    if not strikes: return []

    atm_strike = min(strikes, key=lambda x: abs(x - spot))
    atm_idx = strikes.index(atm_strike)

    for leg in template:
        # Determine Option Type
        opt_type = leg['option'] # CE, PE, STOCK

        strike_logic = leg['strike']
        selected_strike = atm_strike

        iv = 0.2 # Default

        if opt_type == "STOCK":
            selected_strike = 0 # Spot
        else:
            # Logic
            if strike_logic == "ATM":
                selected_strike = atm_strike
            elif strike_logic == "ITM":
                # Call ITM = Lower, Put ITM = Higher
                step = 1
                if opt_type == "CE": idx = max(0, atm_idx - step)
                else: idx = min(len(strikes)-1, atm_idx + step)
                selected_strike = strikes[idx]
            elif strike_logic == "OTM":
                # Call OTM = Higher, Put OTM = Lower
                step = 1
                if opt_type == "CE": idx = min(len(strikes)-1, atm_idx + step)
                else: idx = max(0, atm_idx - step)
                selected_strike = strikes[idx]
            elif "OTM_20_Delta" in strike_logic:
                # Need to find delta.
                # Use IV from chain.
                target_delta = 0.20
                best_s = atm_strike
                min_diff = 999

                df = chain.calls if opt_type=="CE" else chain.puts

                for _, row in df.iterrows():
                    s = row['strike']
                    i_vol = 0.2
                    if 'impliedVolatility' in row:
                        val = row['impliedVolatility']
                        if val is not None and not np.isnan(val) and val > 0:
                            i_vol = val

                    greeks = calculate_greeks(spot, s, T, r, i_vol, opt_type)
                    d = abs(greeks['Delta'])
                    if abs(d - target_delta) < min_diff:
                        min_diff = abs(d - target_delta)
                        best_s = s
                selected_strike = best_s
            elif "Wings" in strike_logic:
                # Far OTM (e.g. 3 strikes)
                step = 4
                if opt_type == "CE": idx = min(len(strikes)-1, atm_idx + step)
                else: idx = max(0, atm_idx - step)
                selected_strike = strikes[idx]
            elif "Hedge" in strike_logic:
                 # 2 strikes OTM
                step = 2
                if opt_type == "CE": idx = min(len(strikes)-1, atm_idx + step)
                else: idx = max(0, atm_idx - step)
                selected_strike = strikes[idx]
            elif "SPOT" == strike_logic:
                pass

        # Get Price and Greeks
        price = spot
        contract_symbol = symbol
        greeks = {"Delta":1, "Gamma":0, "Theta":0, "Vega":0}

        if opt_type != "STOCK":
            df = chain.calls if opt_type=="CE" else chain.puts
            row = df[df['strike'] == selected_strike]
            if not row.empty:
                price = row.iloc[0]['lastPrice']
                contract_symbol = row.iloc[0]['contractSymbol']

                iv = 0.2
                if 'impliedVolatility' in row.columns:
                    val = row.iloc[0]['impliedVolatility']
                    if val is not None and not np.isnan(val) and val > 0:
                        iv = val

                greeks = calculate_greeks(spot, selected_strike, T, r, iv, opt_type)
            else:
                price = 0 # Should not happen if strike is from list

        legs.append({
            "leg_id": leg['leg'],
            "action": leg['type'], # BUY/SELL
            "type": opt_type,
            "strike": selected_strike,
            "qty_mult": leg['qty_mult'],
            "price": price,
            "contractSymbol": contract_symbol,
            "greeks": greeks,
            "iv": iv if opt_type != "STOCK" else 0
        })

    return legs, spot, T

def check_synthetic_future_arb(spot, chain, T, r=0.06):
    # Conversion Arb: Buy Stock, Sell Call, Buy Put
    # Cost = S + P - C
    # Profit = K - Cost

    # Find ATM Call and Put
    if chain is None: return None

    strikes = sorted(list(set(chain.calls['strike'].tolist() + chain.puts['strike'].tolist())))
    if not strikes: return None

    atm_strike = min(strikes, key=lambda x: abs(x - spot))

    call_row = chain.calls[chain.calls['strike'] == atm_strike]
    put_row = chain.puts[chain.puts['strike'] == atm_strike]

    if call_row.empty or put_row.empty: return None

    C = call_row.iloc[0]['lastPrice']
    P = put_row.iloc[0]['lastPrice']
    K = atm_strike
    S = spot

    # Cost to enter Conversion
    Cost = S + P - C

    Profit = K - Cost
    Yield_Abs = (Profit / Cost)
    Yield_Ann = Yield_Abs * (1/T) * 100

    return {
        "Strategy": "Conversion Reversal",
        "Strike": K,
        "Stock": S,
        "Call": C,
        "Put": P,
        "Cost": Cost,
        "Locked_Value": K,
        "Profit": Profit,
        "Yield_Ann": Yield_Ann
    }

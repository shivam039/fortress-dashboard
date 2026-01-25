import urllib.parse
import re
import json
import streamlit as st

# Mapping of NSE Symbols to Dhan Security IDs.
# As Dhan requires specific numerical IDs for Deep Links, this dictionary must be populated.
# If a symbol is not found here, the system falls back to a generic Search URL.
DHAN_SECURITY_IDS = {
    "RELIANCE": "1333",  # Example ID
    "HDFCBANK": "1330",
    "INFY": "1594",
    "TCS": "11536",
    # User to populate this from Dhan Scrip Master
}

# Mapping for MCX Commodities (Monthly expiry dependent - User to update)
MCX_SECURITY_IDS = {
    "GOLD": "Placeholder_ID",
    "CRUDEOIL": "Placeholder_ID",
    "SILVER": "Placeholder_ID",
    "NATURALGAS": "Placeholder_ID"
}

def clean_symbol_for_broker(symbol):
    """
    Removes the suffix from the ticker symbol.
    Example: RELIANCE.NS -> RELIANCE, GOLD.MC -> GOLD
    """
    if not symbol:
        return ""
    return symbol.replace(".NS", "").replace(".BO", "").replace(".MC", "")

def generate_zerodha_url(symbol, qty, transaction_type="BUY", api_key="fortress_v9"):
    """
    Generates a Kite Publisher URL for a predefined basket order.
    Supports NSE (default) and MCX if symbol ends with .MC
    """
    clean_sym = clean_symbol_for_broker(symbol)
    if not clean_sym:
        return None

    qty = int(qty) if qty > 0 else 1

    # Determine Exchange
    exchange = "MCX" if ".MC" in symbol else "NSE"

    # Construct the JSON payload for the basket
    basket_data = f'[{{"exchange":"{exchange}","tradingsymbol":"{clean_sym}","transaction_type":"{transaction_type}","quantity":{qty},"order_type":"MARKET"}}]'

    base_url = "https://kite.zerodha.com/connect/publish"
    encoded_basket = urllib.parse.quote(basket_data)

    return f"{base_url}?api_key={api_key}&symbols={encoded_basket}"

def generate_dhan_url(symbol, qty, price=0, transaction_type="BUY"):
    """
    Generates a Dhan Deep Link or Fallback Search Link.
    Supports NSE (default) and MCX if symbol ends with .MC
    """
    clean_sym = clean_symbol_for_broker(symbol)
    if not clean_sym:
        return None

    qty = int(qty) if qty > 0 else 1

    # Determine Exchange & Segment
    is_mcx = ".MC" in symbol
    exchange = "MCX" if is_mcx else "NSE"
    segment = "COMM" if is_mcx else "EQ"

    # Check ID Map
    sec_id = None
    if is_mcx:
        sec_id = MCX_SECURITY_IDS.get(clean_sym)
    else:
        sec_id = DHAN_SECURITY_IDS.get(clean_sym)

    if sec_id and sec_id != "Placeholder_ID":
        # Construct specific Order Window Deep Link (Web)
        base_url = "https://web.dhan.co/orders/new"
        params = {
            "exchange": exchange,
            "segment": segment,
            "securityId": sec_id,
            "transactionType": transaction_type,
            "quantity": qty,
            "orderType": "MARKET",
            "price": price
        }
        return f"{base_url}?{urllib.parse.urlencode(params)}"
    else:
        # Fallback: Dhan Search / Watchlist
        return f"https://web.dhan.co/watchlist?search={clean_sym}"

def convert_yahoo_to_zerodha(y_symbol):
    # Yahoo: NIFTY231026C01900000 or RELIANCE231026...
    # Regex to parse
    if not y_symbol: return ""
    match = re.match(r"([A-Z\^]+)(\d{2})(\d{2})(\d{2})([CP])(\d+)", y_symbol)
    if not match:
        return y_symbol # Fallback

    symbol, yy, mm, dd, cp, strike_str = match.groups()

    # Clean Symbol
    symbol = symbol.replace("^NSEI", "NIFTY").replace("^NSEBANK", "BANKNIFTY").replace(".NS", "")

    # Month Map for Weekly/Monthly
    m_int = int(mm)
    m_char = f"{m_int}" if m_int < 10 else "O" if m_int==10 else "N" if m_int==11 else "D"

    strike = int(int(strike_str) / 1000)

    cp_type = "CE" if cp == "C" else "PE"

    # Construct Weekly Format: SYMBOL YY M DD STRIKE TYPE
    z_symbol = f"{symbol}{yy}{m_char}{dd}{strike}{cp_type}"
    return z_symbol

def generate_basket_html(legs, broker="Zerodha"):
    # legs: list of dicts {contractSymbol, qty, action, type}
    # action: BUY/SELL
    # type: CE/PE/STOCK

    data = []
    for leg in legs:
        # Map Symbol
        y_sym = leg.get('contractSymbol', '')
        z_sym = convert_yahoo_to_zerodha(y_sym) if leg['type'] != "STOCK" else clean_symbol_for_broker(leg.get('contractSymbol'))

        txn = "BUY" if leg['action'] == "BUY" else "SELL"
        qty = int(leg.get('qty', 1))

        item = {
            "exchange": "NFO" if leg['type'] != "STOCK" else "NSE",
            "tradingsymbol": z_sym,
            "transaction_type": txn,
            "quantity": qty,
            "order_type": "MARKET",
            "product": "MIS"
        }
        data.append(item)

    json_data = json.dumps(data)
    safe_json = json_data.replace("'", "&apos;")

    if broker == "Zerodha":
        # Kite Publisher JS Injection
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://kite.trade/publisher.js?v=3"></script>
        </head>
        <body style="background-color: transparent; margin: 0; padding: 0;">
            <kite-button href="#"
                data-kite="fortress_v9"
                data-action="basket"
                data-data='{safe_json}'>
            </kite-button>
        </body>
        </html>
        """
    else:
        # Dhan Web Overlay / Deep Link Wrapper
        html = f"""
        <div style="display: flex; align-items: center; justify-content: center;">
            <a href="https://web.dhan.co/orders/basket" target="_blank" style="text-decoration: none;">
                <button style="
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: bold;
                    font-family: sans-serif;
                    font-size: 14px;">
                    🚀 Execute via Dhan Web
                </button>
            </a>
        </div>
        """

    st.components.v1.html(html, height=100)

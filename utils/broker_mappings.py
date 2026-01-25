import urllib.parse

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

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

def clean_symbol_for_broker(symbol):
    """
    Removes the .NS suffix from the ticker symbol.
    Example: RELIANCE.NS -> RELIANCE
    """
    if not symbol:
        return ""
    return symbol.replace(".NS", "").replace(".BO", "")

def generate_zerodha_url(symbol, qty, api_key="fortress_v9"):
    """
    Generates a Kite Publisher URL for a predefined basket order.
    """
    clean_sym = clean_symbol_for_broker(symbol)
    if not clean_sym:
        return None

    qty = int(qty) if qty > 0 else 1

    # Construct the JSON payload for the basket
    # Using specific formatting to ensure valid JSON string in URL
    basket_data = f'[{{"exchange":"NSE","tradingsymbol":"{clean_sym}","transaction_type":"BUY","quantity":{qty},"order_type":"MARKET"}}]'

    # Encode parameters
    params = {
        "api_key": api_key,
        "symbols": basket_data
    }

    # Kite Publisher uses 'symbols' parameter for the JSON basket
    # We construct the query manually to ensure the JSON brackets aren't over-encoded in a way that breaks Kite
    # However, urllib.parse.urlencode usually works fine.
    # Let's use the format provided by user reference:
    # symbols=[{...}]

    base_url = "https://kite.zerodha.com/connect/publish"
    encoded_basket = urllib.parse.quote(basket_data)

    return f"{base_url}?api_key={api_key}&symbols={encoded_basket}"

def generate_dhan_url(symbol, qty, price=0):
    """
    Generates a Dhan Deep Link or Fallback Search Link.
    """
    clean_sym = clean_symbol_for_broker(symbol)
    if not clean_sym:
        return None

    qty = int(qty) if qty > 0 else 1

    # Check if we have a mapped Security ID
    sec_id = DHAN_SECURITY_IDS.get(clean_sym)

    if sec_id:
        # Construct specific Order Window Deep Link (Web)
        # Note: This format assumes the standard Dhan Web intent structure.
        # If this changes, update this template.
        # Using a common web-order-window construction:
        base_url = "https://web.dhan.co/orders/new"
        params = {
            "exchange": "NSE",
            "segment": "EQ",
            "securityId": sec_id,
            "transactionType": "BUY",
            "quantity": qty,
            "orderType": "MARKET",
            "price": price  # Optional for LIMIT, ignored for MARKET usually
        }
        return f"{base_url}?{urllib.parse.urlencode(params)}"
    else:
        # Fallback: Dhan Search / Watchlist
        # This redirects the user to the web platform with the symbol search
        return f"https://web.dhan.co/watchlist?search={clean_sym}"

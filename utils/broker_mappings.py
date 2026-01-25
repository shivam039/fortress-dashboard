import urllib.parse
import re
import json

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

    if broker == "Zerodha":
        html = f"""
        <form action="https://kite.zerodha.com/connect/basket" method="post" target="_blank">
            <input type="hidden" name="api_key" value="fortress_v9" />
            <input type="hidden" name="data" value='{json_data}' />
            <button type="submit" style="background-color: #FF5722; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
                🚀 Execute via Zerodha Kite
            </button>
        </form>
        """
        return html
    else:
        # Dhan Fallback
        return f"""
        <div style="padding: 10px; background: #222; color: #fff; border-radius: 5px;">
            Dhan Basket API not configured. <br>
            <a href="https://web.dhan.co/orders/basket" target="_blank" style="color: #4CAF50;">Open Dhan Basket Builder</a>
        </div>
        """

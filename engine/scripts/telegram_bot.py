import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime

# Adjust path to import from engine root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fortress_config import TICKER_GROUPS
from utils.db import _read_df

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8586585011:AAHFal_zfoGEtjol86GMI49OKCSlCgvclMA")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "677141544")

def get_latest_scan_for_universe(universe: str):
    """Fetch the latest completed scan for a given universe."""
    query = """
        SELECT scan_id, timestamp 
        FROM scans 
        WHERE universe = :universe AND status = 'Completed' 
        ORDER BY timestamp DESC 
        LIMIT 1
    """
    scan_info = _read_df(query, {"universe": universe})
    if scan_info.empty:
        return None
    
    scan_id = scan_info.iloc[0]["scan_id"]
    timestamp = scan_info.iloc[0]["timestamp"]
    
    # Fetch details
    details_query = """
        SELECT raw_data 
        FROM scan_history_details 
        WHERE scan_id = :scan_id
    """
    details = _read_df(details_query, {"scan_id": int(scan_id)})
    
    if details.empty or "raw_data" not in details.columns:
        return None
        
    df = pd.json_normalize(details["raw_data"].apply(lambda x: x if isinstance(x, dict) else json.loads(x)))
    return {"df": df, "timestamp": timestamp}

def format_telegram_message(row):
    """Format a single stock pick using the optimized template."""
    ticker = row.get("Symbol", "UNKNOWN")
    buy_zone = row.get("Buy_Zone", "N/A")
    score_raw = row.get("Score", 0)
    score_out_of_10 = min(10.0, score_raw / 10.0)
    
    sector = row.get("Sector", "General")
    strategy = row.get("Strategy", "Momentum Pick")
    
    target_10d = row.get("Target_10D", "N/A")
    tgt_mean = row.get("Tgt_Mean", "N/A")
    
    # Fallback for Target 2
    if pd.isna(tgt_mean) or tgt_mean == 0 or tgt_mean == "N/A":
        # Estimate Target 2 as slightly higher than Target 1
        if isinstance(target_10d, (int, float)):
            target_2 = round(target_10d * 1.05, 2)
        else:
            target_2 = "N/A"
    else:
        target_2 = tgt_mean
        
    stop_loss = row.get("Stop_Loss", "N/A")
    
    msg = f"🚀 STOCK PICK: ${ticker}\n\n"
    msg += f"Action: BUY\n"
    msg += f"Entry Range: {buy_zone}\n\n"
    msg += f"Conviction Score: 🔥 {score_out_of_10:.1f}/10\n"
    msg += f"({sector} | {strategy})\n\n"
    msg += f"Targets & Risk:\n"
    msg += f"🎯 Target 1: {target_10d}\n"
    msg += f"🎯 Target 2: {target_2}\n"
    msg += f"🛑 Stop Loss: {stop_loss} (Closing basis)\n\n"
    msg += f"Timeframe: Swing (5-10 Days)\n"
    msg += f"Disclaimer: Not financial advice. Educational purposes only."
    
    return msg

def send_telegram_message(message: str):
    """Send message via Telegram API to multiple chat IDs."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Skipping Telegram send: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return False
        
    chat_ids = [cid.strip() for cid in TELEGRAM_CHAT_ID.split(",") if cid.strip()]
    success = True
    
    for chat_id in chat_ids:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Telegram message to {chat_id}: {e}")
            if 'response' in locals() and response is not None:
                 print(f"Response: {response.text}")
            success = False
            
    return success

def main():
    print(f"Starting Telegram Bot broadcaster at {datetime.now()}")
    
    indices_to_check = ["Nifty 50", "Nifty Next 50", "Nifty Midcap 150", "Nifty Smallcap 250", "Nifty Microcap 250"]
    
    for index_name in indices_to_check:
        if index_name not in TICKER_GROUPS:
            continue
            
        print(f"Checking {index_name}...")
        scan_data = get_latest_scan_for_universe(index_name)
        
        if not scan_data:
            print(f"  No scan data found for {index_name}.")
            continue
            
        df = scan_data["df"]
        if df.empty or "Score" not in df.columns:
            print(f"  No valid score data found for {index_name}.")
            continue
            
        # Filter top picks (Score >= 60)
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0)
        top_picks = df[df["Score"] >= 60].sort_values("Score", ascending=False).head(5)
        
        if top_picks.empty:
            print(f"  No picks with Score >= 60 for {index_name}.")
            continue
            
        print(f"  Found {len(top_picks)} top picks for {index_name}.")
        
        # Add a header for the index
        header_msg = f"📊 <b>Top Picks for {index_name}</b>\n<i>Scan timestamp: {scan_data['timestamp']}</i>"
        send_telegram_message(header_msg)
        
        # Send each pick
        for _, row in top_picks.iterrows():
            msg = format_telegram_message(row)
            success = send_telegram_message(msg)
            if success:
                print(f"  -> Sent {row.get('Symbol')}")
            else:
                print(f"  -> Failed to send {row.get('Symbol')}")
                
    print("Done.")

if __name__ == "__main__":
    main()

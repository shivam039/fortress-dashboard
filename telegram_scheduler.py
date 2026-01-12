import time
import requests
import schedule
import yfinance as yf
import pandas as pd
from datetime import datetime
from fortress_config import TICKER_GROUPS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, BOT_PORTFOLIO_VALUE, BOT_RISK_PER_TRADE
from fortress_logic import check_institutional_fortress

def send_telegram_message(message):
    """Sends a message to the Telegram chat."""
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("Telegram credentials not set. Message not sent.")
        print(f"Message content: {message}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("Message sent successfully.")
        else:
            print(f"Failed to send message: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def run_bot_scan():
    """Runs the scan for high conviction trades and sends alerts."""
    print(f"Starting scheduled scan at {datetime.now()}")

    # Flatten the list of tickers from all groups
    all_tickers = []
    for group_tickers in TICKER_GROUPS.values():
        all_tickers.extend(group_tickers)

    # Remove duplicates
    all_tickers = list(set(all_tickers))
    print(f"Scanning {len(all_tickers)} tickers...")

    high_conviction_trades = []

    for i, ticker in enumerate(all_tickers):
        try:
            # Basic progress logging
            if i % 10 == 0:
                print(f"Processed {i}/{len(all_tickers)}...")

            tkr = yf.Ticker(ticker)
            hist = yf.download(ticker, period="2y", progress=False)

            if not hist.empty:
                res = check_institutional_fortress(
                    ticker,
                    hist,
                    tkr,
                    BOT_PORTFOLIO_VALUE,
                    BOT_RISK_PER_TRADE
                )

                if res and res["Verdict"] == "🔥 HIGH":
                    high_conviction_trades.append(res)

            # Avoid hitting rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"Error scanning {ticker}: {e}")

    if high_conviction_trades:
        print(f"Found {len(high_conviction_trades)} high conviction trades.")
        for trade in high_conviction_trades:
            # Format the message
            # "current price , target price, SL, HOLDING period"
            msg = (
                f"🚨 *HIGH CONVICTION ALERT* 🚨\n\n"
                f"📌 *Symbol:* `{trade['Symbol']}`\n"
                f"💰 *Price:* ₹{trade['Price']}\n"
                f"🎯 *Target (10D):* ₹{trade['Target_10D']}\n"
                f"🛑 *Stop Loss:* ₹{trade['Stop_Loss']}\n"
                f"⏳ *Holding Period:* 10 Days\n"
                f"📊 *Score:* {trade['Score']}/100\n"
                f"🏢 *Sector:* {trade['Sector']}"
            )
            send_telegram_message(msg)
    else:
        print("No high conviction trades found today.")

def job():
    # Only run on weekdays (Monday=0, Sunday=6)
    if datetime.today().weekday() < 5:
        run_bot_scan()
    else:
        print("Weekend - Skipping scan.")

# Schedule the job
# Note: For testing purposes in sandbox, we might want to run it immediately once,
# but the requirement is "alert to come at 9 am on all weekday"
schedule.every().day.at("09:00").do(job)

if __name__ == "__main__":
    print("Telegram Bot Scheduler Started...")
    print("Press Ctrl+C to exit.")

    # Check credentials on startup
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("WARNING: Telegram Bot Token is not set in fortress_config.py")

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

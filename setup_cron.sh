#!/bin/bash
echo "Setting up daily Telegram Bot schedule at 7:30 PM..."
crontab -l 2>/dev/null > /tmp/current_cron
echo "30 19 * * * cd /Users/shivamdixit/Desktop/fortress-dashboard && python3 engine/scripts/telegram_bot.py >> engine/telegram_bot.log 2>&1" >> /tmp/current_cron
crontab /tmp/current_cron
echo "Cron job scheduled successfully!"

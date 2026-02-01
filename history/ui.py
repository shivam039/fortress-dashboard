import streamlit as st
import pandas as pd
from datetime import datetime
from utils.db import fetch_timestamps, fetch_history_data
from utils.broker_mappings import generate_zerodha_url, generate_dhan_url

def render():
    st.header("📜 Master Scan History")

    # --- Sidebar Controls ---
    st.sidebar.subheader("History Filters")

    # Scan Type Selection
    scan_type = st.sidebar.selectbox(
        "Scan Type",
        ["STOCK", "MF", "OPTIONS", "COMMODITY"],
        index=0
    )

    # Date Filter
    # Get all timestamps for the type
    all_timestamps = fetch_timestamps(scan_type=scan_type)

    if not all_timestamps:
        st.info(f"No scan history found for {scan_type}.")
        return

    # Convert to datetime objects for date input filter
    # Assuming timestamp format "YYYY-MM-DD HH:MM:SS"
    try:
        dt_objs = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in all_timestamps]
        min_date = min(dt_objs).date()
        max_date = max(dt_objs).date()
    except:
        min_date = datetime.now().date()
        max_date = datetime.now().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date)
    )

    # Filter timestamps based on date range
    filtered_timestamps = []
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        for ts in all_timestamps:
            try:
                d = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").date()
                if start_d <= d <= end_d:
                    filtered_timestamps.append(ts)
            except:
                pass
    else:
        filtered_timestamps = all_timestamps

    # Scan Selection
    selected_timestamp = st.selectbox(
        "Select Scan",
        filtered_timestamps if filtered_timestamps else all_timestamps
    )

    if not selected_timestamp:
        return

    # Broker Choice for Links
    broker_choice = st.sidebar.selectbox("Execution Broker", ["Zerodha", "Dhan"])

    # --- Fetch Data ---
    with st.spinner(f"Loading {scan_type} scan from {selected_timestamp}..."):
        # We pass table_name="" because fetch_history_data logic relies on scan_type logic now
        # or falls back to standard tables if scan_type matches.
        df = fetch_history_data(table_name="", timestamp=selected_timestamp, scan_type=scan_type)

    if df.empty:
        st.warning("No details found for this scan.")
        return

    st.subheader(f"{scan_type} Results - {selected_timestamp}")
    st.caption(f"Found {len(df)} records.")

    # --- Display Logic with Links ---

    # Helper for links
    def get_link(row):
        symbol = row.get('Symbol') or row.get('symbol')
        if not symbol: return None

        # Quantity logic
        qty = row.get('Position_Qty', 1)
        if pd.isna(qty): qty = 1

        # Price for Dhan
        price = row.get('Price', 0)
        if pd.isna(price): price = 0

        # Transaction Type (for Commodities/Options)
        # Check 'action' or 'Action' or 'Trade_Type'
        txn_type = "BUY"
        if 'Trade_Type' in row and row['Trade_Type']:
            txn_type = row['Trade_Type']
        elif 'action' in row and row['action']:
            txn_type = row['action']

        if broker_choice == "Zerodha":
            return generate_zerodha_url(symbol, qty, transaction_type=txn_type)
        else:
            return generate_dhan_url(symbol, qty, price, transaction_type=txn_type)

    # Apply links if Symbol exists
    if 'Symbol' in df.columns or 'symbol' in df.columns:
        df['Execute'] = df.apply(get_link, axis=1)

        # Move Execute to front
        cols = list(df.columns)
        if 'Execute' in cols:
            cols.insert(0, cols.pop(cols.index('Execute')))
            df = df[cols]

    # Column Config
    column_config = {
        "Execute": st.column_config.LinkColumn("⚡ Trade"),
        "Price": st.column_config.NumberColumn(format="%.2f"),
        "Score": st.column_config.NumberColumn(format="%.1f"),
        "Fortress Score": st.column_config.NumberColumn(format="%.1f"),
    }

    # Hide some technical columns
    hide_cols = ['scan_id', 'id']
    show_cols = [c for c in df.columns if c not in hide_cols]

    st.dataframe(
        df[show_cols],
        use_container_width=True,
        column_config=column_config,
        hide_index=True
    )

    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV",
        csv,
        f"scan_history_{scan_type}_{selected_timestamp}.csv",
        "text/csv"
    )

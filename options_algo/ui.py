import streamlit as st
import pandas as pd
import json
from options_algo.templates import STRATEGY_TEMPLATES
from options_algo.logic import resolve_strategy_legs, check_synthetic_future_arb, fetch_option_chain
from utils.broker_mappings import generate_zerodha_url, generate_dhan_url, generate_basket_html
from utils.db import log_algo_trade, fetch_active_trades, close_all_trades

TICKER_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS"
}

def render(broker_choice="Zerodha"):
    # Defer Imports
    from options_algo.templates import STRATEGY_TEMPLATES
    from options_algo.logic import resolve_strategy_legs, check_synthetic_future_arb, fetch_option_chain
    from utils.broker_mappings import generate_zerodha_url, generate_dhan_url, generate_basket_html
    from utils.db import log_algo_trade, fetch_active_trades, close_all_trades

    st.header("🤖 Options Algo Terminal")
    st.caption("Institutional Strategies • Live Greeks • Basket Execution")

    try:
        # Sidebar / Top Controls
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            symbol_key = st.selectbox("Underlying", list(TICKER_MAP.keys()) + ["Other"])
            if symbol_key == "Other":
                symbol = st.text_input("Enter Symbol (Yahoo fmt)", "^NSEI")
            else:
                symbol = TICKER_MAP[symbol_key]

        with c2:
            strategy_name = st.selectbox("Select Strategy", list(STRATEGY_TEMPLATES.keys()))

        with c3:
            multiplier = st.number_input("Lot Multiplier", 1, 100, 1)

        # Main Analysis
        if st.button("🚀 Analyze Strategy", type="primary"):
            with st.spinner("Fetching Option Chain & Calculating Greeks..."):
                template = STRATEGY_TEMPLATES[strategy_name]
                # Resolve Legs
                legs, spot, T = resolve_strategy_legs(template, symbol)

                if not legs:
                    st.error("Failed to fetch option chain data. Check symbol or internet.")
                else:
                    st.session_state['algo_legs'] = legs
                    st.session_state['algo_spot'] = spot
                    st.session_state['algo_strategy'] = strategy_name
                    st.session_state['algo_symbol'] = symbol

        # Display Results if Available
        if 'algo_legs' in st.session_state:
            legs = st.session_state['algo_legs']
            spot = st.session_state['algo_spot']

            st.subheader(f"Strategy: {st.session_state['algo_strategy']} ({st.session_state['algo_symbol']})")
            st.metric("Spot Price", f"{spot:,.2f}")

            # Display Legs Table
            legs_df = pd.DataFrame(legs)

            # Format for Display
            display_df = legs_df.copy()
            display_df['Strike'] = display_df['strike']
            display_df['Type'] = display_df['type']
            display_df['Action'] = display_df['action']
            display_df['Price'] = display_df['price'].apply(lambda x: f"{x:.2f}")

            # Extract Greeks
            greeks_df = display_df['greeks'].apply(pd.Series)
            display_df = pd.concat([display_df, greeks_df], axis=1)

            # Select Columns
            cols = ['leg_id', 'Action', 'Type', 'Strike', 'Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'contractSymbol']
            st.dataframe(display_df[cols], use_container_width=True, hide_index=True)

            # Net Greeks
            net_delta = sum([g['Delta'] * (1 if l['action']=="BUY" else -1) * l['qty_mult'] * multiplier for l, g in zip(legs, display_df['greeks'])])
            net_theta = sum([g['Theta'] * (1 if l['action']=="BUY" else -1) * l['qty_mult'] * multiplier for l, g in zip(legs, display_df['greeks'])])

            g1, g2 = st.columns(2)
            g1.metric("Net Delta", f"{net_delta:.2f}")
            g2.metric("Net Theta (Daily Decay)", f"{net_theta:.2f}")

            # Execution
            st.markdown("---")
            if st.button("⚡ Generate Basket Link"):
                # Prepare legs for HTML generator & Logging
                legs_for_html = []
                for l in legs:
                    l_copy = l.copy()
                    l_copy['qty'] = l['qty_mult'] * multiplier
                    legs_for_html.append(l_copy)

                # Log Trade
                details = json.dumps(legs_for_html, default=str)
                log_algo_trade(st.session_state['algo_strategy'], st.session_state['algo_symbol'], "ENTRY", details)

                # Generate HTML
                html_code = generate_basket_html(legs_for_html, broker_choice)
                st.components.v1.html(html_code, height=100)

                st.success("Trade Logged! Click the button above to execute.")

        # Synthetic Future Arb
        st.markdown("---")
        st.subheader("🧪 Arbitrage Scanner")
        if st.button("Scan Conversion Arb"):
            chain, T, spot = fetch_option_chain(symbol)
            arb = check_synthetic_future_arb(spot, chain, T)
            if arb:
                st.write(arb)
                if arb['Yield_Ann'] > 10:
                    st.success(f"🔥 Arbitrage Opportunity! Yield: {arb['Yield_Ann']:.2f}%")
                else:
                    st.info(f"Yield {arb['Yield_Ann']:.2f}% below threshold.")
            else:
                st.warning("No Arb data found.")

        # Universal Kill Switch
        st.markdown("---")
        st.subheader("🚨 Risk Control")

        active_trades = fetch_active_trades()
        if not active_trades.empty:
            st.warning(f"{len(active_trades)} Active Strategies Detected.")
            st.dataframe(active_trades)

            if st.button("💀 PANIC EXIT (Close All)"):
                # Generate Reverse Basket
                all_reverse_legs = []
                for idx, row in active_trades.iterrows():
                    try:
                        legs_data = json.loads(row['details'])
                        for l in legs_data:
                            rev_l = l.copy()
                            rev_l['action'] = "SELL" if l.get('action') == "BUY" else "BUY"
                            all_reverse_legs.append(rev_l)
                    except:
                        pass

                if all_reverse_legs:
                    html_code = generate_basket_html(all_reverse_legs, broker_choice)
                    st.components.v1.html(html_code, height=100)
                    st.error("EXIT BASKET GENERATED. CLICK ABOVE TO EXECUTE.")
                else:
                    st.error("Could not parse active trades to generate exit.")

                close_all_trades()
                st.error("ALL TRADES MARKED CLOSED IN DB.")
        else:
            st.success("No Active Strategies.")
    except Exception as e:
        st.error(f"Options Algo Error: {e}")

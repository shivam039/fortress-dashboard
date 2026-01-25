import streamlit as st
import pandas as pd
from .logic import analyze_arbitrage, check_correlations
from utils.broker_mappings import generate_zerodha_url, generate_dhan_url

def render(broker_choice="Zerodha"):
    st.header("🌍 Commodities Intelligence Terminal")
    st.caption("Global-to-Local Arbitrage • Cash-and-Carry Analysis • Parity Audits")

    if st.button("🔄 Refresh Market Data", type="primary"):
        st.cache_data.clear() # Clear cache if logic uses it (logic currently doesn't use st.cache_data, but good practice)

    with st.spinner("Analyzing Global vs Local Markets..."):
        df = analyze_arbitrage()
        correlations = check_correlations()

    if df.empty:
        st.warning("No market data available. Please check internet connection or try again later.")
        return

    # --- TOP METRICS ---
    # Identify best opportunity
    df['Abs_Yield'] = df['Ann. Yield (%)'].abs()
    best_opp = df.loc[df['Abs_Yield'].idxmax()] if not df.empty else None

    m1, m2, m3, m4 = st.columns(4)
    if best_opp is not None:
        m1.metric("Top Opportunity", best_opp['Commodity'], f"{best_opp['Ann. Yield (%)']:.1f}% Ann.")
        m2.metric("Spread Gap", f"₹ {best_opp['Spread (₹)']:,.0f}", best_opp['Action'])

    avg_yield = df['Ann. Yield (%)'].mean()
    m3.metric("Avg Market Yield", f"{avg_yield:.1f}%")
    m4.metric("USD/INR", f"₹ {df['USD/INR'].iloc[0]:.2f}" if not df.empty else "N/A")

    # --- MAIN TABLE ---
    st.subheader("🌐 Global-to-Local Arbitrage Table")

    # Generate Execution Links
    def get_link(row):
        # Only generating link if Action is valid (Trade Type exists)
        if not row['Trade_Type']:
            return None

        qty = 1 # Default Lot
        symbol = row['Symbol (Local)']
        t_type = row['Trade_Type'] # BUY or SELL

        # Note: Shorting on Dhan/Zerodha via basket is standard.
        # But 'SELL' transaction type in basket works.

        if broker_choice == "Zerodha":
            return generate_zerodha_url(symbol, qty, transaction_type=t_type)
        else:
            return generate_dhan_url(symbol, qty, transaction_type=t_type)

    df['Execute'] = df.apply(get_link, axis=1)

    # Display Configuration
    display_cols = [
        'Commodity', 'Symbol (Local)', 'Global Price ($)', 'Parity Price (₹)',
        'MCX Price (₹)', 'Spread (₹)', 'Yield (%)', 'Ann. Yield (%)',
        'Action', 'Execute'
    ]

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        column_config={
            "Global Price ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Parity Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "MCX Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "Spread (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "Yield (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Ann. Yield (%)": st.column_config.NumberColumn(format="%.2f%%", help="Annualized Yield based on 1-month duration assumption"),
            "Execute": st.column_config.LinkColumn("⚡ Trade", display_text="Execute")
        },
        hide_index=True
    )

    # --- DETAILS SECTION ---
    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Yield & Cost Breakdown")
        st.info("Calculations include Import Duties, Conversion Factors, and Warehousing Costs.")
        # Show breakdown for selected row? Or just table.
        # Let's show the breakdown table (hidden cols from main)
        breakdown_cols = ['Commodity', 'Duty Paid', 'Warehousing']
        st.dataframe(
            df[breakdown_cols],
            use_container_width=True,
            column_config={
                "Duty Paid": st.column_config.NumberColumn(format="₹%.2f"),
                "Warehousing": st.column_config.NumberColumn(format="₹%.2f")
            },
            hide_index=True
        )

    with c2:
        st.subheader("📦 Cash-and-Carry Intelligence")
        st.caption("Spot vs Future Spread Analysis")
        # Placeholder logic for now as 'Next Month' tickers are hard to generalize dynamically
        st.warning("⚠️ Live Cash-and-Carry data requires specific expiry contract selection.")
        st.write("Current logic optimizes for Global-to-Local Parity gaps.")

        # Educational Text
        with st.expander("How to read this?"):
            st.markdown("""
            *   **Parity Price**: Theoretical fair value of MCX contract based on Global Spot + Duties + Costs.
            *   **Spread**: Difference between Actual MCX Price and Parity.
            *   **Positive Spread**: MCX is expensive -> **Short MCX**.
            *   **Negative Spread**: MCX is cheap -> **Long MCX**.
            """)

    # --- CORRELATION INTELLIGENCE ---
    st.markdown("---")
    st.subheader("🔗 Sector Correlation Intelligence")
    if correlations:
        col_c1, col_c2 = st.columns(2)
        for i, alert in enumerate(correlations):
            target_col = col_c1 if i % 2 == 0 else col_c2
            with target_col:
                color = "red" if alert['Impact'] == "Negative" else "green"
                icon = "📉" if alert['Impact'] == "Negative" else "📈"
                st.markdown(f"#### {icon} {alert['Source']} {alert['Change']}")
                st.write(f"**Impact:** :{color}[{alert['Impact']}] on **{alert['Sectors']}**")
                st.caption(f"Thesis: {alert['Thesis']}")
                st.markdown("---")
    else:
        st.info("No significant sector correlations detected based on commodity movements.")

import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import sys
import os
from utils.db import fetch_timestamps, fetch_history_data, fetch_symbol_history
from mf_lab.logic import apply_drift_status, get_category, calculate_fortress_score, generate_health_check_report

def render():
    st.subheader("🛡️ Fortress MF Pro: Consistency Lab")

    # ---------------- ADMIN TOOLS (SIDEBAR) ----------------
    with st.sidebar.expander("🛠️ Admin Tools"):
        st.info("Trigger the background scanner to audit the full universe (800+ funds). This runs independently.")
        if st.button("🚀 Run Background Discovery Scan"):
            try:
                # Run independent process
                script_path = os.path.join(os.getcwd(), "cron_mf_audit.py")
                if os.path.exists(script_path):
                    subprocess.Popen([sys.executable, script_path])
                    st.toast("✅ Background scan started! It will take ~15 mins.")
                else:
                    st.error(f"Script not found at {script_path}")
            except Exception as e:
                st.error(f"Failed to start scan: {e}")

        st.markdown("---")
        if st.button("📋 Generate Health Check"):
            with st.spinner("Analyzing Market Breadth & Sector Rotation..."):
                timestamps = fetch_timestamps("scan_mf")
                if timestamps:
                    current_ts = timestamps[0]
                    current_df = fetch_history_data("scan_mf", current_ts)

                    previous_df = None
                    if len(timestamps) > 1:
                        previous_df = fetch_history_data("scan_mf", timestamps[1])

                    report = generate_health_check_report(current_df, previous_df)
                    st.markdown("### 📋 Monthly Health Check Report")
                    st.code(report, language='markdown')
                else:
                    st.error("No scan data available. Please run a scan first.")

    # ---------------- ASSET CLASS SELECTOR (SIDEBAR) ----------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Asset Class")
    asset_class = st.sidebar.radio(
        "Select Universe:",
        ["Equity", "Debt"],
        index=0,
        help="Switch between Equity Mutual Funds and Debt Instruments."
    )

    # ---------------- VIEW RESULTS ----------------
    # 1. Fetch Latest Data
    timestamps = fetch_timestamps("scan_mf")

    if not timestamps:
        st.warning("No audit history found. Please run a scan from the Admin Tools.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_ts = st.selectbox("Select Audit Date", timestamps, index=0)

    # Fetch Data
    raw_df = fetch_history_data("scan_mf", selected_ts)

    if raw_df.empty:
        st.error("No data found for selected timestamp.")
        return

    # PATCH: Schema Evolution & Categorization
    if 'Category' not in raw_df.columns and 'Symbol' in raw_df.columns:
        raw_df['Category'] = raw_df['Symbol'].apply(get_category)
        st.caption("ℹ️ Legacy Data Detected: Categories were inferred from fund names.")

    # --- DRIFT ANALYSIS APPLICATION ---
    raw_df = apply_drift_status(raw_df)

    # 2. FILTER BY ASSET CLASS
    # Define Categories per Asset Class
    equity_cats = ["Large Cap", "Mid Cap", "Small Cap", "Flexi/Multi Cap", "Focused", "Value/Contra", "ELSS"]
    debt_cats = ["Liquid/Overnight", "Ultra Short/Low Duration", "Corporate Bond", "Gilt/Dynamic Bond"]

    if asset_class == "Equity":
        display_cats = equity_cats
        filtered_df = raw_df[raw_df['Category'].isin(equity_cats)]
    else:
        display_cats = debt_cats
        filtered_df = raw_df[raw_df['Category'].isin(debt_cats)]

    if filtered_df.empty:
        st.info(f"No funds found for {asset_class} in this scan.")
        # Optional: Show 'Other' if any categories didn't match standard lists?
        # For now, stick to strict taxonomy.
    else:
        # --- DRIFT WATCHLIST (Contextual to Asset Class) ---
        with st.expander(f"🚨 {asset_class} Strategy Watchdog", expanded=True):
            critical_drifts = filtered_df[filtered_df['Drift Status'] == "Critical"]
            moderate_drifts = filtered_df[filtered_df['Drift Status'] == "Moderate"]
            stable_count = len(filtered_df[filtered_df['Drift Status'] == "Stable"])
            total_funds = len(filtered_df)

            integrity_score = (stable_count / total_funds * 100) if total_funds > 0 else 0

            m1, m2 = st.columns([1, 3])
            m1.metric("Integrity Score", f"{integrity_score:.1f}%", f"{stable_count}/{total_funds} Stable")

            if not critical_drifts.empty:
                for _, row in critical_drifts.head(3).iterrows():
                     m2.error(f"🚨 **CRITICAL**: {row['Symbol']} — {row['Drift Message']}")
            elif not moderate_drifts.empty:
                m2.warning(f"⚠️ {len(moderate_drifts)} funds showing moderate drift.")
            else:
                m2.success("✅ No significant integrity breaches detected.")


        # 3. Category Leaderboards (Tabs)
        st.markdown(f"### 🏆 {asset_class} Leaderboards")

        # Create Tabs
        tabs = st.tabs(display_cats)

        final_display_df = pd.DataFrame() # Accumulate for charts

        for i, cat in enumerate(display_cats):
            with tabs[i]:
                cat_df = raw_df[raw_df['Category'] == cat].copy()

                if not cat_df.empty:
                    # Normalize Score 0-100 per category
                    cat_df = calculate_fortress_score(cat_df)

                    # Sort by Fortress Score
                    cat_df = cat_df.sort_values("Fortress Score", ascending=False)

                    # Columns Config
                    # Put Integrity First
                    base_cols = ["Integrity", "Symbol", "Fortress Score", "Alpha (True)", "Sortino", "Win Rate", "Upside Cap", "Downside Cap"]

                    # Add Debt specific cols if Debt
                    if asset_class == "Debt":
                        # Debt might care about Yield/CAGR more?
                        # Assuming 'Alpha' captures excess return over liquid bees, which is good.
                        pass

                    disp_cols = base_cols + ["Verdict"]
                    valid_cols = [c for c in disp_cols if c in cat_df.columns]

                    st.dataframe(
                        cat_df[valid_cols].style.background_gradient(subset=['Fortress Score'], cmap='RdYlGn'),
                        use_container_width=True,
                        height=500,
                        hide_index=True
                    )

                    final_display_df = pd.concat([final_display_df, cat_df])
                else:
                    st.info(f"No funds found in {cat} category.")

        # 4. Global Analysis (Charts) - Only if data exists
        if not final_display_df.empty:
            st.markdown("---")
            st.subheader("📊 Global Map")

            c1, c2 = st.columns(2)

            # Chart 1: Risk-Reward
            with c1:
                required_cols = ["Downside Cap", "Alpha (True)"]
                missing_mandatory = [c for c in required_cols if c not in final_display_df.columns]

                if not missing_mandatory:
                     # Clip Fortress Score for sizing
                    if "Fortress Score" in final_display_df.columns:
                        final_display_df["Size_Score"] = final_display_df["Fortress Score"].clip(lower=0.1).fillna(10.0)

                    fig = px.scatter(
                        final_display_df,
                        x="Downside Cap",
                        y="Alpha (True)",
                        size="Size_Score" if "Size_Score" in final_display_df.columns else None,
                        color="Category",
                        hover_name="Symbol",
                        title="Risk-Reward Map (Size = Score)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for Risk-Reward Map.")

            # Chart 2: Integrity Map
            with c2:
                drift_cols = ["Tracking Error", "Alpha (True)"]
                if "Tracking Error" not in final_display_df.columns and "te" in final_display_df.columns:
                    final_display_df["Tracking Error"] = final_display_df["te"]

                if "Tracking Error" in final_display_df.columns and "Alpha (True)" in final_display_df.columns:
                    fig_drift = px.scatter(
                        final_display_df,
                        x="Tracking Error",
                        y="Alpha (True)",
                        color="Drift Status",
                        hover_name="Symbol",
                        color_discrete_map={"Stable": "green", "Moderate": "orange", "Critical": "red", "Unknown": "gray"},
                        title="Integrity Map: Drift vs Alpha",
                        height=400
                    )
                    st.plotly_chart(fig_drift, use_container_width=True)
                else:
                    st.info("Insufficient data for Integrity Map.")

    # 5. Deep Dive Modal
    st.markdown("---")
    st.subheader("🕵️ Fund Deep Dive")

    # Select Fund from filtered list (or all?) - Better to let user search all
    all_symbols = sorted(raw_df['Symbol'].unique())
    selected_fund = st.selectbox("Search Any Fund History", all_symbols)

    if selected_fund:
        hist_df = fetch_symbol_history("scan_mf", selected_fund)
        if not hist_df.empty:
             st.markdown(f"#### Historical Performance: {selected_fund}")

             # Convert timestamp
             hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
             hist_df = hist_df.sort_values('timestamp')

             # Plotting
             # We want Score and Drift metric
             metric2 = "Tracking Error"
             if "Tracking Error" not in hist_df.columns:
                 if "te" in hist_df.columns: metric2 = "te"
                 elif "Beta" in hist_df.columns: metric2 = "Beta"
                 else: metric2 = None

             from plotly.subplots import make_subplots
             import plotly.graph_objects as go

             fig_hist = make_subplots(specs=[[{"secondary_y": True}]])

             # Score
             if "Score" in hist_df.columns:
                 fig_hist.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['Score'], name="Raw Score", mode='lines+markers'), secondary_y=False)

             # Metric 2
             if metric2 and metric2 in hist_df.columns:
                  fig_hist.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df[metric2], name=metric2, mode='lines', line=dict(dash='dot')), secondary_y=True)

             fig_hist.update_layout(title="Historical Trend Analysis", height=400)
             st.plotly_chart(fig_hist, use_container_width=True)

             with st.expander("View Raw History Data"):
                 st.dataframe(hist_df.sort_values('timestamp', ascending=False))
        else:
            st.info("No history found.")

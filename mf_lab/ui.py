import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import sys
import os
from utils.db import fetch_timestamps, fetch_history_data, fetch_symbol_history
# Removed fetch_fund_history, calculate_correlation_matrix, run_crisis_audit from logic.py
# as they should be imported from their source if needed, or left if they still exist in logic.py
# logic.py hasn't been deleted, but superseded. However, UI still uses some functions from it.
# We need to make sure logic.py doesn't conflict or we port those logic functions to UI or services.
# For now, logic.py exists, so we keep imports.
from mf_lab.logic import get_category, calculate_fortress_score, generate_health_check_report
from mf_lab.logic import calculate_correlation_matrix, run_crisis_audit

# We need a local apply_drift_status that respects DB values if present
def apply_drift_status_ui(df):
    """
    Applies drift calculation to a dataframe.
    Prioritizes existing DB columns 'integrity_label', 'drift_status', 'drift_message'.
    Falls back to logic calculation if missing (legacy support).
    """
    if df.empty: return df

    # Check if DB columns exist
    if 'drift_status' in df.columns and 'integrity_label' in df.columns:
        # Map DB columns to UI expected columns
        df['Drift Status'] = df['drift_status']
        df['Integrity'] = df['integrity_label']
        df['Drift Message'] = df['drift_message']
        return df

    # Fallback to legacy logic
    from mf_lab.logic import apply_drift_status
    return apply_drift_status(df)

def render_blender_suite(aggregated_selection):
    """
    Renders the Portfolio Blender & Backtester suite.
    aggregated_selection: dict of {scheme_code: scheme_name}
    """
    st.markdown("---")
    st.subheader("🧪 Fortress Portfolio Blender")

    if not aggregated_selection:
        st.info("Select funds from the tables above to activate the Quant-Suite Backtester.")
        return

    st.success(f"Loaded {len(aggregated_selection)} funds into the Blender.")

    with st.expander("🔬 Quant-Suite Analysis", expanded=True):
        if st.button("🚀 Run Crisis Audit & Overlap Check", use_container_width=True):
            with st.spinner("Fetching daily NAV history & crunching stress tests..."):

                # 1. Correlation Matrix
                corr_matrix, high_overlaps = calculate_correlation_matrix(aggregated_selection)

                # 2. Crisis Audit
                audit_results = run_crisis_audit(aggregated_selection)

                # RENDER RESULTS

                # --- A. Overlap Radar ---
                st.markdown("#### 📡 Risk Factor Radar (Correlation Heatmap)")
                if not corr_matrix.empty:
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="Return Correlation Matrix"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                    if high_overlaps:
                        st.warning(f"⚠️ **High Overlap Detected (>0.85)**: {len(high_overlaps)} pairs found.")
                        for pair, val in high_overlaps:
                            st.caption(f"- {pair}: {val:.2f}")
                    else:
                        st.success("✅ Good Diversification: No overlapping pairs > 0.85 found.")
                else:
                    st.error("Insufficient data for correlation analysis.")

                # --- B. Crisis Audit ---
                st.markdown("#### 💥 Fortress Crisis Audit")
                st.markdown("Stress testing selected funds against historical black swan events.")

                if audit_results:
                    audit_df = pd.DataFrame(audit_results)

                    # Ensure "Recovery" is shown if available
                    # Pivot for cleaner view
                    pivot_mdd = audit_df.pivot(index="Entity", columns="Window", values="Max Drawdown")

                    st.markdown("**Max Drawdown**")
                    st.dataframe(pivot_mdd.style.highlight_min(axis=1, color="pink"), use_container_width=True)

                    if "Recovery" in audit_df.columns:
                        pivot_rec = audit_df.pivot(index="Entity", columns="Window", values="Recovery")
                        st.markdown("**Recovery Time**")
                        st.dataframe(pivot_rec, use_container_width=True)

                    st.caption("Values represent **Max Drawdown** and **Recovery Time** during the event window.")
                else:
                    st.error("Audit returned no results.")

def render():
    st.subheader("🛡️ Fortress MF Pro: Consistency Lab")

    # ---------------- ADMIN TOOLS (SIDEBAR) ----------------
    with st.sidebar.expander("🛠️ Admin Tools"):
        st.info("Trigger the background scanner to audit the full universe (800+ funds). This runs independently.")

        # Scan Trigger
        if st.button("🚀 Run Background Discovery Scan"):
            try:
                # Run independent process
                script_path = os.path.join(os.getcwd(), "cron_mf_audit.py")
                if os.path.exists(script_path):
                    subprocess.Popen([sys.executable, script_path])
                    st.toast("✅ Background scan started! Check logs/status.")
                else:
                    st.error(f"Script not found at {script_path}")
            except Exception as e:
                st.error(f"Failed to start scan: {e}")

        st.markdown("---")

        # View Logs Button
        if st.button("📜 View Audit Logs"):
            try:
                if os.path.exists("audit.log"):
                    with open("audit.log", "r") as f:
                        logs = f.readlines()
                        # Show last 20 lines
                        st.text("".join(logs[-20:]))
                else:
                    st.info("No audit logs found.")
            except: pass

        st.markdown("---")

        # Health Check Report
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
    # Use UI wrapper that respects DB integrity
    raw_df = apply_drift_status_ui(raw_df)

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

        # Initialize Local Blender Basket for THIS render cycle
        # We rebuild the basket from the active selection state of all widgets
        # This prevents the 'infinite accumulation' bug by relying on current widget state
        local_blender_basket = {}

        current_tab_selection = [] # List of Symbols selected in the CURRENT interaction

        for i, cat in enumerate(display_cats):
            with tabs[i]:
                cat_df = raw_df[raw_df['Category'] == cat].copy()

                if not cat_df.empty:
                    # Normalize Score 0-100 per category
                    # Note: Cron already normalizes 'Score' but calls it 'Score'.
                    # UI expects 'Fortress Score'. If DB returned 'Fortress Score' (aliased from Score), we are good.
                    if 'Fortress Score' not in cat_df.columns:
                         cat_df['Fortress Score'] = cat_df['Score']

                    # Sort by Fortress Score
                    cat_df = cat_df.sort_values("Fortress Score", ascending=False)

                    # Columns Config
                    # Put Integrity First
                    base_cols = ["Integrity", "Symbol", "Fortress Score", "Alpha (True)", "Sortino", "Win Rate", "Upside Cap", "Downside Cap"]

                    disp_cols = base_cols + ["Verdict"]
                    valid_cols = [c for c in disp_cols if c in cat_df.columns]

                    # Unique Key for Dataframe
                    df_key = f"df_{asset_class}_{cat}"

                    # Render Dataframe with Selection
                    event = st.dataframe(
                        cat_df[valid_cols].style.background_gradient(subset=['Fortress Score'], cmap='RdYlGn'),
                        use_container_width=True,
                        height=500,
                        hide_index=True,
                        key=df_key,
                        on_select="rerun",
                        selection_mode="multi-row"
                    )

                    # Handle Selection
                    if len(event.selection['rows']) > 0:
                        selected_rows = cat_df.iloc[event.selection['rows']]

                        # Add to CURRENT filter context
                        current_tab_selection.extend(selected_rows['Symbol'].tolist())

                        # Add to Local BLENDER context
                        for idx, row in selected_rows.iterrows():
                            # Find scheme code from full row data if available (New Schema)
                            if 'Scheme Code' in row:
                                code = row['Scheme Code']
                            else:
                                # Fallback: Try to find scheme code from raw_df using Symbol (Legacy support)
                                # But row is a Series, so checking 'Scheme Code' key works.
                                # If it's NaN or missing, we have an issue.
                                code = None
                                # Try look up in raw_df
                                try:
                                    match = raw_df[raw_df['Symbol'] == row['Symbol']]
                                    if not match.empty and 'Scheme Code' in match.columns:
                                        code = match['Scheme Code'].values[0]
                                except: pass

                            # If we found a code, add to basket
                            if code:
                                local_blender_basket[code] = row['Symbol']
                            else:
                                # Logic.py requires a code. If missing, we can't blend.
                                # But maybe logic.py handles symbol if code is symbol? Unlikely.
                                # For legacy rows (scan_mf), scheme code might be missing.
                                pass

                    final_display_df = pd.concat([final_display_df, cat_df])
                else:
                    st.info(f"No funds found in {cat} category.")

        # 4. Global Analysis (Charts) - Only if data exists
        if not final_display_df.empty:
            st.markdown("---")
            st.subheader("📊 Global Map")

            # --- DYNAMIC FILTERING LOGIC ---
            # If current_tab_selection has items, filter the chart_df
            chart_df = final_display_df.copy()
            if current_tab_selection:
                chart_df = chart_df[chart_df['Symbol'].isin(current_tab_selection)]

            c1, c2 = st.columns(2)

            # Chart 1: Risk-Reward
            with c1:
                required_cols = ["Downside Cap", "Alpha (True)"]
                missing_mandatory = [c for c in required_cols if c not in chart_df.columns]

                if not missing_mandatory:
                     # Clip Fortress Score for sizing
                    if "Fortress Score" in chart_df.columns:
                        chart_df["Size_Score"] = chart_df["Fortress Score"].clip(lower=0.1).fillna(10.0)

                    fig = px.scatter(
                        chart_df,
                        x="Downside Cap",
                        y="Alpha (True)",
                        size="Size_Score" if "Size_Score" in chart_df.columns else None,
                        color="Category",
                        hover_name="Symbol",
                        title=f"Risk-Reward Map ({'Filtered' if current_tab_selection else 'All Funds'})",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for Risk-Reward Map.")

            # Chart 2: Integrity Map
            with c2:
                if "Tracking Error" not in chart_df.columns and "te" in chart_df.columns:
                    chart_df["Tracking Error"] = chart_df["te"]

                if "Tracking Error" in chart_df.columns and "Alpha (True)" in chart_df.columns:
                    fig_drift = px.scatter(
                        chart_df,
                        x="Tracking Error",
                        y="Alpha (True)",
                        color="Drift Status",
                        hover_name="Symbol",
                        color_discrete_map={"Stable": "green", "Moderate": "orange", "Critical": "red", "Unknown": "gray"},
                        title=f"Integrity Map ({'Filtered' if current_tab_selection else 'All Funds'})",
                        height=400
                    )
                    st.plotly_chart(fig_drift, use_container_width=True)
                else:
                    st.info("Insufficient data for Integrity Map.")

        # 5. RENDER BLENDER
        render_blender_suite(local_blender_basket)

    # 6. Deep Dive Modal
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

             fig_hist = go.Figure() # Replace make_subplots for simplicity if standard not available
             from plotly.subplots import make_subplots
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

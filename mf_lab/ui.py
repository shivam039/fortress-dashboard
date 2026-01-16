import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import sys
import os
from utils.db import fetch_timestamps, fetch_history_data, fetch_symbol_history
from mf_lab.logic import apply_drift_status

def render():
    st.subheader("🛡️ Fortress MF Pro: Consistency Lab")

    # ---------------- ADMIN TOOLS ----------------
    with st.sidebar.expander("🛠️ Admin Tools"):
        st.info("Trigger the background scanner to audit the full universe (800+ funds). This runs independently.")
        if st.button("🚀 Run Background Discovery Scan"):
            try:
                # Run independent process
                # Ensure we point to the correct file path
                script_path = os.path.join(os.getcwd(), "cron_mf_audit.py")
                if os.path.exists(script_path):
                    subprocess.Popen([sys.executable, script_path])
                    st.toast("✅ Background scan started! It will take ~15 mins.")
                else:
                    st.error(f"Script not found at {script_path}")
            except Exception as e:
                st.error(f"Failed to start scan: {e}")

    # ---------------- VIEW RESULTS ----------------
    # 1. Fetch Latest Data
    timestamps = fetch_timestamps("scan_mf")

    if not timestamps:
        st.warning("No audit history found. Please run a scan from the Admin Tools.")
        return

    selected_ts = st.selectbox("Select Audit Date", timestamps, index=0)

    # Fetch Data
    # fetch_history_data returns the raw dataframe stored in DB
    raw_df = fetch_history_data("scan_mf", selected_ts)

    if raw_df.empty:
        st.error("No data found for selected timestamp.")
        return

    # PATCH: Schema Evolution for Old Scans
    # If "Category" column is missing, infer it from Symbol using shared logic
    if 'Category' not in raw_df.columns and 'Symbol' in raw_df.columns:
        from mf_lab.logic import get_category
        # Apply inference
        raw_df['Category'] = raw_df['Symbol'].apply(get_category)
        st.caption("ℹ️ Legacy Data Detected: Categories were inferred from fund names.")

    # --- DRIFT ANALYSIS APPLICATION ---
    # Apply the drift logic (Integrity Badges)
    raw_df = apply_drift_status(raw_df)

    # --- DRIFT WATCHLIST ---
    st.markdown("### 🚨 Style Drift Watchlist")

    # Filter for critical drift
    if "Drift Status" in raw_df.columns:
        critical_drifts = raw_df[raw_df['Drift Status'] == "Critical"]
        moderate_drifts = raw_df[raw_df['Drift Status'] == "Moderate"]
        stable_count = len(raw_df[raw_df['Drift Status'] == "Stable"])
        total_funds = len(raw_df)

        # Integrity Score Metric
        integrity_score = (stable_count / total_funds * 100) if total_funds > 0 else 0
        st.metric("Total Market Integrity Score", f"{integrity_score:.1f}%", f"{stable_count} Stable Funds")

        if not critical_drifts.empty:
            for _, row in critical_drifts.head(3).iterrows():
                 st.error(f"🚨 **CRITICAL DRIFT**: {row['Symbol']} - {row['Drift Message']}")
            if len(critical_drifts) > 3:
                st.caption(f"...and {len(critical_drifts) - 3} more critical drifts.")

        if critical_drifts.empty and not moderate_drifts.empty:
             st.warning(f"⚠️ **Moderate Drift Alert**: {len(moderate_drifts)} funds are showing minor style deviations.")

        if critical_drifts.empty and moderate_drifts.empty:
            st.success("✅ Market Integrity Check: No significant style drifts detected.")

    st.success(f"Loaded {len(raw_df)} funds from audit on {selected_ts}")

    # 2. Category-Wise Normalization & Leaderboards
    # We recalculate the 0-100 Score here to ensure it's relative to the *current* peer group displayed

    tab_list = ["Large Cap", "Mid Cap", "Small Cap", "Flexi/Other"]
    tabs = st.tabs(tab_list)

    final_display_df = pd.DataFrame()

    for i, cat in enumerate(tab_list):
        with tabs[i]:
            cat_df = raw_df[raw_df['Category'] == cat].copy()

            if not cat_df.empty:
                st.markdown(f"### 🏆 {cat} Consistency Leaderboard")

                # Normalize Score 0-100 per category (Peer Normalization)
                if len(cat_df) > 1:
                    c_min, c_max = cat_df['Score'].min(), cat_df['Score'].max()
                    if c_max != c_min:
                        # Min-Max Scaling
                        cat_df['Fortress Score'] = ((cat_df['Score'] - c_min) / (c_max - c_min)) * 100
                    else:
                        cat_df['Fortress Score'] = 50.0
                else:
                    cat_df['Fortress Score'] = 100.0

                cat_df['Fortress Score'] = cat_df['Fortress Score'].round(1)

                # Sort by Fortress Score
                cat_df = cat_df.sort_values("Fortress Score", ascending=False)

                # Format Columns for Display
                disp_cols = ["Integrity", "Symbol", "Fortress Score", "Alpha (True)", "Sortino", "Win Rate", "Upside Cap", "Downside Cap", "Verdict"]

                # Ensure columns exist (handle potential missing cols from old scans)
                valid_cols = [c for c in disp_cols if c in cat_df.columns]

                # Styling
                st.dataframe(
                    cat_df[valid_cols].style.background_gradient(subset=['Fortress Score'], cmap='RdYlGn'),
                    use_container_width=True,
                    height=500
                )

                final_display_df = pd.concat([final_display_df, cat_df])
            else:
                st.info(f"No funds found in {cat} category.")

    # 3. Global Analysis (Scatter Plot)
    if not final_display_df.empty:
        st.markdown("---")
        st.subheader("📊 Global Map: Alpha vs Downside Protection")

        # --- RESILIENT PLOTTING STRATEGY ---
        # 1. Mandatory Columns Check
        required_cols = ["Downside Cap", "Alpha (True)"]
        missing_mandatory = [c for c in required_cols if c not in final_display_df.columns]

        if missing_mandatory:
            st.info(f"Scatter plot unavailable for legacy scans. Missing metrics: {', '.join(missing_mandatory)}")
        else:
            # 2. Prepare Optional Attributes (Size & Color)
            plot_kwargs = {
                "data_frame": final_display_df,
                "x": "Downside Cap",
                "y": "Alpha (True)",
                "text": "Symbol",
                "title": "Risk-Reward Map",
                "hover_data": [c for c in ["Sortino", "Win Rate", "Category"] if c in final_display_df.columns]
            }

            # Handle Color (Verdict)
            if "Verdict" in final_display_df.columns:
                final_display_df["Verdict"] = final_display_df["Verdict"].fillna("Historical")
                plot_kwargs["color"] = "Verdict"

            # Handle Size (Fortress Score)
            if "Fortress Score" in final_display_df.columns:
                # Clip to min 0.1 to avoid size=0 errors, fill NaNs
                final_display_df["Fortress Score"] = final_display_df["Fortress Score"].clip(lower=0.1).fillna(10.0)
                plot_kwargs["size"] = "Fortress Score"
                plot_kwargs["title"] += " (Size = Fortress Score)"

            try:
                fig = px.scatter(**plot_kwargs)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render chart due to data format issues: {e}")

    # 4. Integrity Audit (Drift Map)
    if not final_display_df.empty:
        st.markdown("---")
        st.subheader("🔍 Integrity Audit: Alpha vs Tracking Error")

        # Check required columns for Drift Map
        drift_cols = ["Tracking Error", "Alpha (True)"]
        # Use 'te' if 'Tracking Error' is missing (handle alias)
        if "Tracking Error" not in final_display_df.columns and "te" in final_display_df.columns:
            final_display_df["Tracking Error"] = final_display_df["te"]

        missing_drift = [c for c in drift_cols if c not in final_display_df.columns]

        if missing_drift:
             st.info(f"Drift map unavailable. Missing metrics: {', '.join(missing_drift)}")
        else:
             drift_kwargs = {
                "data_frame": final_display_df,
                "x": "Tracking Error",
                "y": "Alpha (True)",
                "text": "Symbol",
                "title": "Drift Map (Color = Drift Status)",
                "hover_data": [c for c in ["Beta", "Category", "Drift Message"] if c in final_display_df.columns]
            }

             if "Drift Status" in final_display_df.columns:
                 drift_kwargs["color"] = "Drift Status"
                 # Custom color map
                 drift_kwargs["color_discrete_map"] = {
                     "Stable": "green", "Moderate": "orange", "Critical": "red", "Unknown": "gray"
                 }

             try:
                fig_drift = px.scatter(**drift_kwargs)
                # Add zones or lines if needed? For now simple scatter.
                st.plotly_chart(fig_drift, use_container_width=True)
             except Exception as e:
                st.error(f"Could not render drift map: {e}")

    # 5. Deep Dive Modal
    st.markdown("---")
    st.subheader("🕵️ Fund Deep Dive")

    # Select Fund
    all_symbols = sorted(raw_df['Symbol'].unique())
    selected_fund = st.selectbox("Select Fund for Historical Analysis", all_symbols)

    if selected_fund:
        hist_df = fetch_symbol_history("scan_mf", selected_fund)
        if not hist_df.empty:
             st.markdown(f"#### Historical Performance: {selected_fund}")

             # Create Dual-Axis Chart: Fortress Score vs Tracking Error (or Beta)
             # First, ensure date/timestamp is datetime
             hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
             hist_df = hist_df.sort_values('timestamp')

             # Check available columns in history
             # Note: fetch_symbol_history now returns * so we should have everything stored in history.
             # However, old history might lack 'Beta' or 'Tracking Error' columns if schema evolved.

             cols_to_plot = []
             if "Score" in hist_df.columns: cols_to_plot.append("Score") # Raw Score

             # Metric 2
             metric2 = None
             if "Tracking Error" in hist_df.columns: metric2 = "Tracking Error"
             elif "te" in hist_df.columns: metric2 = "te"
             elif "Beta" in hist_df.columns: metric2 = "Beta"

             if metric2 and cols_to_plot:
                 from plotly.subplots import make_subplots
                 import plotly.graph_objects as go

                 fig_hist = make_subplots(specs=[[{"secondary_y": True}]])

                 # Trace 1: Fortress Score (or Raw Score)
                 fig_hist.add_trace(
                     go.Scatter(x=hist_df['timestamp'], y=hist_df['Score'], name="Fortress Score", mode='lines+markers'),
                     secondary_y=False
                 )

                 # Trace 2: Drift Metric
                 fig_hist.add_trace(
                     go.Scatter(x=hist_df['timestamp'], y=hist_df[metric2], name=metric2, mode='lines+markers', line=dict(dash='dot')),
                     secondary_y=True
                 )

                 fig_hist.update_layout(title=f"Trend: Score vs {metric2}")
                 fig_hist.update_yaxes(title_text="Score", secondary_y=False)
                 fig_hist.update_yaxes(title_text=metric2, secondary_y=True)

                 st.plotly_chart(fig_hist, use_container_width=True)
             else:
                 st.warning("Insufficient historical metrics for deep dive chart.")
                 st.dataframe(hist_df) # Fallback
        else:
            st.info("No historical data found for this fund.")

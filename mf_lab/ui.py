import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import sys
import os
from utils.db import fetch_timestamps, fetch_history_data

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
                disp_cols = ["Symbol", "Fortress Score", "Alpha (True)", "Sortino", "Win Rate", "Upside Cap", "Downside Cap", "Verdict"]

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

        # Ensure Fortress Score exists in final_display_df (it should from the loop)
        fig = px.scatter(
            final_display_df,
            x="Downside Cap",
            y="Alpha (True)",
            color="Verdict",
            size="Fortress Score",
            text="Symbol",
            hover_data=["Sortino", "Win Rate", "Category"],
            title="Risk-Reward Map (Size = Fortress Score)"
        )
        # Reverse X axis because Lower Downside Cap is better?
        # Actually standard interpretation: Low Downside Cap is good.
        # Let's keep it standard but maybe note it.
        # Usually scatter plots are intuitive: Top Left (High Alpha, Low Downside) is best?
        # Yes, low downside cap is < 100.

        st.plotly_chart(fig, use_container_width=True)

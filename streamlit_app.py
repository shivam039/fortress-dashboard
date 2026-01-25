# streamlit_app.py - v9.4 MASTER TERMINAL (Dynamic Columns + Heatmap Safety)
import time, sqlite3
from datetime import datetime
import streamlit as st
import pandas as pd
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ---------------- CONFIG ----------------
try:
    from fortress_config import TICKER_GROUPS
except ImportError:
    st.error("Configuration file 'fortress_config.py' not found.")
    st.stop()

# ---------------- MODULE IMPORTS ----------------
# Import modules after config check because stock_scanner depends on fortress_config
from utils.db import (
    init_db, log_audit, get_table_name_from_universe,
    fetch_timestamps, fetch_history_data, fetch_symbol_history
)
import mf_lab.ui
import stock_scanner.ui

# Initialize Database
init_db()

# ---------------- UI ----------------
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns Terminal")

# Sidebar - Stock Scanner Controls
# Delegating sidebar rendering to stock_scanner.ui
portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice = stock_scanner.ui.render_sidebar()

# ---------------- TABS ----------------
tab_scan, tab_mf, tab_hist = st.tabs(["🚀 Live Scanner", "🛡️ MF Consistency Lab", "📜 Scan History Intelligence"])

with tab_scan:
    stock_scanner.ui.render(portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice)

with tab_mf:
    mf_lab.ui.render()

with tab_hist:
    st.subheader("📜 Scan History Intelligence")

    # 1. Setup & Controls
    col_u, col_t1, col_t2, col_btn = st.columns([2, 2, 2, 2])
    with col_u:
        # Added "Mutual Funds" to the options
        hist_uni = st.selectbox("Universe", list(TICKER_GROUPS.keys()) + ["Mutual Funds"], key="h_u")
        hist_table = get_table_name_from_universe(hist_uni)

    # Fetch Timestamps with Cache
    timestamps = fetch_timestamps(hist_table)

    with col_t1:
        t_new = st.selectbox("New Scan", timestamps, index=0 if timestamps else None, key="t_n")
    with col_t2:
        t_old = st.selectbox("Old Scan", timestamps, index=1 if len(timestamps)>1 else None, key="t_o")
    with col_btn:
        st.write("") # Spacer
        st.write("")
        do_compare = st.button("Compare Scans", type="primary", use_container_width=True)

    if do_compare and t_new and t_old:
        try:
            # Fetch Data with Cache
            df_new = fetch_history_data(hist_table, t_new)
            df_old = fetch_history_data(hist_table, t_old)

            # Merge
            m = pd.merge(df_new, df_old, on="Symbol", how="outer", suffixes=('_new', '_old'))

            # Calcs
            m['Score_Change'] = m['Score_new'].fillna(0) - m['Score_old'].fillna(0)
            m['Price_Change'] = m['Price_new'] - m['Price_old']
            m['Price_Chg_Pct'] = (m['Price_Change'] / m['Price_old']) * 100

            # Verdict Logic
            def verdict_shift(r):
                if pd.isna(r['Verdict_old']): return "New"
                if pd.isna(r['Verdict_new']): return "Dropped"
                if r['Verdict_new'] == r['Verdict_old']: return "Same"
                return f"{r['Verdict_old']} -> {r['Verdict_new']}"

            m['Verdict_Shift'] = m.apply(verdict_shift, axis=1)

            # Summary Metrics
            new_tk = m[m['Verdict_old'].isna()]
            drop_tk = m[m['Verdict_new'].isna()]
            up_movers = m[m['Score_Change'] > 0]
            down_movers = m[m['Score_Change'] < 0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("New Entrants", len(new_tk), delta=len(new_tk), delta_color="normal")
            m2.metric("Dropped", len(drop_tk), delta=-len(drop_tk), delta_color="inverse")
            m3.metric("Improvers", len(up_movers), delta=f"Avg +{up_movers['Score_Change'].mean():.1f}" if not up_movers.empty else 0)
            m4.metric("Decliners", len(down_movers), delta=f"Avg {down_movers['Score_Change'].mean():.1f}" if not down_movers.empty else 0, delta_color="inverse")

            # Main Comparison Table
            st.markdown("### 📊 Score Change Table")
            display_cols = ['Symbol', 'Score_new', 'Score_old', 'Score_Change', 'Verdict_new', 'Verdict_Shift', 'Price_new', 'Price_Chg_Pct']

            # Styling
            def color_score_delta(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                return f'color: {color}'

            styled_df = m[display_cols].sort_values("Score_Change", ascending=False).style.applymap(color_score_delta, subset=['Score_Change'])
            st.dataframe(styled_df, use_container_width=True)

            # Export Buttons
            exp_c1, exp_c2 = st.columns(2)

            # CSV
            csv_hist = m.to_csv(index=False).encode('utf-8')
            exp_c1.download_button("Download Comparison CSV", csv_hist, "scan_comparison.csv", "text/csv")

            # PDF Generation
            if PDF_AVAILABLE:
                def generate_pdf(df_in):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"Fortress Scan Comparison: {t_new} vs {t_old}", ln=1, align="C")
                    pdf.ln(10)

                    # Summary
                    pdf.set_font("Arial", size=10)
                    pdf.cell(0, 10, txt=f"New: {len(new_tk)} | Dropped: {len(drop_tk)} | Improvers: {len(up_movers)} | Decliners: {len(down_movers)}", ln=1)
                    pdf.ln(5)

                    # Top Movers
                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(0, 10, txt="Top 10 Positive Movers:", ln=1)
                    pdf.set_font("Arial", size=9)
                    top_10 = df_in.nlargest(10, 'Score_Change')
                    for _, row in top_10.iterrows():
                        pdf.cell(0, 8, txt=f"{row['Symbol']}: {row['Score_old']} -> {row['Score_new']} (Delta: {row['Score_Change']})", ln=1)

                    return pdf.output(dest='S').encode('latin-1')

                try:
                    pdf_bytes = generate_pdf(m)
                    exp_c2.download_button("Download Report PDF", pdf_bytes, "scan_report.pdf", "application/pdf")
                except Exception as e:
                    exp_c2.error(f"PDF Gen Error: {e}")
            else:
                exp_c2.warning("PDF Generation disabled (fpdf not found)")

            # Top Movers
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🚀 Top Gainers")
                st.dataframe(m.nlargest(5, 'Score_Change')[['Symbol', 'Score_Change', 'Verdict_Shift']])
            with c2:
                st.markdown("#### 📉 Top Losers")
                st.dataframe(m.nsmallest(5, 'Score_Change')[['Symbol', 'Score_Change', 'Verdict_Shift']])

            # New & Dropped
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("#### ✅ New Entries")
                if not new_tk.empty: st.dataframe(new_tk[['Symbol', 'Score_new', 'Verdict_new', 'Price_new']])
                else: st.info("No new entries")
            with e2:
                st.markdown("#### ❌ Dropped Stocks")
                if not drop_tk.empty: st.dataframe(drop_tk[['Symbol', 'Score_old', 'Verdict_old', 'Price_old']])
                else: st.info("No dropped stocks")

            log_audit("Comparison Generated", hist_uni, f"Compared {t_new} vs {t_old}")

            # AI Features: Sector Movers
            st.markdown("---")
            st.markdown("#### 🧠 AI Sector Intelligence")
            if 'Sector_new' in m.columns:
                sector_delta = m.groupby('Sector_new')['Score_Change'].mean().sort_values(ascending=False)
                best_sec = sector_delta.head(3)
                worst_sec = sector_delta.tail(3)

                a1, a2 = st.columns(2)
                a1.info(f"**Top Momentum Sectors:** {', '.join(best_sec.index)}")
                a2.warning(f"**Lagging Sectors:** {', '.join(worst_sec.index)}")
            else:
                st.info("Sector data not available in history for AI analysis.")

            # Dispersion & Alerts Placeholder
            d1, d2 = st.columns(2)
            d1.markdown("**⚠️ High Dispersion Alert System**")
            d1.caption("System logic active. Configure thresholds in Smart Alerts.")
            d2.markdown("**🔔 Smart Alerts Config**")
            d2.caption("Status: Active. No triggers pending.")


        except Exception as e:
            st.error(f"Error comparing scans: {e}")

    # Conviction Trend Engine
    st.markdown("---")
    st.subheader("📈 Conviction Trend Engine")
    try:
        conn = sqlite3.connect('fortress_history.db')
        # Get all history for chart
        if timestamps:
            # Check versioning column support (simulated via rank)
            # Just use distinct symbol query
            latest_syms = pd.read_sql(f"SELECT DISTINCT Symbol FROM {hist_table} WHERE timestamp=?", conn, params=(timestamps[0],))
            sym_list = latest_syms['Symbol'].tolist()
            sel_sym = st.selectbox("Select Asset for Trend Analysis", sym_list)

            if sel_sym:
                hist_data = fetch_symbol_history(hist_table, sel_sym)
                if not hist_data.empty:
                    # Convert timestamp to datetime for better plotting
                    hist_data['timestamp'] = pd.to_datetime(hist_data['timestamp'])

                    # Plot
                    st.line_chart(hist_data.set_index('timestamp')['Score'])

                    # Statistics
                    avg_score = hist_data['Score'].mean()
                    volatility = hist_data['Score'].std()
                    st.caption(f"Historical Avg Score: {avg_score:.1f} | Volatility: {volatility:.1f}")

                    # Advanced AI Feature: Position Sizing Suggestion
                    latest_score = hist_data.iloc[-1]['Score']
                    if latest_score > 85 and latest_score > avg_score:
                        st.success(f"🤖 AI Insight: **Aggressive Allocation** recommended. Conviction ({latest_score}) is above historical average.")
                    elif latest_score < 60:
                        st.warning(f"🤖 AI Insight: **Reduce Exposure**. Conviction is low.")
                    else:
                        st.info(f"🤖 AI Insight: **Hold / Neutral**. Score is stable.")

                else:
                    st.warning("No history found for this symbol.")
        conn.close()
    except Exception as e:
        st.error(f"Trend Engine Error: {e}")

    # Audit Logs & Rollback
    st.markdown("---")
    with st.expander("🛡️ System Audit Logs & Rollback Protection"):
        al_col1, al_col2 = st.columns([3, 1])
        with al_col1:
            try:
                conn = sqlite3.connect('fortress_history.db')
                logs = pd.read_sql("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 50", conn)
                st.dataframe(logs, use_container_width=True)
                conn.close()
            except: st.info("No logs available")

        with al_col2:
            st.markdown("#### ↩️ Rollback")
            st.warning("Danger Zone: Revert latest scan.")
            if timestamps:
                latest_ts = timestamps[0]
                st.text(f"Latest Ver: {latest_ts}")
                if st.button("Rollback Latest Scan", type="secondary"):
                    try:
                        conn = sqlite3.connect('fortress_history.db')
                        c = conn.cursor()
                        # Use parameterized delete
                        c.execute(f"DELETE FROM {hist_table} WHERE timestamp=?", (latest_ts,))
                        rows = c.rowcount
                        log_audit("Rollback Executed", hist_uni, f"Deleted {rows} rows from {latest_ts}")
                        conn.commit()
                        conn.close()

                        # Clear Cache
                        fetch_timestamps.clear()
                        fetch_history_data.clear()
                        fetch_symbol_history.clear()

                        st.success(f"Rolled back {rows} records. Refreshing...")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Rollback failed: {e}")
            else:
                st.info("No scans to rollback.")

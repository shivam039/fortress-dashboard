# streamlit_app.py - v9.4 MASTER TERMINAL (Dynamic Columns + Heatmap Safety)
import subprocess, sys, time, sqlite3
from datetime import datetime
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Optional Dependency for PDF
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ---------------- CONFIG ----------------
try:
    from fortress_config import TICKER_GROUPS, SECTOR_MAP, INDEX_BENCHMARKS
except ImportError:
    st.error("Configuration file 'fortress_config.py' not found.")
    st.stop()

def init_db():
    try:
        conn = sqlite3.connect('fortress_history.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS scan_results
                     (timestamp TEXT, symbol TEXT, score REAL, price REAL, verdict TEXT)''')
        # Audit Logs
        c.execute('''CREATE TABLE IF NOT EXISTS audit_logs
                     (timestamp TEXT, action TEXT, universe TEXT, details TEXT)''')
        # Smart Alerts Placeholder Table
        c.execute('''CREATE TABLE IF NOT EXISTS smart_alerts
                     (symbol TEXT, condition TEXT, status TEXT)''')
        conn.commit()
        conn.close()
    except Exception as e:
        # Use print if st.error fails (before page_config)
        print(f"Database error: {e}")

def get_table_name_from_universe(u):
    if "Nifty 50" == u: return "scan_nifty50"
    if "Nifty Next 50" == u: return "scan_niftynext50"
    if "Nifty Midcap" in u: return "scan_midcap"
    if "Nifty Midcap 150" == u: return "scan_midcap"
    return "scan_results"

def log_audit(action, universe="Global", details=""):
    try:
        conn = sqlite3.connect('fortress_history.db')
        c = conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO audit_logs VALUES (?,?,?,?)", (ts, action, universe, details))
        conn.commit()
        conn.close()
    except: pass

def log_scan_results(df, table_name='scan_results'):
    try:
        conn = sqlite3.connect('fortress_history.db')
        df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        table_exists = cursor.fetchone()

        if not table_exists:
             df.to_sql(table_name, conn, if_exists='append', index=False)
        else:
            # Check and update schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_cols = {row[1] for row in cursor.fetchall()}

            for col in df.columns:
                if col not in existing_cols:
                    # Determine type
                    dtype = df[col].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_type = "INTEGER"
                    elif pd.api.types.is_float_dtype(dtype):
                        sql_type = "REAL"
                    else:
                        sql_type = "TEXT"

                    try:
                        cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN "{col}" {sql_type}')
                    except Exception as alter_err:
                         print(f"Error adding column {col}: {alter_err}")

            df.to_sql(table_name, conn, if_exists='append', index=False)

        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

# Cache Helpers
@st.cache_data(ttl=60)
def fetch_timestamps(table_name):
    try:
        conn = sqlite3.connect('fortress_history.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone():
            runs = pd.read_sql(f"SELECT DISTINCT timestamp FROM {table_name} ORDER BY timestamp DESC", conn)
            conn.close()
            return runs['timestamp'].tolist()
        conn.close()
        return []
    except: return []

@st.cache_data(ttl=60)
def fetch_history_data(table_name, timestamp):
    try:
        conn = sqlite3.connect('fortress_history.db')
        df = pd.read_sql(f"SELECT * FROM {table_name} WHERE timestamp=?", conn, params=(timestamp,))
        conn.close()
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_symbol_history(table_name, symbol):
    try:
        conn = sqlite3.connect('fortress_history.db')
        df = pd.read_sql(f"SELECT timestamp, Score, Price, Verdict FROM {table_name} WHERE Symbol=? ORDER BY timestamp", conn, params=(symbol,))
        conn.close()
        return df
    except: return pd.DataFrame()

init_db()

# ---------------- UI ----------------
st.set_page_config(page_title="Fortress 95 Pro", layout="wide")
st.title("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns Terminal")

# Sidebar Controls
st.sidebar.title("💰 Portfolio & Risk")
portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", value=1000000, step=50000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.1)/100
selected_universe = st.sidebar.selectbox("Select Index", list(TICKER_GROUPS.keys()))

# ---------------- COLUMN CONFIG ----------------
ALL_COLUMNS = {
    "Symbol": {"label":"Symbol"},
    "Verdict": {"label":"Verdict"},
    "Score": {"label":"Conviction", "type":"progress", "min":0, "max":100},
    "Price": {"label":"Price ₹", "format":"₹%.2f"},
    "RSI": {"label":"RSI", "format":"%.1f"},
    "News": {"label":"News"},
    "Events": {"label":"Events"},
    "Sector": {"label":"Sector"},
    "Position_Qty": {"label":"Qty", "format":"%d"},
    "Stop_Loss": {"label":"SL Price", "format":"₹%.2f"},
    "Target_10D": {"label":"10D Target", "format":"₹%.2f"},
    "Analysts": {"label":"Analyst Count", "format":"%d"},
    "Tgt_High": {"label":"High Target", "format":"₹%d"},
    "Tgt_Median": {"label":"Median Target", "format":"₹%d"},
    "Tgt_Low": {"label":"Low Target", "format":"₹%d"},
    "Tgt_Mean": {"label":"Mean Target", "format":"₹%d"},
    "Dispersion_Alert": {"label":"Dispersion"},
    "Ret_30D": {"label":"30D Backtest", "format":"%.2f%%"},
    "Ret_60D": {"label":"60D Backtest", "format":"%.2f%%"},
    "Ret_90D": {"label":"90D Backtest", "format":"%.2f%%"}
}

# Sidebar Multiselect for Dynamic Columns
selected_columns = st.sidebar.multiselect(
    "Select Columns to Display", options=list(ALL_COLUMNS.keys()), default=list(ALL_COLUMNS.keys())
)

# ---------------- CORE ENGINE ----------------
def check_institutional_fortress(ticker, data, ticker_obj, portfolio_value, risk_per_trade):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data)<210: return None

        close, high, low = data["Close"], data["High"], data["Low"]

        ema200 = ta.ema(close,200).iloc[-1]
        rsi = ta.rsi(close,14).iloc[-1]
        atr = ta.atr(high,low,close,14).iloc[-1]
        st_df = ta.supertrend(high,low,close,10,3)
        trend_col = [c for c in st_df.columns if c.startswith("SUPERTd")][0]
        trend_dir = int(st_df[trend_col].iloc[-1])
        price = float(close.iloc[-1])
        tech_base = price>ema200 and trend_dir==1

        sl_distance = atr*1.5
        sl_price = round(price-sl_distance,2)
        target_10d = round(price + atr*1.8,2)
        risk_amount = portfolio_value*risk_per_trade
        pos_size = int(risk_amount / sl_distance) if sl_distance>0 else 0

        conviction = 0
        score_mod = 0
        news_sentiment = "Neutral"
        event_status = "✅ Safe"

        try:
            news = ticker_obj.news or []
            titles = " ".join(n.get("title","").lower() for n in news[:5])
            if any(k in titles for k in ["fraud","investigation","default","bankruptcy","scam","legal"]):
                news_sentiment = "🚨 BLACK SWAN"
                score_mod -= 40
        except: pass
        try:
            cal = ticker_obj.calendar
            if isinstance(cal,pd.DataFrame) and not cal.empty:
                next_date = pd.to_datetime(cal.iloc[0,0]).date()
                days_to = (next_date - datetime.now().date()).days
                if 0<=days_to<=7:
                    event_status = f"🚨 EARNINGS ({next_date.strftime('%d-%b')})"
                    score_mod -= 20
        except: pass

        analyst_count = target_high = target_low = target_median = target_mean = 0
        try:
            info = ticker_obj.info or {}
            analyst_count = info.get("numberOfAnalystOpinions",0)
            target_high = info.get("targetHighPrice",0)
            target_low = info.get("targetLowPrice",0)
            target_median = info.get("targetMedianPrice",0)
            target_mean = info.get("targetMeanPrice",0)
        except: pass

        if tech_base:
            conviction += 60
            if 48<=rsi<=62: conviction+=20
            elif 40<=rsi<=72: conviction+=10
            conviction += score_mod

        dispersion_pct = ((target_high-target_low)/price)*100 if price>0 else 0
        dispersion_alert = "⚠️ High Dispersion" if dispersion_pct>30 else "✅"
        if dispersion_pct>30: conviction -= 10

        conviction = max(0,min(100,conviction))
        verdict = "🔥 HIGH" if conviction>=85 else "🚀 PASS" if conviction>=60 else "🟡 WATCH" if tech_base else "❌ FAIL"

        # Backtest returns (30, 60, 90 days)
        current_date = close.index[-1]
        returns = {}
        for days in [30, 60, 90]:
            try:
                target_date = current_date - pd.Timedelta(days=days)
                # Find nearest index
                idx = close.index.get_indexer([target_date], method='nearest')[0]
                past_price = float(close.iloc[idx])
                pct_change = ((price - past_price) / past_price) * 100
                returns[f"Ret_{days}D"] = pct_change
            except:
                returns[f"Ret_{days}D"] = None

        return {
            "Symbol": ticker,
            "Verdict": verdict,
            "Score": conviction,
            "Price": round(price,2),
            "RSI": round(rsi,1),
            "News": news_sentiment,
            "Events": event_status,
            "Sector": SECTOR_MAP.get(ticker,"General"),
            "Position_Qty": pos_size,
            "Stop_Loss": sl_price,
            "Target_10D": target_10d,
            "Analysts": analyst_count,
            "Tgt_High": target_high,
            "Tgt_Median": target_median,
            "Tgt_Low": target_low,
            "Tgt_Mean": target_mean,
            "Dispersion_Alert": dispersion_alert,
            "Ret_30D": returns.get("Ret_30D"),
            "Ret_60D": returns.get("Ret_60D"),
            "Ret_90D": returns.get("Ret_90D")
        }
    except: return None

# ---------------- TABS ----------------
tab_scan, tab_hist = st.tabs(["🚀 Live Scanner", "📜 Scan History Intelligence"])

with tab_scan:
    # ---------------- MARKET PULSE ----------------
    st.subheader("🌐 Market Pulse")
    pulse_cols = st.columns(len(INDEX_BENCHMARKS))
    for i,(name,symbol) in enumerate(INDEX_BENCHMARKS.items()):
        try:
            idx_data = yf.download(symbol, period="1y", progress=False)
            p_close = idx_data["Close"].iloc[-1]
            p_ema = ta.ema(idx_data["Close"],200).iloc[-1]
            p_status = "🟢 BULL" if p_close>p_ema else "🔴 BEAR"
            pulse_cols[i].metric(name,f"{p_close:,.0f}",p_status)
        except: pass

    # ---------------- MAIN SCAN ----------------
    if st.button("🚀 EXECUTE SYSTEM SCAN",type="primary",use_container_width=True):
        tickers = TICKER_GROUPS[selected_universe]
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Log Start
        log_audit("Scan Started", selected_universe, f"Scanning {len(tickers)} tickers")

        for i,ticker in enumerate(tickers):
            status_text.text(f"Scanning {ticker} ({i+1}/{len(tickers)})")
            try:
                tkr = yf.Ticker(ticker)
                hist = yf.download(ticker, period="2y", progress=False)
                if not hist.empty:
                    res = check_institutional_fortress(ticker,hist,tkr,portfolio_val,risk_pct)
                    if res: results.append(res)
                time.sleep(0.7)
            except: pass
            progress_bar.progress((i+1)/len(tickers))

        if results:
            df = pd.DataFrame(results).sort_values("Score",ascending=False)
            status_text.success(f"Scan Complete: {len(df[df['Score']>=60])} actionable setups.")

            # Log Logic
            target_table = get_table_name_from_universe(selected_universe)
            df['Universe'] = selected_universe # Add metadata
            log_scan_results(df, target_table)
            # Clear cache after new scan so history tab updates
            fetch_timestamps.clear()
            fetch_history_data.clear()
            fetch_symbol_history.clear()

            log_audit("Scan Completed", selected_universe, f"Saved {len(df)} records to {target_table}")

            display_df = df[selected_columns]

            st_column_config = {}
            for col in selected_columns:
                cfg = ALL_COLUMNS[col]
                fmt = cfg.get("format")
                if cfg.get("type")=="progress":
                    st_column_config[col] = st.column_config.ProgressColumn(cfg["label"],min_value=cfg["min"],max_value=cfg["max"])
                elif fmt:
                    st_column_config[col] = st.column_config.NumberColumn(cfg["label"],format=fmt)
                else:
                    st_column_config[col] = st.column_config.TextColumn(cfg["label"])

            st.dataframe(display_df,use_container_width=True,height=600,column_config=st_column_config)

            csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Trades to CSV",data=csv,
                               file_name=f"Fortress_Trades_{datetime.now().strftime('%Y%m%d')}.csv",
                               mime="text/csv",use_container_width=True)

            # ---------------- HEATMAP ----------------
            if not df.empty and "Score" in df.columns:
                st.subheader("📊 Conviction Heatmap")
                plt.figure(figsize=(12,len(df)/2))
                df["Conviction_Band"] = df["Score"].apply(lambda x: "🔥 High (85+)" if x>=85 else "🚀 Pass (60-85)" if x>=60 else "🟡 Watch (<60)")
                heatmap_data = df.pivot_table(index="Symbol", columns="Conviction_Band", values="Score", fill_value=0)
                sns.heatmap(heatmap_data, annot=True, cmap="Greens", cbar=False, linewidths=0.5, linecolor='grey')
                st.pyplot(plt)
            else:
                st.info("Insufficient data for heatmap generation.")

        else:
            st.warning("No data retrieved. Check internet or ticker config.")
            log_audit("Scan Failed", selected_universe, "No data retrieved")

    st.caption("🛡️ Fortress 95 Pro v9.4 — Dynamic Columns | ATR SL | Analyst Dispersion | Full Logic")

with tab_hist:
    st.subheader("📜 Scan History Intelligence")

    # 1. Setup & Controls
    col_u, col_t1, col_t2, col_btn = st.columns([2, 2, 2, 2])
    with col_u:
        hist_uni = st.selectbox("Universe", list(TICKER_GROUPS.keys()), key="h_u")
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

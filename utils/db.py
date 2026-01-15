import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime

# ---------------- DB LOGIC ----------------
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
    if "Mutual Funds" == u: return "scan_mf"
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

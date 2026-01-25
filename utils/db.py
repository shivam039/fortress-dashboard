import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
import logging

# Configure basic logging for DB operations (will be overridden by main app config)
logger = logging.getLogger(__name__)

DB_NAME = 'fortress_history.db'

# ---------------- DB INITIALIZATION ----------------
def init_db():
    """Initializes the database with the new Enterprise Schema."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # 1. Scans Metadata Table
        c.execute('''CREATE TABLE IF NOT EXISTS scans (
                        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        universe TEXT,
                        status TEXT
                    )''')

        # 2. Scan Results (Fact Table) - RENAMED TO scan_entries TO AVOID COLLISION
        # scan_id links to scans.id
        c.execute('''CREATE TABLE IF NOT EXISTS scan_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_id INTEGER,
                        symbol TEXT,
                        scheme_code TEXT,
                        category TEXT,
                        score REAL,
                        price REAL,
                        integrity_label TEXT,
                        drift_status TEXT,
                        drift_message TEXT,
                        FOREIGN KEY(scan_id) REFERENCES scans(scan_id)
                    )''')

        # 3. Fund Metrics (Details)
        c.execute('''CREATE TABLE IF NOT EXISTS fund_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_id INTEGER,
                        symbol TEXT,
                        alpha REAL,
                        beta REAL,
                        te REAL,
                        sortino REAL,
                        max_dd REAL,
                        win_rate REAL,
                        upside REAL,
                        downside REAL,
                        cagr REAL,
                        FOREIGN KEY(scan_id) REFERENCES scans(scan_id)
                    )''')

        # 4. Alerts
        c.execute('''CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_id INTEGER,
                        symbol TEXT,
                        alert_type TEXT,
                        severity TEXT,
                        message TEXT,
                        timestamp TEXT,
                        FOREIGN KEY(scan_id) REFERENCES scans(scan_id)
                    )''')

        # 5. Benchmark History (Caching)
        # Composite PK: ticker + date
        c.execute('''CREATE TABLE IF NOT EXISTS benchmark_history (
                        ticker TEXT,
                        date TEXT,
                        close REAL,
                        ret REAL,
                        PRIMARY KEY (ticker, date)
                    )''')

        # 6. Commodity Scans
        c.execute('''CREATE TABLE IF NOT EXISTS scan_commodities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_id INTEGER,
                        symbol TEXT,
                        global_price REAL,
                        local_price REAL,
                        usd_inr REAL,
                        parity_price REAL,
                        spread REAL,
                        arb_yield REAL,
                        action_label TEXT,
                        FOREIGN KEY(scan_id) REFERENCES scans(scan_id)
                    )''')

        # Legacy Tables Support (Optional: keep them if needed or let them be)
        # c.execute('''CREATE TABLE IF NOT EXISTS scan_results ...''') # Old flat table

        conn.commit()

        # Create Indexes for Performance
        c.execute("CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entries_scan_sym ON scan_entries(scan_id, symbol)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_metrics_scan_sym ON fund_metrics(scan_id, symbol)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp)")

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database initialization error: {e}")

# ---------------- HELPER FUNCTIONS ----------------

def get_connection():
    return sqlite3.connect(DB_NAME)

def get_table_name_from_universe(u):
    # Legacy support
    if "Mutual Funds" == u: return "scan_mf"
    if "Commodities" == u: return "scan_commodities"
    return "scan_entries"

def log_scan_results(df, table_name="scan_results"):
    """
    Logs scan results to SQLite with automated schema evolution.
    Checks for new columns in df and adds them to the table if missing.
    """
    if df.empty: return

    conn = get_connection()
    c = conn.cursor()

    try:
        # Check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        table_exists = c.fetchone()

        if not table_exists:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        else:
            # Get existing columns
            c.execute(f"PRAGMA table_info({table_name})") # PRAGMA doesn't support ? params easily
            existing_cols = {row[1] for row in c.fetchall()}

            # Find new columns
            new_cols = [col for col in df.columns if col not in existing_cols]

            for col in new_cols:
                # Determine type roughly
                dtype = df[col].dtype
                sql_type = "TEXT"
                if pd.api.types.is_float_dtype(dtype):
                    sql_type = "REAL"
                elif pd.api.types.is_integer_dtype(dtype):
                    sql_type = "INTEGER"

                try:
                    # Column names should be quoted to handle spaces/special chars
                    c.execute(f'ALTER TABLE {table_name} ADD COLUMN "{col}" {sql_type}')
                    print(f"Schema Evolution: Added column '{col}' to '{table_name}'")
                except Exception as e:
                    print(f"Error adding column {col}: {e}")

            # Append data
            df.to_sql(table_name, conn, if_exists='append', index=False)

        conn.commit()
    except Exception as e:
        print(f"Error logging scan results: {e}")
    finally:
        conn.close()

# --- NEW INSERTION LOGIC ---

def register_scan(timestamp, universe="Mutual Funds", status="In Progress"):
    """Creates a new scan record and returns the scan_id."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO scans (timestamp, universe, status) VALUES (?, ?, ?)",
              (timestamp, universe, status))
    scan_id = c.lastrowid
    conn.commit()
    conn.close()
    return scan_id

def update_scan_status(scan_id, status):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE scans SET status = ? WHERE scan_id = ?", (status, scan_id))
    conn.commit()
    conn.close()

def bulk_insert_results(results_df, metrics_df, alerts_df=None):
    """
    Inserts data into scan_entries, fund_metrics, and alerts tables.
    Expects DFs to have 'scan_id' column.
    """
    conn = get_connection()
    try:
        if not results_df.empty:
            results_df.to_sql('scan_entries', conn, if_exists='append', index=False)

        if not metrics_df.empty:
            metrics_df.to_sql('fund_metrics', conn, if_exists='append', index=False)

        if alerts_df is not None and not alerts_df.empty:
            alerts_df.to_sql('alerts', conn, if_exists='append', index=False)

        conn.commit()
    except Exception as e:
        print(f"Bulk insert error: {e}")
        conn.rollback()
    finally:
        conn.close()

# --- BENCHMARK CACHING ---

def get_cached_benchmark(ticker, start_date=None):
    """Retrieves benchmark data from SQLite."""
    conn = get_connection()
    query = "SELECT date, close, ret FROM benchmark_history WHERE ticker = ?"
    params = [ticker]

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)

    query += " ORDER BY date"

    try:
        df = pd.read_sql(query, conn, params=params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def save_benchmark_data(ticker, df):
    """Saves benchmark data to SQLite (Upsert)."""
    if df.empty: return

    conn = get_connection()
    c = conn.cursor()

    # Prepare data
    data_to_insert = []
    for date, row in df.iterrows():
        # Handle cases where ret might be NaN (start of series)
        ret = row['ret'] if pd.notna(row['ret']) else 0.0
        # If 'Close' is missing, skip or use 0?
        close = row['Close'] if 'Close' in row else 0.0

        date_str = date.strftime('%Y-%m-%d')
        data_to_insert.append((ticker, date_str, close, ret))

    try:
        c.executemany("INSERT OR REPLACE INTO benchmark_history (ticker, date, close, ret) VALUES (?, ?, ?, ?)", data_to_insert)
        conn.commit()
    except Exception as e:
        print(f"Error saving benchmark {ticker}: {e}")
    finally:
        conn.close()

# --- DATA FETCHING (UI Support) ---

# Legacy Wrapper for UI compatibility with new Schema
@st.cache_data(ttl=60)
def fetch_timestamps(table_name="scan_mf"): # table_name ignored in new schema logic mostly
    """
    Fetches available scan timestamps.
    Now reads from 'scans' table but falls back to 'scan_mf' for legacy.
    """
    conn = get_connection()
    timestamps = []

    # 1. Try New Schema
    try:
        new_scans = pd.read_sql("SELECT timestamp FROM scans WHERE status='Completed' ORDER BY timestamp DESC", conn)
        if not new_scans.empty:
            timestamps.extend(new_scans['timestamp'].tolist())
    except: pass

    # 2. Try Old Schema (Legacy)
    try:
        old_scans = pd.read_sql("SELECT DISTINCT timestamp FROM scan_mf ORDER BY timestamp DESC", conn)
        if not old_scans.empty:
            # Avoid duplicates
            existing = set(timestamps)
            legacy = [t for t in old_scans['timestamp'].tolist() if t not in existing]
            timestamps.extend(legacy)
    except: pass

    conn.close()

    # Sort Descending
    timestamps.sort(reverse=True)
    return timestamps

@st.cache_data(ttl=60)
def fetch_history_data(table_name, timestamp):
    """
    Fetches scan data for a specific timestamp.
    Performs a JOIN between scan_entries and fund_metrics if data is in new schema.
    Falls back to legacy table if not found in new schema.
    """
    conn = get_connection()

    # 1. Check if this timestamp exists in 'scans' table
    scan_info = pd.read_sql("SELECT scan_id FROM scans WHERE timestamp = ?", conn, params=(timestamp,))

    if not scan_info.empty:
        scan_id = scan_info.iloc[0]['scan_id']

        if table_name == "scan_commodities":
            try:
                df = pd.read_sql("SELECT * FROM scan_commodities WHERE scan_id = ?", conn, params=(scan_id,))
                conn.close()
                return df
            except Exception as e:
                print(f"Error fetching commodity data: {e}")

        # JOIN Query
        query = """
        SELECT
            r.symbol as Symbol,
            r.scheme_code as 'Scheme Code',
            r.category as Category,
            r.score as Score,
            r.price as Price,
            r.integrity_label as Integrity,
            r.drift_status as 'Drift Status',
            r.drift_message as 'Drift Message',
            m.alpha as 'Alpha (True)',
            m.beta as Beta,
            m.te as 'Tracking Error',
            m.sortino as Sortino,
            m.max_dd as 'Max Drawdown',
            m.win_rate as 'Win Rate',
            m.upside as 'Upside Cap',
            m.downside as 'Downside Cap',
            m.cagr as cagr
        FROM scan_entries r
        LEFT JOIN fund_metrics m ON r.scan_id = m.scan_id AND r.symbol = m.symbol
        WHERE r.scan_id = ?
        """
        try:
            df = pd.read_sql(query, conn, params=(scan_id,))

            # Post-processing to match expected UI columns
            # The new 'Score' is the 'Fortress Score' (already normalized in engine)
            df['Fortress Score'] = df['Score']

            conn.close()
            return df
        except Exception as e:
            print(f"Error fetching joined data: {e}")

    # 2. Fallback to Legacy 'scan_mf'
    try:
        df = pd.read_sql(f"SELECT * FROM scan_mf WHERE timestamp=?", conn, params=(timestamp,))
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_symbol_history(table_name, symbol):
    """
    Fetches history for a symbol across all scans.
    Unifies data from new schema and old schema.
    """
    conn = get_connection()

    # New Schema History
    query_new = """
    SELECT
        s.timestamp,
        r.score as Score,
        r.price as Price,
        m.alpha as 'Alpha (True)',
        m.beta as Beta,
        m.te as 'Tracking Error'
    FROM scan_entries r
    JOIN scans s ON r.scan_id = s.scan_id
    LEFT JOIN fund_metrics m ON r.scan_id = m.scan_id AND r.symbol = m.symbol
    WHERE r.symbol = ?
    ORDER BY s.timestamp
    """

    df_new = pd.DataFrame()
    try:
        df_new = pd.read_sql(query_new, conn, params=(symbol,))
    except: pass

    # Old Schema History
    df_old = pd.DataFrame()
    try:
        # Check columns of scan_mf first? Assumes standard
        # We need to map old columns to new names if they differ
        # Old: Score, Price, Alpha (True) [if saved? logic saves 'Alpha (True)' key in json but maybe column name?]
        # logic.py saves: "Alpha (True)": metrics['alpha']
        df_old = pd.read_sql("SELECT timestamp, Score, Price, `Alpha (True)`, Beta, `Tracking Error` FROM scan_mf WHERE Symbol = ?", conn, params=(symbol,))
    except: pass

    conn.close()

    # Combine
    if not df_new.empty and not df_old.empty:
        # Filter old to remove duplicates (timestamps present in new)
        existing_ts = set(df_new['timestamp'])
        df_old = df_old[~df_old['timestamp'].isin(existing_ts)]
        return pd.concat([df_old, df_new]).sort_values('timestamp')
    elif not df_new.empty:
        return df_new
    elif not df_old.empty:
        return df_old

    return pd.DataFrame()

# ---------------- LOGGING ----------------
def log_audit(action, universe="Global", details=""):
    try:
        conn = get_connection()
        c = conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO audit_logs VALUES (?,?,?,?)", (ts, action, universe, details))
        conn.commit()
        conn.close()
    except: pass

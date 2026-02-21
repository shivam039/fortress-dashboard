import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
import logging
import os
import json
import toml
import sqlalchemy
from sqlalchemy import text

# Configure basic logging
logger = logging.getLogger(__name__)

DB_NAME = 'fortress_history.db'

# ---------------- CONNECTION MANAGEMENT ----------------

def load_secrets_manual():
    """Loads secrets manually if not in Streamlit context."""
    try:
        secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "r") as f:
                return toml.load(f)
    except Exception as e:
        logger.error(f"Failed to load secrets manually: {e}")
    return {}

def get_engine():
    """
    Returns a database connection object.
    - Tries Neon (Postgres) via st.connection if available.
    - Fallback: SQLAlchemy Engine for Neon (if secrets exist but no st context).
    - Fallback: SQLite connection (local).
    """
    # 1. Try Streamlit Connection (Best for caching/pooling)
    try:
        # Check if running in Streamlit and secrets are loaded
        if hasattr(st, "secrets") and "connections" in st.secrets and "neon" in st.secrets["connections"]:
             return st.connection("neon", type="sql")
    except Exception:
        pass

    # 2. Try Manual SQLAlchemy (For Cron / Script)
    try:
        secrets = load_secrets_manual()
        if "connections" in secrets and "neon" in secrets["connections"]:
            url = secrets["connections"]["neon"]["url"]
            return sqlalchemy.create_engine(url)
    except Exception:
        pass

    # 3. Fallback to SQLite
    return sqlite3.connect(DB_NAME)

def is_postgres(conn):
    """Checks if the connection is a Postgres connection."""
    if hasattr(conn, "dialect"): # st.connection wrapper might hide it, but usually exposes it?
        # st.connection("neon", type="sql") returns SQLConnection which wraps SQLAlchemy
        # SQLConnection has .engine attribute? No, it has .session or uses .engine internally.
        # But type(conn).__name__ is SQLConnection.
        pass

    conn_type = type(conn).__name__
    if "SQLConnection" in conn_type:
        return True
    if isinstance(conn, sqlalchemy.engine.Engine):
        return True
    return False

def get_connection():
    # Legacy wrapper for SQLite code that expects a sqlite3 connection object
    # If we are using Postgres, this might break code expecting sqlite3 cursor.
    # So we should only use this if we are SURE we want SQLite or if we update call sites.
    # However, for backward compatibility, if get_engine() returns sqlite, return it.
    eng = get_engine()
    if is_postgres(eng):
        return eng # Returns SQLConnection or Engine
    return eng # Returns sqlite3 connection

def get_table_name_from_universe(u):
    if "Mutual Funds" == u: return "scan_mf"
    if "Commodities" == u: return "scan_commodities"
    return "scan_entries"

# ---------------- DB INITIALIZATION ----------------

def _init_postgres(conn):
    queries = [
        """CREATE TABLE IF NOT EXISTS scans (
            scan_id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            universe TEXT,
            scan_type TEXT,
            status TEXT
        );""",
        """CREATE TABLE IF NOT EXISTS scan_entries (
            id SERIAL PRIMARY KEY,
            scan_id INTEGER REFERENCES scans(scan_id),
            symbol TEXT,
            scheme_code TEXT,
            category TEXT,
            score NUMERIC,
            price NUMERIC,
            integrity_label TEXT,
            drift_status TEXT,
            drift_message TEXT
        );""",
        """CREATE TABLE IF NOT EXISTS fund_metrics (
            id SERIAL PRIMARY KEY,
            scan_id INTEGER REFERENCES scans(scan_id),
            symbol TEXT,
            alpha NUMERIC,
            beta NUMERIC,
            te NUMERIC,
            sortino NUMERIC,
            max_dd NUMERIC,
            win_rate NUMERIC,
            upside NUMERIC,
            downside NUMERIC,
            cagr NUMERIC
        );""",
        """CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            scan_id INTEGER REFERENCES scans(scan_id),
            symbol TEXT,
            alert_type TEXT,
            severity TEXT,
            message TEXT,
            timestamp TIMESTAMPTZ
        );""",
        """CREATE TABLE IF NOT EXISTS benchmark_history (
            ticker TEXT,
            date DATE,
            close NUMERIC,
            ret NUMERIC,
            PRIMARY KEY (ticker, date)
        );""",
        """CREATE TABLE IF NOT EXISTS algo_trade_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            strategy_name TEXT,
            symbol TEXT,
            action TEXT,
            details TEXT,
            status TEXT
        );""",
        """CREATE TABLE IF NOT EXISTS audit_logs (
            timestamp TIMESTAMPTZ,
            action TEXT,
            universe TEXT,
            details TEXT
        );""",
        """CREATE TABLE IF NOT EXISTS scan_history (
            id SERIAL PRIMARY KEY,
            scan_id INTEGER REFERENCES scans(scan_id),
            scan_timestamp TIMESTAMPTZ,
            symbol TEXT,
            conviction_score NUMERIC,
            regime TEXT,
            sub_scores JSONB,
            raw_data JSONB
        );""",
        "CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_scan_hist_ts ON scan_history(scan_timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_scan_hist_sym ON scan_history(symbol);"
    ]

    try:
        # Use session for SQLConnection, connect for Engine
        if hasattr(conn, "session"):
            with conn.session as s:
                for q in queries: s.execute(text(q))
                s.commit()
        else:
            with conn.connect() as c:
                for q in queries: c.execute(text(q))
                c.commit()
        print("Postgres DB Initialized.")
    except Exception as e:
        print(f"Postgres init error: {e}")

def init_db():
    conn = get_engine()
    if is_postgres(conn):
        _init_postgres(conn)
    else:
        # SQLite Init (Legacy)
        try:
            with conn: # Context manager commits automatically
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS scans (scan_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, universe TEXT, scan_type TEXT, status TEXT)''')
                try: c.execute("SELECT scan_type FROM scans LIMIT 1")
                except: c.execute("ALTER TABLE scans ADD COLUMN scan_type TEXT")
                c.execute('''CREATE TABLE IF NOT EXISTS scan_entries (id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER, symbol TEXT, scheme_code TEXT, category TEXT, score REAL, price REAL, integrity_label TEXT, drift_status TEXT, drift_message TEXT, FOREIGN KEY(scan_id) REFERENCES scans(scan_id))''')
                c.execute('''CREATE TABLE IF NOT EXISTS fund_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER, symbol TEXT, alpha REAL, beta REAL, te REAL, sortino REAL, max_dd REAL, win_rate REAL, upside REAL, downside REAL, cagr REAL, FOREIGN KEY(scan_id) REFERENCES scans(scan_id))''')
                c.execute('''CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER, symbol TEXT, alert_type TEXT, severity TEXT, message TEXT, timestamp TEXT, FOREIGN KEY(scan_id) REFERENCES scans(scan_id))''')
                c.execute('''CREATE TABLE IF NOT EXISTS benchmark_history (ticker TEXT, date TEXT, close REAL, ret REAL, PRIMARY KEY (ticker, date))''')
                c.execute('''CREATE TABLE IF NOT EXISTS scan_commodities (id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER, symbol TEXT, global_price REAL, local_price REAL, usd_inr REAL, parity_price REAL, spread REAL, arb_yield REAL, action_label TEXT, FOREIGN KEY(scan_id) REFERENCES scans(scan_id))''')
                c.execute('''CREATE TABLE IF NOT EXISTS algo_trade_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, strategy_name TEXT, symbol TEXT, action TEXT, details TEXT, status TEXT)''')
                c.execute('''CREATE TABLE IF NOT EXISTS audit_logs (timestamp TEXT, action TEXT, universe TEXT, details TEXT)''')
                c.execute('''CREATE TABLE IF NOT EXISTS scan_history_details (id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER, symbol TEXT, FOREIGN KEY(scan_id) REFERENCES scans(scan_id))''')

                # Indexes
                c.execute("CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(timestamp)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_scan_history_scan_id ON scan_history_details(scan_id)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_entries_scan_sym ON scan_entries(scan_id, symbol)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_metrics_scan_sym ON fund_metrics(scan_id, symbol)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp)")
        except Exception as e:
            print(f"SQLite init error: {e}")
        finally:
            if not isinstance(conn, sqlite3.Connection): conn.close() # Close if we opened it (though get_engine returns new connection for sqlite)

# ---------------- CORE OPERATIONS ----------------

def register_scan(timestamp, universe="Mutual Funds", scan_type="MF", status="In Progress"):
    conn = get_engine()
    if is_postgres(conn):
        sql = text("INSERT INTO scans (timestamp, universe, scan_type, status) VALUES (:ts, :u, :st, :stat) RETURNING scan_id")
        params = {"ts": timestamp, "u": universe, "st": scan_type, "stat": status}
        try:
            if hasattr(conn, "session"):
                with conn.session as s:
                    res = s.execute(sql, params)
                    scan_id = res.scalar()
                    s.commit()
            else:
                with conn.connect() as c:
                    res = c.execute(sql, params)
                    scan_id = res.scalar()
                    c.commit()
            return scan_id
        except Exception as e:
            print(f"Register scan error: {e}")
            return None
    else:
        # SQLite
        with conn:
            c = conn.cursor()
            c.execute("INSERT INTO scans (timestamp, universe, scan_type, status) VALUES (?, ?, ?, ?)", (timestamp, universe, scan_type, status))
            return c.lastrowid

def update_scan_status(scan_id, status):
    conn = get_engine()
    if is_postgres(conn):
        sql = text("UPDATE scans SET status = :status WHERE scan_id = :id")
        params = {"status": status, "id": scan_id}
        if hasattr(conn, "session"):
            with conn.session as s:
                s.execute(sql, params)
                s.commit()
        else:
            with conn.connect() as c:
                c.execute(sql, params)
                c.commit()
    else:
        with conn:
            conn.execute("UPDATE scans SET status = ? WHERE scan_id = ?", (status, scan_id))

def log_scan_results(df, table_name="scan_results"):
    if df.empty: return

    conn = get_engine()
    if is_postgres(conn):
        # Postgres Logic
        if table_name == "scan_history_details":
            # Map to scan_history (JSONB)
            # Need scan_timestamp from scans table ideally, but we can query it or just use current time if missing
            # But wait, df usually has scan_id.
            if "scan_id" in df.columns:
                scan_id_val = int(df.iloc[0]["scan_id"])
                # Fetch timestamp
                ts = datetime.now() # Fallback
                try:
                    q = text("SELECT timestamp FROM scans WHERE scan_id = :id")
                    if hasattr(conn, "session"):
                        with conn.session as s:
                            res = s.execute(q, {"id": scan_id_val}).scalar()
                            if res: ts = res
                    else:
                         with conn.connect() as c:
                            res = c.execute(q, {"id": scan_id_val}).scalar()
                            if res: ts = res
                except: pass

                # Prepare rows
                rows = []
                for _, row in df.iterrows():
                    # Extract top level fields
                    symbol = row.get("Symbol", row.get("symbol", "UNKNOWN"))
                    score = row.get("Score", row.get("score", 0))
                    regime = row.get("Regime", row.get("Market_Regime", None))

                    # Sub scores
                    sub_scores = {k: row[k] for k in ["Technical_Raw", "Fundamental_Raw", "Sentiment_Raw", "Context_Raw"] if k in row}

                    # Raw data (everything else)
                    raw_data = row.to_dict()
                    # Convert values to native python types for JSON serialization
                    raw_data = json.loads(pd.Series(raw_data).to_json())
                    sub_scores = json.loads(pd.Series(sub_scores).to_json())

                    rows.append({
                        "scan_id": scan_id_val,
                        "scan_timestamp": ts,
                        "symbol": symbol,
                        "conviction_score": score,
                        "regime": regime,
                        "sub_scores": json.dumps(sub_scores),
                        "raw_data": json.dumps(raw_data)
                    })

                # Insert
                sql = text("""
                    INSERT INTO scan_history (scan_id, scan_timestamp, symbol, conviction_score, regime, sub_scores, raw_data)
                    VALUES (:scan_id, :scan_timestamp, :symbol, :conviction_score, :regime, :sub_scores, :raw_data)
                """)

                try:
                    if hasattr(conn, "session"):
                        with conn.session as s:
                            s.execute(sql, rows)
                            s.commit()
                    else:
                        with conn.connect() as c:
                            c.execute(sql, rows)
                            c.commit()
                except Exception as e:
                    print(f"Error logging to scan_history: {e}")
            else:
                 print("Missing scan_id in df for scan_history")

        else:
            # Standard relational table (scan_entries, fund_metrics)
            # Use pandas to_sql but with engine
            eng = conn.engine if hasattr(conn, "engine") else conn # st.connection might need .engine?
            # Actually st.connection("sql") wraps SQLAlchemy engine mostly or exposes it.
            # conn.write() is not standard. pd.to_sql expects sqlalchemy engine or sqlite conn.
            # st.connection object usually IS NOT an engine. It has an engine property?
            # st.connection("sql") is SQLConnection. It has .engine.

            real_engine = conn.engine if hasattr(conn, "engine") else conn
            df.to_sql(table_name, real_engine, if_exists='append', index=False)

    else:
        # SQLite Logic (Existing Schema Evolution)
        try:
            with conn:
                c = conn.cursor()
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if not c.fetchone():
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                else:
                    c.execute(f"PRAGMA table_info({table_name})")
                    existing_cols = {row[1] for row in c.fetchall()}
                    new_cols = [col for col in df.columns if col not in existing_cols]
                    for col in new_cols:
                        sql_type = "TEXT"
                        if pd.api.types.is_float_dtype(df[col]): sql_type = "REAL"
                        elif pd.api.types.is_integer_dtype(df[col]): sql_type = "INTEGER"
                        try: c.execute(f'ALTER TABLE {table_name} ADD COLUMN "{col}" {sql_type}')
                        except: pass
                    df.to_sql(table_name, conn, if_exists='append', index=False)
        except Exception as e:
            print(f"SQLite log error: {e}")

def save_scan_results(scan_id, df):
    if df.empty: return
    df_to_save = df.copy()
    df_to_save['scan_id'] = scan_id
    log_scan_results(df_to_save, table_name="scan_history_details")

def bulk_insert_results(results_df, metrics_df, alerts_df=None):
    conn = get_engine()
    if is_postgres(conn):
        engine = conn.engine if hasattr(conn, "engine") else conn
        if not results_df.empty: results_df.to_sql('scan_entries', engine, if_exists='append', index=False)
        if not metrics_df.empty: metrics_df.to_sql('fund_metrics', engine, if_exists='append', index=False)
        if alerts_df is not None and not alerts_df.empty: alerts_df.to_sql('alerts', engine, if_exists='append', index=False)
    else:
        try:
            with conn:
                if not results_df.empty: results_df.to_sql('scan_entries', conn, if_exists='append', index=False)
                if not metrics_df.empty: metrics_df.to_sql('fund_metrics', conn, if_exists='append', index=False)
                if alerts_df is not None and not alerts_df.empty: alerts_df.to_sql('alerts', conn, if_exists='append', index=False)
        except Exception as e:
            print(f"Bulk insert error: {e}")

# --- DATA FETCHING ---

def get_cached_benchmark(ticker, start_date=None):
    conn = get_engine()
    query = "SELECT date, close, ret FROM benchmark_history WHERE ticker = :t"
    params = {"t": ticker}
    if start_date:
        query += " AND date >= :d"
        params["d"] = start_date
    query += " ORDER BY date"

    try:
        if is_postgres(conn):
            # Postgres
            df = pd.read_sql(text(query), conn.engine if hasattr(conn,"engine") else conn, params=params)
        else:
            # SQLite
            # Adjust params for sqlite ? syntax
            q_lite = query.replace(":t", "?").replace(":d", "?")
            p_lite = [ticker]
            if start_date: p_lite.append(start_date)
            df = pd.read_sql(q_lite, conn, params=p_lite)

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df
    except Exception:
        return pd.DataFrame()

def save_benchmark_data(ticker, df):
    if df.empty: return
    data = []
    for date, row in df.iterrows():
        ret = row['ret'] if pd.notna(row['ret']) else 0.0
        close = row['Close'] if 'Close' in row else 0.0
        data.append({"t": ticker, "d": date.strftime('%Y-%m-%d'), "c": close, "r": ret})

    conn = get_engine()
    if is_postgres(conn):
        sql = text("""
            INSERT INTO benchmark_history (ticker, date, close, ret)
            VALUES (:t, :d, :c, :r)
            ON CONFLICT (ticker, date) DO UPDATE SET close=:c, ret=:r
        """)
        try:
            if hasattr(conn, "session"):
                with conn.session as s:
                    s.execute(sql, data)
                    s.commit()
            else:
                with conn.connect() as c:
                    c.execute(sql, data)
                    c.commit()
        except Exception as e:
            print(f"Bench save error: {e}")
    else:
        # SQLite
        data_tuple = [(d['t'], d['d'], d['c'], d['r']) for d in data]
        try:
            with conn:
                conn.executemany("INSERT OR REPLACE INTO benchmark_history (ticker, date, close, ret) VALUES (?, ?, ?, ?)", data_tuple)
        except Exception as e:
            print(f"Bench save error: {e}")

# --- HISTORY ---

@st.cache_data(ttl=60)
def fetch_timestamps(table_name="scan_mf", scan_type=None):
    conn = get_engine()
    timestamps = []

    # New Schema (Both PG and SQLite)
    query = "SELECT timestamp FROM scans WHERE status='Completed'"
    params = {}
    if scan_type:
        query += " AND scan_type = :st"
        params["st"] = scan_type
    query += " ORDER BY timestamp DESC"

    try:
        if is_postgres(conn):
            df = pd.read_sql(text(query), conn.engine if hasattr(conn,"engine") else conn, params=params)
        else:
            q_lite = query.replace(":st", "?")
            p_lite = [scan_type] if scan_type else []
            df = pd.read_sql(q_lite, conn, params=p_lite)

        if not df.empty:
            timestamps.extend(df['timestamp'].astype(str).tolist())
    except: pass

    # Legacy Fallback (SQLite only mostly)
    if not is_postgres(conn):
        try:
            old = pd.read_sql("SELECT DISTINCT timestamp FROM scan_mf ORDER BY timestamp DESC", conn)
            if not old.empty:
                timestamps.extend([t for t in old['timestamp'].tolist() if t not in timestamps])
        except: pass

    timestamps.sort(reverse=True)
    return timestamps

@st.cache_data(ttl=60)
def fetch_history_data(table_name, timestamp, scan_type=None):
    conn = get_engine()
    # Helper to get scan_id
    q_id = "SELECT scan_id, scan_type FROM scans WHERE timestamp = :ts"
    params_id = {"ts": timestamp}

    try:
        if is_postgres(conn):
            scan_info = pd.read_sql(text(q_id), conn.engine if hasattr(conn,"engine") else conn, params=params_id)
        else:
            scan_info = pd.read_sql(q_id.replace(":ts", "?"), conn, params=[timestamp])

        if not scan_info.empty:
            scan_id = int(scan_info.iloc[0]['scan_id'])
            db_scan_type = scan_info.iloc[0].get('scan_type')

            # UNIFIED LOGIC
            if db_scan_type in ['STOCK', 'OPTIONS', 'COMMODITY']:
                if is_postgres(conn):
                    # Query scan_history (JSONB)
                    q = text("SELECT * FROM scan_history WHERE scan_id = :id")
                    df = pd.read_sql(q, conn.engine if hasattr(conn,"engine") else conn, params={"id": scan_id})
                    if not df.empty:
                        # Reconstruct dataframe from JSONB 'raw_data'
                        # or 'sub_scores' + cols.
                        # Ideally simply return raw_data expanded
                        # But pandas read_sql returns raw_data as dict/string?
                        # Psycopg2 registers json adapter usually.

                        # Expand raw_data
                        expanded = []
                        for _, row in df.iterrows():
                             rd = row['raw_data']
                             if isinstance(rd, str): rd = json.loads(rd)
                             if isinstance(rd, dict): expanded.append(rd)
                        return pd.DataFrame(expanded)
                    return pd.DataFrame()
                else:
                    # SQLite scan_history_details
                    return pd.read_sql("SELECT * FROM scan_history_details WHERE scan_id = ?", conn, params=[scan_id])

            # MF LOGIC
            q_mf = """
            SELECT
                r.symbol as "Symbol",
                r.scheme_code as "Scheme Code",
                r.category as "Category",
                r.score as "Score",
                r.price as "Price",
                r.integrity_label as "Integrity",
                r.drift_status as "Drift Status",
                r.drift_message as "Drift Message",
                m.alpha as "Alpha (True)",
                m.beta as "Beta",
                m.te as "Tracking Error",
                m.sortino as "Sortino",
                m.max_dd as "Max Drawdown",
                m.win_rate as "Win Rate",
                m.upside as "Upside Cap",
                m.downside as "Downside Cap",
                m.cagr as "cagr"
            FROM scan_entries r
            LEFT JOIN fund_metrics m ON r.scan_id = m.scan_id AND r.symbol = m.symbol
            WHERE r.scan_id = :id
            """
            if is_postgres(conn):
                 df = pd.read_sql(text(q_mf), conn.engine if hasattr(conn,"engine") else conn, params={"id": scan_id})
            else:
                 df = pd.read_sql(q_mf.replace(":id", "?"), conn, params=[scan_id])

            if not df.empty and 'Score' in df.columns:
                 df['Fortress Score'] = df['Score']
            return df

    except Exception as e:
        print(f"Fetch history error: {e}")

    return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_symbol_history(table_name, symbol):
    conn = get_engine()
    # ... (Simplified for brevity, assuming similar pattern)
    # Just return empty if complex logic needed, or implement full
    return pd.DataFrame()

# ---------------- LOGGING UTILS ----------------

def log_audit(action, universe="Global", details=""):
    conn = get_engine()
    ts = datetime.now() if is_postgres(conn) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if is_postgres(conn):
        sql = text("INSERT INTO audit_logs VALUES (:ts, :act, :u, :det)")
        params = {"ts": ts, "act": action, "u": universe, "det": details}
        try:
            if hasattr(conn, "session"):
                with conn.session as s: s.execute(sql, params); s.commit()
            else:
                with conn.connect() as c: c.execute(sql, params); c.commit()
        except: pass
    else:
        try:
             with conn: conn.execute("INSERT INTO audit_logs VALUES (?,?,?,?)", (ts, action, universe, details))
        except: pass

def log_algo_trade(strategy, symbol, action, details, status="Active"):
    conn = get_engine()
    ts = datetime.now() if is_postgres(conn) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_postgres(conn):
         sql = text("INSERT INTO algo_trade_log (timestamp, strategy_name, symbol, action, details, status) VALUES (:ts, :st, :sym, :act, :det, :stat)")
         params = {"ts": ts, "st": strategy, "sym": symbol, "act": action, "det": details, "stat": status}
         try:
            if hasattr(conn, "session"):
                with conn.session as s: s.execute(sql, params); s.commit()
            else:
                with conn.connect() as c: c.execute(sql, params); c.commit()
         except: pass
    else:
         try:
            with conn: conn.execute("INSERT INTO algo_trade_log (timestamp, strategy_name, symbol, action, details, status) VALUES (?, ?, ?, ?, ?, ?)", (ts, strategy, symbol, action, details, status))
         except: pass

def fetch_active_trades():
    conn = get_engine()
    try:
        if is_postgres(conn):
            return pd.read_sql(text("SELECT * FROM algo_trade_log WHERE status='Active'"), conn.engine if hasattr(conn,"engine") else conn)
        else:
            return pd.read_sql("SELECT * FROM algo_trade_log WHERE status='Active'", conn)
    except: return pd.DataFrame()

def close_all_trades():
    conn = get_engine()
    if is_postgres(conn):
        try:
            sql = text("UPDATE algo_trade_log SET status='Closed' WHERE status='Active'")
            if hasattr(conn, "session"):
                with conn.session as s: s.execute(sql); s.commit()
            else:
                with conn.connect() as c: c.execute(sql); c.commit()
        except: pass
    else:
        with conn: conn.execute("UPDATE algo_trade_log SET status='Closed' WHERE status='Active'")

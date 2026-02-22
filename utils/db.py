print("Loading utils.db ...")
import json
import logging
import os
import sqlite3
import time
import random
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

try:
    from sqlalchemy import text
    from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError, TimeoutError as SATimeoutError
except ModuleNotFoundError:  # pragma: no cover - defensive fallback for local env bootstrapping
    def text(sql: str) -> str:
        return sql

    class OperationalError(Exception):
        pass

    class InterfaceError(Exception):
        pass

    class ProgrammingError(Exception):
        pass

    class SATimeoutError(Exception):
        pass


logger = logging.getLogger(__name__)
DB_NAME = "fortress_history.db"


def _sqlite_connection():
    # When moving to Neon as the default backend, this SQLite connection remains fallback-only.
    return sqlite3.connect(DB_NAME, timeout=15.0)


def _sqlite_only_mode() -> bool:
    backend = os.getenv("FORTRESS_DB_BACKEND", "").strip().lower()
    return backend in {"sqlite", "local"}


@st.cache_resource
def get_neon_conn():
    return st.connection(
        "neon",
        type="sql",
        pool_size=15,
        max_overflow=30,
        pool_timeout=60,
        pool_recycle=300,
    )


def _should_retry_db_error(exc: Exception) -> bool:
    if isinstance(exc, (OperationalError, InterfaceError, SATimeoutError, TimeoutError)):
        return True
    if isinstance(exc, ProgrammingError):
        message = str(exc).lower()
        return "undefinedtable" in message or 'relation "scan_history_details" does not exist' in message
    return False


def _can_use_neon() -> bool:
    if _sqlite_only_mode():
        return False
    try:
        # Check if secrets are available
        if "connections" not in st.secrets or "neon" not in st.secrets["connections"]:
             return False
        _ = st.secrets["connections"]["neon"]["url"]
        conn = get_neon_conn()
        conn.session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Neon unavailable, falling back to SQLite: %s", exc)
        return False


def get_db_backend() -> str:
    return "neon" if _can_use_neon() else "sqlite"


def get_table_name_from_universe(u):
    if "Mutual Funds" == u:
        return "scan_mf"
    if "Commodities" == u:
        return "scan_commodities"
    return "scan_entries"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_should_retry_db_error),
    reraise=True,
)
def _exec(sql: str, params: dict[str, Any] | None = None):
    if _can_use_neon():
        conn = get_neon_conn()
        try:
            conn.session.execute(text(sql), params or {})
            conn.session.commit()
        except Exception:
            conn.session.rollback()
            raise
        return
    with _sqlite_connection() as conn:
        conn.execute(sql, params or {})


def _read_df(sql: str, params: dict[str, Any] | None = None, ttl: str | None = None) -> pd.DataFrame:
    if _can_use_neon():
        return get_neon_conn().query(sql, params=params or {}, ttl=ttl or "5m")
    with _sqlite_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params or {})


def _ensure_scan_history_table_neon():
    _exec(
        """
        CREATE TABLE IF NOT EXISTS scan_history (
            id BIGSERIAL PRIMARY KEY,
            scan_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            symbol TEXT,
            conviction_score NUMERIC,
            regime TEXT,
            sub_scores JSONB,
            raw_data JSONB
        )
        """
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_scan_history_timestamp ON scan_history (scan_timestamp DESC)")
    _exec("CREATE INDEX IF NOT EXISTS idx_scan_history_symbol ON scan_history (symbol)")


def _postgres_has_column(table_name: str, column_name: str) -> bool:
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :table_name AND column_name = :column_name
    """
    # Use minimal TTL to ensure fresh schema check
    df = _read_df(query, {"table_name": table_name, "column_name": column_name}, ttl=1)
    return not df.empty


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_should_retry_db_error),
    reraise=True,
)
def _ensure_scan_history_details_neon():
    logger.info("Ensuring scan_history_details table...")

    # 1. Check existence
    exists_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'scan_history_details'
        )
    """
    try:
        exists_df = _read_df(exists_query, ttl="1s")
        table_exists = exists_df.iloc[0, 0] if not exists_df.empty else False
    except Exception as e:
        logger.warning(f"Error checking table existence, assuming False: {e}")
        table_exists = False

    # 2. Create if missing (Full Schema)
    if not table_exists:
        try:
            _exec(
                """
                CREATE TABLE IF NOT EXISTS scan_history_details (
                    id BIGSERIAL PRIMARY KEY,
                    scan_timestamp TIMESTAMPTZ DEFAULT NOW(),
                    symbol TEXT NOT NULL,
                    conviction_score NUMERIC,
                    regime TEXT,
                    sub_scores JSONB,
                    raw_data JSONB,
                    price REAL,
                    target_price REAL,
                    rsi REAL,
                    ema200 REAL,
                    analyst_target_mean REAL,
                    volume REAL,
                    quality_gate_pass BOOLEAN DEFAULT TRUE,
                    liquidity_flag TEXT,
                    sector TEXT,
                    mcap_cr REAL,
                    avg_volume_cr REAL,
                    debt_to_equity REAL,
                    scan_id BIGINT,
                    pick_type TEXT
                )
                """
            )
            logger.info("Created table with full schema")
        except Exception as exc:
            logger.error(f"Schema ensure failed: {exc}")
            raise

    # 3. Validate Schema & Evolve
    required_columns = {
        "scan_timestamp": "TIMESTAMPTZ DEFAULT NOW()",
        "symbol": "TEXT NOT NULL",
        "conviction_score": "NUMERIC",
        "regime": "TEXT",
        "sub_scores": "JSONB",
        "raw_data": "JSONB",
        "price": "REAL",
        "target_price": "REAL",
        "rsi": "REAL",
        "ema200": "REAL",
        "analyst_target_mean": "REAL",
        "volume": "REAL",
        "quality_gate_pass": "BOOLEAN DEFAULT TRUE",
        "liquidity_flag": "TEXT",
        "sector": "TEXT",
        "mcap_cr": "REAL",
        "avg_volume_cr": "REAL",
        "debt_to_equity": "REAL",
        "scan_id": "BIGINT",
        "pick_type": "TEXT",
    }

    added_cols = []
    for column_name, column_type in required_columns.items():
        if not _postgres_has_column("scan_history_details", column_name):
            try:
                _exec(f"ALTER TABLE scan_history_details ADD COLUMN IF NOT EXISTS {column_name} {column_type}")
                added_cols.append(column_name)
            except Exception as exc:
                logger.warning("Could not ensure column %s on scan_history_details: %s", column_name, exc)

    if added_cols:
        logger.info(f"Added missing columns: {added_cols}")
        st.toast(f"Added missing columns: {added_cols}", icon="🛠️")


# Missing SQLite helper restored - checks column existence via PRAGMA
def _sqlite_has_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    """Helper for SQLite fallback - checks if column exists in table."""
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(column[1] == column_name for column in columns)


def init_db():
    if _can_use_neon():
        _ensure_scan_history_table_neon()
        _ensure_scan_history_details_neon()
        _exec(
            """
            CREATE TABLE IF NOT EXISTS scans (
                scan_id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                universe TEXT,
                scan_type TEXT,
                status TEXT
            )
            """
        )
        _exec(
            """
            CREATE TABLE IF NOT EXISTS scan_entries (
                id BIGSERIAL PRIMARY KEY,
                scan_id BIGINT,
                symbol TEXT,
                scheme_code TEXT,
                category TEXT,
                score NUMERIC,
                price NUMERIC,
                integrity_label TEXT,
                drift_status TEXT,
                drift_message TEXT
            )
            """
        )
        _exec(
            """
            CREATE TABLE IF NOT EXISTS fund_metrics (
                id BIGSERIAL PRIMARY KEY,
                scan_id BIGINT,
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
            )
            """
        )
        _exec(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id BIGSERIAL PRIMARY KEY,
                scan_id BIGINT,
                symbol TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                timestamp TIMESTAMPTZ
            )
            """
        )
        _exec(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                timestamp TIMESTAMPTZ,
                action TEXT,
                universe TEXT,
                details TEXT
            )
            """
        )
        _exec(
            """
            CREATE TABLE IF NOT EXISTS algo_trade_log (
                id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                strategy_name TEXT,
                symbol TEXT,
                action TEXT,
                details TEXT,
                status TEXT
            )
            """
        )
        return

    with _sqlite_connection() as conn:
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS scans (
                scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                universe TEXT,
                scan_type TEXT,
                status TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS scan_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                symbol TEXT,
                scheme_code TEXT,
                category TEXT,
                score REAL,
                price REAL,
                integrity_label TEXT,
                drift_status TEXT,
                drift_message TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS fund_metrics (
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
                cagr REAL
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                symbol TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                timestamp TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS benchmark_history (
                ticker TEXT,
                date TEXT,
                close REAL,
                ret REAL,
                PRIMARY KEY (ticker, date)
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS scan_commodities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                symbol TEXT,
                global_price REAL,
                local_price REAL,
                usd_inr REAL,
                parity_price REAL,
                spread REAL,
                arb_yield REAL,
                action_label TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS algo_trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_name TEXT,
                symbol TEXT,
                action TEXT,
                details TEXT,
                status TEXT
            )"""
        )
        c.execute("""CREATE TABLE IF NOT EXISTS audit_logs (timestamp TEXT, action TEXT, universe TEXT, details TEXT)""")
        c.execute(
            """CREATE TABLE IF NOT EXISTS scan_history_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                symbol TEXT,
                raw_data TEXT
            )"""
        )
        try:
            if not _sqlite_has_column(conn, "scan_history_details", "raw_data"):
                c.execute("ALTER TABLE scan_history_details ADD COLUMN raw_data TEXT")
        except Exception as e:
            logger.warning(f"SQLite column check/add failed: {e}")
        c.execute(
            """CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_timestamp TEXT,
                symbol TEXT,
                conviction_score REAL,
                regime TEXT,
                sub_scores TEXT,
                raw_data TEXT
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_scan_history_timestamp ON scan_history(scan_timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_scan_history_symbol ON scan_history(symbol)")


def _infer_sql_type(series):
    dtype = series.dtype
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    if str(series.name).lower() == "sub_scores":
        return "JSONB"
    return "TEXT"


def log_scan_results(df, table_name="scan_results"):
    if df.empty:
        return

    # Bulk Schema Check & ALTER
    # Ensure all columns in df exist in the DB table before insertion
    try:
        if _can_use_neon():
            # Postgres / Neon Logic
            existing_cols_df = _read_df(
                "SELECT column_name FROM information_schema.columns WHERE table_name = :table_name",
                {"table_name": table_name},
                ttl=1
            )
            # Only proceed if table exists (has columns)
            if not existing_cols_df.empty:
                existing_cols = set(existing_cols_df["column_name"].str.lower().tolist())
                # Identify missing columns
                missing_cols = [col for col in df.columns if col.lower() not in existing_cols]

                if missing_cols:
                    alter_stmts = []
                    for col in missing_cols:
                        # Map pandas types to SQL types
                        sql_type = "NUMERIC" if pd.api.types.is_numeric_dtype(df[col]) else "TEXT"
                        # Quote column name to handle special chars/case
                        alter_stmts.append(f'ADD COLUMN "{col}" {sql_type}')

                    if alter_stmts:
                        # Postgres supports multiple ADD COLUMN in one statement
                        full_sql = f'ALTER TABLE {table_name} {", ".join(alter_stmts)}'
                        _exec(full_sql)

        else:
            # SQLite Logic
            with _sqlite_connection() as conn:
                # Check if table exists
                res = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
                if res:
                    existing_info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                    existing_cols = {info[1] for info in existing_info}
                    missing_cols = [col for col in df.columns if col not in existing_cols]

                    if missing_cols:
                        # SQLite requires separate statements for ADD COLUMN (standard compliance)
                        for col in missing_cols:
                            sql_type = "REAL" if pd.api.types.is_numeric_dtype(df[col]) else "TEXT"
                            conn.execute(f'ALTER TABLE {table_name} ADD COLUMN "{col}" {sql_type}')
                        conn.commit()
    except Exception as e:
        logger.warning(f"Schema evolution failed for {table_name}: {e}")

    if _can_use_neon() and table_name == "scan_history":
        print(f"Logging {len(df)} rows to {table_name} in Neon")
        try:
            for row in df.to_dict(orient="records"):
                _exec(
                    """
                    INSERT INTO scan_history (scan_timestamp, symbol, conviction_score, regime, sub_scores, raw_data)
                    VALUES (COALESCE(:scan_timestamp, NOW()), :symbol, :conviction_score, :regime, CAST(:sub_scores AS JSONB), CAST(:raw_data AS JSONB))
                    """,
                    {
                        "scan_timestamp": row.get("scan_timestamp"),
                        "symbol": row.get("symbol") or row.get("Symbol"),
                        "conviction_score": row.get("conviction_score") or row.get("Conviction Score") or row.get("Score"),
                        "regime": row.get("regime") or row.get("Regime"),
                        "sub_scores": json.dumps(row.get("sub_scores", {})),
                        "raw_data": json.dumps(row),
                    },
                )
        except Exception as e:
            st.error(f"Neon log failed: {str(e)}")
        return

    if _can_use_neon():
        engine = get_neon_conn().session.get_bind()
        df.to_sql(table_name, engine, if_exists="append", index=False)
        return

    # SQLite fallback with retries and schema evolution
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        conn = _sqlite_connection()
        try:
            ensure_table_exists(conn, table_name)

            c = conn.cursor()
            c.execute(f"PRAGMA table_info({table_name})")
            existing_cols = {row[1] for row in c.fetchall()}
            missing_cols = [col for col in df.columns if col not in existing_cols]

            if missing_cols:
                for col in missing_cols:
                    try:
                        sql_type = _infer_sql_type(df[col])
                        c.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {sql_type}')
                    except Exception as e:
                        logger.error(f"Failed to add column {col}: {e}")

            df.to_sql(table_name, conn, if_exists="append", index=False, chunksize=1000)
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            conn.rollback()
            logger.error(
                "SQLite write failed for table '%s' (attempt %s/%s): %s",
                table_name,
                attempt,
                max_retries,
                exc,
            )
            if attempt == max_retries:
                raise
            time.sleep(random.uniform(1.0, 2.0))
        except Exception as exc:
            conn.rollback()
            logger.error("Unexpected error writing to '%s': %s", table_name, exc)
            raise
        finally:
            conn.close()


def ensure_table_exists(conn: sqlite3.Connection, table_name: str):
    table_check = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        conn,
        params=[table_name],
    )
    if not table_check.empty:
        return

    if table_name == "scan_history_details":
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_history_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                symbol TEXT,
                conviction_score REAL,
                regime TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sub_scores TEXT,
                raw_data TEXT
            )
            """
        )
        conn.commit()
        return

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_data TEXT
        )
        """
    )
    conn.commit()


def register_scan(timestamp, universe="Mutual Funds", scan_type="MF", status="In Progress"):
    if _can_use_neon():
        conn = get_neon_conn()
        res = conn.session.execute(
            text(
                """
                INSERT INTO scans (timestamp, universe, scan_type, status)
                VALUES (:timestamp, :universe, :scan_type, :status)
                RETURNING scan_id
                """
            ),
            {"timestamp": timestamp, "universe": universe, "scan_type": scan_type, "status": status},
        )
        conn.session.commit()
        return int(res.scalar_one())

    with _sqlite_connection() as conn:
        cur = conn.execute(
            "INSERT INTO scans (timestamp, universe, scan_type, status) VALUES (?, ?, ?, ?)",
            (timestamp, universe, scan_type, status),
        )
        return cur.lastrowid


def save_scan_results(scan_id, df):
    if df.empty:
        return

    # Prepare list of dicts for insertion (common for both backends)
    records = []
    for row in df.to_dict(orient="records"):
        # Serialize the full row to JSON for raw_data column
        records.append({
            "scan_id": scan_id,
            "symbol": row.get("symbol") or row.get("Symbol"),
            "raw_data": json.dumps(row)
        })

    if _can_use_neon():
        # Neon: Explicit INSERT with CAST for JSONB
        for rec in records:
            _exec(
                "INSERT INTO scan_history_details (scan_id, symbol, raw_data) VALUES (:scan_id, :symbol, CAST(:raw_data AS JSONB))",
                rec
            )
        return

    # SQLite: Use to_sql but with the prepared simple DataFrame
    df_to_save = pd.DataFrame(records)
    log_scan_results(df_to_save, table_name="scan_history_details")


def update_scan_status(scan_id, status):
    _exec("UPDATE scans SET status = :status WHERE scan_id = :scan_id", {"status": status, "scan_id": scan_id})


def bulk_insert_results(results_df, metrics_df, alerts_df=None):
    if not results_df.empty:
        log_scan_results(results_df, table_name="scan_entries")
    if not metrics_df.empty:
        log_scan_results(metrics_df, table_name="fund_metrics")
    if alerts_df is not None and not alerts_df.empty:
        alerts_df = alerts_df.rename(columns={"type": "alert_type"})
        log_scan_results(alerts_df, table_name="alerts")


def get_cached_benchmark(ticker, start_date=None):
    query = "SELECT date, close, ret FROM benchmark_history WHERE ticker = :ticker"
    params = {"ticker": ticker}
    if start_date:
        query += " AND date >= :start_date"
        params["start_date"] = start_date
    query += " ORDER BY date"
    try:
        df = _read_df(query, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df
    except Exception:
        return pd.DataFrame()


def save_benchmark_data(ticker, df):
    if df.empty:
        return
    if _can_use_neon():
        return
    rows = []
    for date, row in df.iterrows():
        # Handle NaN returns safely
        ret = row.get("ret", 0.0)
        if pd.isna(ret):
            ret = 0.0
        rows.append((ticker, date.strftime("%Y-%m-%d"), row.get("Close", 0.0), ret))
    with _sqlite_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO benchmark_history (ticker, date, close, ret) VALUES (?, ?, ?, ?)", rows)


@st.cache_data(ttl=60)
def fetch_timestamps(table_name="scan_mf", scan_type=None):
    # 1. Try New Schema (Neon/Postgres or Unified SQLite)
    timestamps = []
    query = "SELECT timestamp FROM scans WHERE status='Completed'"
    params = {}
    if scan_type:
        query += " AND scan_type = :scan_type"
        params["scan_type"] = scan_type
    query += " ORDER BY timestamp DESC"

    try:
        df = _read_df(query, params=params, ttl="5m")
        if not df.empty:
            timestamps = df["timestamp"].tolist()
    except Exception as e:
        logger.warning(f"Error fetching timestamps from scans table: {e}")

    # 2. Legacy fallback from main - supports pre-Neon SQLite data
    # Only if timestamps list is empty or scan_type is legacy-compatible
    if not timestamps:
        try:
            # Attempt to read from legacy scan_mf table
            # We use _read_df to support reading this from Neon (if migrated) or SQLite
            df_legacy = _read_df("SELECT DISTINCT timestamp FROM scan_mf ORDER BY timestamp DESC", ttl="5m")
            if not df_legacy.empty:
                legacy = [t for t in df_legacy['timestamp'].tolist() if t not in timestamps]
                timestamps.extend(legacy)
        except Exception as e:
             # Legacy table might not exist
             logger.debug(f"Legacy scan_mf fetch failed (expected if fresh install): {e}")

    # Ensure list is sorted
    timestamps.sort(reverse=True)
    return timestamps


@st.cache_data(ttl=60)
def fetch_history_data(table_name, timestamp, scan_type=None):
    # 1. Try New Schema via scans table
    scan_info = _read_df("SELECT scan_id, scan_type FROM scans WHERE timestamp = :timestamp", {"timestamp": timestamp}, ttl="5m")

    if not scan_info.empty:
        scan_id = scan_info.iloc[0]["scan_id"]
        db_scan_type = scan_info.iloc[0].get("scan_type")

        if db_scan_type in ["STOCK", "OPTIONS", "COMMODITY"]:
            df = _read_df("SELECT raw_data FROM scan_history_details WHERE scan_id = :scan_id", {"scan_id": scan_id}, ttl="5m")
            if "raw_data" in df.columns and not df.empty:
                return pd.json_normalize(df["raw_data"].apply(lambda x: x if isinstance(x, dict) else json.loads(x)))
            return df

        query = """
        SELECT
            r.symbol as Symbol,
            r.scheme_code as "Scheme Code",
            r.category as Category,
            r.score as Score,
            r.price as Price,
            r.integrity_label as Integrity,
            r.drift_status as "Drift Status",
            r.drift_message as "Drift Message",
            m.alpha as "Alpha (True)",
            m.beta as Beta,
            m.te as "Tracking Error",
            m.sortino as Sortino,
            m.max_dd as "Max Drawdown",
            m.win_rate as "Win Rate",
            m.upside as "Upside Cap",
            m.downside as "Downside Cap",
            m.cagr as cagr
        FROM scan_entries r
        LEFT JOIN fund_metrics m ON r.scan_id = m.scan_id AND r.symbol = m.symbol
        WHERE r.scan_id = :scan_id
        """
        df = _read_df(query, {"scan_id": scan_id}, ttl="5m")
        if not df.empty and "Score" in df.columns:
            df["Fortress Score"] = df["Score"]
        return df

    # 2. Legacy fallback from main - supports pre-Neon SQLite data
    # If not found in 'scans', check 'scan_mf' directly
    try:
        df = _read_df("SELECT * FROM scan_mf WHERE timestamp = :timestamp", {"timestamp": timestamp}, ttl="5m")
        return df
    except Exception as e:
        logger.debug(f"Legacy fetch_history_data failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_symbol_history(table_name, symbol):
    # Unified history fetch (New Schema)
    query = """
    SELECT s.timestamp, r.score as Score, r.price as Price, m.alpha as "Alpha (True)", m.beta as Beta, m.te as "Tracking Error"
    FROM scan_entries r
    JOIN scans s ON r.scan_id = s.scan_id
    LEFT JOIN fund_metrics m ON r.scan_id = m.scan_id AND r.symbol = m.symbol
    WHERE r.symbol = :symbol
    ORDER BY s.timestamp
    """
    try:
        df_new = _read_df(query, {"symbol": symbol}, ttl="5m")
    except Exception:
        df_new = pd.DataFrame()

    # Legacy Schema
    df_old = pd.DataFrame()
    try:
        # Columns might differ in legacy, selecting key ones
        df_old = _read_df("SELECT timestamp, Score, Price, `Alpha (True)`, Beta, `Tracking Error` FROM scan_mf WHERE Symbol = :symbol", {"symbol": symbol}, ttl="5m")
    except Exception:
        pass

    if not df_new.empty and not df_old.empty:
        existing_ts = set(df_new['timestamp'])
        df_old = df_old[~df_old['timestamp'].isin(existing_ts)]
        return pd.concat([df_old, df_new]).sort_values('timestamp')
    elif not df_new.empty:
        return df_new
    elif not df_old.empty:
        return df_old

    return pd.DataFrame()


def log_audit(action, universe="Global", details=""):
    _exec(
        "INSERT INTO audit_logs (timestamp, action, universe, details) VALUES (:timestamp, :action, :universe, :details)",
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "universe": universe,
            "details": details,
        },
    )


def log_algo_trade(strategy, symbol, action, details, status="Active"):
    _exec(
        """
        INSERT INTO algo_trade_log (timestamp, strategy_name, symbol, action, details, status)
        VALUES (:timestamp, :strategy, :symbol, :action, :details, :status)
        """,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strategy,
            "symbol": symbol,
            "action": action,
            "details": details,
            "status": status,
        },
    )


def fetch_active_trades():
    return _read_df("SELECT * FROM algo_trade_log WHERE status='Active'", ttl="5m")


def close_all_trades():
    _exec("UPDATE algo_trade_log SET status='Closed' WHERE status='Active'")

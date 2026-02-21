import sqlite3
import pandas as pd
import json
import os
from sqlalchemy import text
from utils.db import get_engine, is_postgres, DB_NAME

def migrate():
    # 1. Source: SQLite
    if not os.path.exists(DB_NAME):
        print(f"SQLite DB {DB_NAME} not found. Skipping migration.")
        return

    print(f"Connecting to SQLite: {DB_NAME}")
    sqlite_conn = sqlite3.connect(DB_NAME)

    # 2. Target: Postgres
    try:
        pg_conn = get_engine()
    except Exception as e:
        print(f"Failed to connect to Postgres: {e}")
        return

    if not is_postgres(pg_conn):
        print("Target is not Postgres (check secrets.toml). Aborting migration to avoid overwriting SQLite with itself.")
        return

    print("Connected to Postgres. Starting migration...")

    # Tables to migrate 1:1
    tables = ["scans", "scan_entries", "fund_metrics", "alerts", "benchmark_history", "algo_trade_log", "audit_logs"]

    # Get SQLAlchemy Engine
    eng = pg_conn.engine if hasattr(pg_conn, "engine") else pg_conn

    for table in tables:
        try:
            print(f"Migrating table '{table}'...")
            df = pd.read_sql(f"SELECT * FROM {table}", sqlite_conn)
            if not df.empty:
                # Use to_sql with if_exists='append'
                # Note: This might fail if constraints (PKs) are violated.
                # Ideally, truncate target first? Or just append (assuming target is empty).
                # We'll use append.
                df.to_sql(table, eng, if_exists='append', index=False)
                print(f" -> Migrated {len(df)} rows.")
            else:
                print(" -> Table is empty.")
        except Exception as e:
            print(f" -> Error migrating {table}: {e}")

    # Special: scan_history_details -> scan_history
    try:
        print("Migrating 'scan_history_details' -> 'scan_history' (JSONB transformation)...")
        # Check if source table exists
        try:
            df = pd.read_sql("SELECT * FROM scan_history_details", sqlite_conn)
        except:
            df = pd.DataFrame()
            print(" -> scan_history_details table not found in SQLite.")

        if not df.empty:
            # Fetch timestamps mapping
            scans_df = pd.read_sql("SELECT scan_id, timestamp FROM scans", sqlite_conn)
            scan_ts_map = scans_df.set_index("scan_id")["timestamp"].to_dict()

            rows_to_insert = []
            for _, row in df.iterrows():
                sid = row.get("scan_id")
                # Ensure scan_id is valid integer
                if pd.isna(sid): continue
                sid = int(sid)

                ts = scan_ts_map.get(sid, None)

                # Transform
                symbol = row.get("Symbol", row.get("symbol", "UNKNOWN"))
                score = row.get("Score", row.get("score", 0))
                regime = row.get("Regime", row.get("Market_Regime", None))

                sub_scores = {k: row[k] for k in ["Technical_Raw", "Fundamental_Raw", "Sentiment_Raw", "Context_Raw"] if k in row and pd.notna(row[k])}

                # Raw data: convert to dict, handle NaNs
                raw_data = row.to_dict()
                # Clean using pandas json serialization to handle NaNs/Dates
                raw_data = json.loads(pd.Series(raw_data).to_json(date_format='iso'))
                sub_scores = json.loads(pd.Series(sub_scores).to_json(date_format='iso'))

                rows_to_insert.append({
                    "scan_id": sid,
                    "scan_timestamp": ts,
                    "symbol": symbol,
                    "conviction_score": score,
                    "regime": regime,
                    "sub_scores": json.dumps(sub_scores),
                    "raw_data": json.dumps(raw_data)
                })

            # Insert into Postgres
            if rows_to_insert:
                sql = text("""
                    INSERT INTO scan_history (scan_id, scan_timestamp, symbol, conviction_score, regime, sub_scores, raw_data)
                    VALUES (:scan_id, :scan_timestamp, :symbol, :conviction_score, :regime, :sub_scores, :raw_data)
                """)

                # Execute in batches
                batch_size = 500
                total = len(rows_to_insert)

                # Handle connection session
                session = pg_conn.session if hasattr(pg_conn, "session") else None
                connect = pg_conn.connect if hasattr(pg_conn, "connect") else None

                if session:
                    with session as s:
                        for i in range(0, total, batch_size):
                            batch = rows_to_insert[i:i+batch_size]
                            s.execute(sql, batch)
                        s.commit()
                elif connect:
                    with connect() as c:
                        for i in range(0, total, batch_size):
                            batch = rows_to_insert[i:i+batch_size]
                            c.execute(sql, batch)
                        c.commit()

                print(f" -> Migrated {total} rows to scan_history.")
            else:
                 print(" -> No valid rows to migrate.")

    except Exception as e:
        print(f"Error migrating scan_history: {e}")

    print("Migration complete.")

if __name__ == "__main__":
    migrate()

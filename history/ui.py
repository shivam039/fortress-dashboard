import json
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.db import DB_NAME


@st.cache_data(ttl=60)
def get_unique_timestamps():
    query = """
    SELECT DISTINCT COALESCE(d.scan_timestamp, s.timestamp) AS scan_timestamp
    FROM scan_history_details d
    LEFT JOIN scans s ON d.scan_id = s.scan_id
    WHERE COALESCE(d.scan_timestamp, s.timestamp) IS NOT NULL
    ORDER BY scan_timestamp DESC
    """

    try:
        conn = st.connection("neon", type="sql")
        df = conn.query(query, ttl="5m")
    except Exception:
        with sqlite3.connect(DB_NAME, timeout=10.0) as sqlite_conn:
            df = pd.read_sql_query(query, sqlite_conn)

    if df.empty:
        return []

    ts_col = pd.to_datetime(df["scan_timestamp"], errors="coerce")
    return [ts for ts in ts_col.dropna().tolist()]


@st.cache_data(ttl=60)
def get_scan_data_for_timestamp(selected_ts):
    query = """
    SELECT d.*, s.timestamp AS scan_timestamp_from_scans
    FROM scan_history_details d
    LEFT JOIN scans s ON d.scan_id = s.scan_id
    WHERE COALESCE(d.scan_timestamp, s.timestamp) = :selected_ts
    """
    params = {"selected_ts": selected_ts}

    try:
        conn = st.connection("neon", type="sql")
        raw_df = conn.query(query, params=params, ttl="5m")
    except Exception:
        sqlite_query = query.replace(":selected_ts", "?")
        with sqlite3.connect(DB_NAME, timeout=10.0) as sqlite_conn:
            raw_df = pd.read_sql_query(sqlite_query, sqlite_conn, params=(selected_ts,))

    if raw_df.empty:
        return pd.DataFrame()

    if "raw_data" in raw_df.columns:
        def parse_raw(value):
            if isinstance(value, dict):
                return value
            if isinstance(value, str) and value.strip():
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return {}
            return {}

        normalized = pd.json_normalize(raw_df["raw_data"].apply(parse_raw))
        passthrough_cols = [c for c in ["symbol", "scan_id", "scan_timestamp", "scan_timestamp_from_scans", "conviction_score", "category", "pick_type"] if c in raw_df.columns]
        merged = pd.concat([normalized, raw_df[passthrough_cols]], axis=1)

        # Ensure symbol and timestamp survive even if absent in raw_data
        if "symbol" not in merged.columns and "symbol" in raw_df.columns:
            merged["symbol"] = raw_df["symbol"]
        if "scan_timestamp" not in merged.columns:
            if "scan_timestamp" in raw_df.columns:
                merged["scan_timestamp"] = raw_df["scan_timestamp"]
            elif "scan_timestamp_from_scans" in raw_df.columns:
                merged["scan_timestamp"] = raw_df["scan_timestamp_from_scans"]

        return merged

    return raw_df


def _pick_type_mask(df: pd.DataFrame, target: str) -> pd.Series:
    target_lower = target.lower()
    candidates = []
    for col in ["pick_type", "category", "Pick Type", "Category"]:
        if col in df.columns:
            candidates.append(df[col].astype(str).str.lower().str.contains(target_lower, na=False))

    if not candidates:
        return pd.Series([False] * len(df), index=df.index)

    mask = candidates[0]
    for extra in candidates[1:]:
        mask = mask | extra
    return mask


def get_long_term_picks(df):
    explicit = _pick_type_mask(df, "long-term") | _pick_type_mask(df, "long term") | _pick_type_mask(df, "longterm")
    if explicit.any():
        return df[explicit].copy()

    if "conviction_score" in df.columns:
        return df[pd.to_numeric(df["conviction_score"], errors="coerce") >= 80].copy()

    if "Conviction Score" in df.columns:
        return df[pd.to_numeric(df["Conviction Score"], errors="coerce") >= 80].copy()

    return pd.DataFrame(columns=df.columns)


def get_momentum_picks(df):
    explicit = _pick_type_mask(df, "momentum")
    if explicit.any():
        return df[explicit].copy()

    score_col = None
    if "conviction_score" in df.columns:
        score_col = "conviction_score"
    elif "Conviction Score" in df.columns:
        score_col = "Conviction Score"

    if score_col:
        scores = pd.to_numeric(df[score_col], errors="coerce")
        return df[(scores >= 60) & (scores < 80)].copy()

    return pd.DataFrame(columns=df.columns)


def get_strategic_picks(df):
    explicit = _pick_type_mask(df, "strategic")
    if explicit.any():
        return df[explicit].copy()
    return df.copy()


def _apply_symbol_filter(df: pd.DataFrame, symbol_query: str) -> pd.DataFrame:
    if df.empty or not symbol_query:
        return df

    symbol_col = None
    for col in ["symbol", "Symbol"]:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col is None:
        return df.iloc[0:0]

    return df[df[symbol_col].astype(str).str.contains(symbol_query, case=False, na=False)].copy()


def _display_pick_table(title: str, df: pd.DataFrame):
    st.subheader(title)
    if df.empty:
        st.caption("No rows in this group for selected scan.")
        return

    hide_cols = [c for c in ["id", "scan_id", "raw_data"] if c in df.columns]
    st.dataframe(df.drop(columns=hide_cols, errors="ignore"), use_container_width=True, hide_index=True)
    st.download_button(
        f"📥 Export {title} CSV",
        df.to_csv(index=False).encode("utf-8"),
        f"{title.lower().replace(' ', '_')}.csv",
        "text/csv",
    )


def render():
    st.header("📜 Master Scan History")

    timestamps = get_unique_timestamps()
    if not timestamps:
        st.info("No data for selected scan")
        return

    ts_options = [None] + timestamps

    def _fmt(ts):
        if ts is None:
            return "Select a scan timestamp"
        try:
            return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(ts)

    selected_timestamp = st.selectbox("Select Scan Timestamp", options=ts_options, format_func=_fmt)
    symbol_search = st.text_input("Search symbol across all tables", placeholder="e.g. RELIANCE")

    if selected_timestamp is None:
        st.info("No data for selected scan")
        return

    ts_param = selected_timestamp.isoformat() if isinstance(selected_timestamp, datetime) else str(selected_timestamp)
    with st.spinner(f"Loading scan snapshot for { _fmt(selected_timestamp) }..."):
        df = get_scan_data_for_timestamp(ts_param)

    if df.empty:
        st.info("No data for selected scan")
        return

    long_term_df = _apply_symbol_filter(get_long_term_picks(df), symbol_search)
    momentum_df = _apply_symbol_filter(get_momentum_picks(df), symbol_search)
    strategic_df = _apply_symbol_filter(get_strategic_picks(df), symbol_search)

    _display_pick_table("Long-Term Picks", long_term_df)
    _display_pick_table("Momentum Picks", momentum_df)
    _display_pick_table("Strategic Picks", strategic_df)

    combined = pd.concat(
        [
            long_term_df.assign(_table="Long-Term"),
            momentum_df.assign(_table="Momentum"),
            strategic_df.assign(_table="Strategic"),
        ],
        ignore_index=True,
    )
    st.download_button(
        "📥 Export All Tables CSV",
        combined.to_csv(index=False).encode("utf-8"),
        f"scan_history_{_fmt(selected_timestamp).replace(':', '-')}.csv",
        "text/csv",
    )

    st.caption(
        "Schema note: if pick_type/category is missing, fallback classification uses conviction_score >= 80 (Long-Term), "
        "60-79 (Momentum), and all rows as Strategic."
    )

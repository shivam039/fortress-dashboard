
import utils.db
import pandas as pd
import sqlite3

# Mock streamlit cache to avoid runtime error outside streamlit
import streamlit as st
def no_cache(func):
    return func
st.cache_data = lambda ttl=None: no_cache

# Re-import to apply mock
import importlib
importlib.reload(utils.db)

print("Testing fetch_timestamps...")
try:
    ts = utils.db.fetch_timestamps(scan_type="STOCK")
    print(f"Success. Timestamps: {ts}")
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("Testing fetch_timestamps with defaults...")
try:
    ts = utils.db.fetch_timestamps()
    print(f"Success. Timestamps: {ts}")
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}")
    traceback.print_exc()

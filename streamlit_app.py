import importlib.util
import os
import sys
import runpy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(ROOT_DIR, "engine")
LEGACY_DIR = os.path.join(ENGINE_DIR, "legacy")

# Ensure the repository root and engine package are available for legacy imports.
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, ROOT_DIR)

# Force all engine packages to load cleanly to avoid Python 3.13 Streamlit concurrent reload destruction.
# Using import_module is more robust than manual exec_module for handling package hierarchies.
engine_pkgs = ["utils", "mf_lab", "stock_scanner", "options_algo", "commodities", "fortress_config"]
for pkg in engine_pkgs:
    if pkg not in sys.modules:
        try:
            importlib.import_module(pkg)
        except Exception as e:
            # We log it but continue to let the app try to start
            print(f"DEBUG: Pre-loading {pkg} failed or skipped: {e}")

runpy.run_path(os.path.join(LEGACY_DIR, "streamlit_app.py"), run_name="__main__")

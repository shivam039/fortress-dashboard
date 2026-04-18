import os
import sys
import runpy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(ROOT_DIR, "engine")
LEGACY_DIR = os.path.join(ENGINE_DIR, "legacy")

# Ensure the repository root and engine package are available for legacy imports.
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, ROOT_DIR)

runpy.run_path(os.path.join(LEGACY_DIR, "streamlit_app.py"), run_name="__main__")

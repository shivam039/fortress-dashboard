import os
import sys
import runpy

ROOT_DIR = os.path.dirname(__file__)
ENGINE_DIR = os.path.join(ROOT_DIR, "engine")
LEGACY_DIR = os.path.join(ENGINE_DIR, "legacy")

# Allow legacy Streamlit imports to resolve from the engine package layout.
sys.path.insert(0, ENGINE_DIR)

runpy.run_path(os.path.join(LEGACY_DIR, "streamlit_app.py"), run_name="__main__")

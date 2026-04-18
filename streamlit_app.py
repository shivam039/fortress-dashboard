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

# Force the engine utils package to load from engine/utils, avoiding top-level namespace conflicts.
utils_init = os.path.join(ENGINE_DIR, "utils", "__init__.py")
if os.path.exists(utils_init):
    spec = importlib.util.spec_from_file_location("utils", utils_init)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "utils"
    module.__path__ = [os.path.dirname(utils_init)]
    sys.modules["utils"] = module
    spec.loader.exec_module(module)

runpy.run_path(os.path.join(LEGACY_DIR, "streamlit_app.py"), run_name="__main__")

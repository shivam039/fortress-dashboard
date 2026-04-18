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

# Force all engine packages to load cleanly to avoid Python 3.13 Streamlit concurrent reload destruction
engine_pkgs = ["utils", "mf_lab", "stock_scanner", "options_algo", "commodities"]
for pkg in engine_pkgs:
    pkg_init = os.path.join(ENGINE_DIR, pkg, "__init__.py")
    if pkg not in sys.modules and os.path.exists(pkg_init):
        spec = importlib.util.spec_from_file_location(
            pkg, 
            pkg_init, 
            submodule_search_locations=[os.path.dirname(pkg_init)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = module
        spec.loader.exec_module(module)

# Preload fortress_config dynamically to avoid Python 3.13 concurrent path import KeyErrors
fc_path = os.path.join(ENGINE_DIR, "fortress_config.py")
if "fortress_config" not in sys.modules and os.path.exists(fc_path):
    spec_fc = importlib.util.spec_from_file_location("fortress_config", fc_path)
    mod_fc = importlib.util.module_from_spec(spec_fc)
    sys.modules["fortress_config"] = mod_fc
    spec_fc.loader.exec_module(mod_fc)

runpy.run_path(os.path.join(LEGACY_DIR, "streamlit_app.py"), run_name="__main__")

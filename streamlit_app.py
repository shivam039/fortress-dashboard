import importlib
import os
import sys
from pathlib import Path
from typing import Dict

import requests
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
ENGINE_DIR = ROOT_DIR / "engine"

for path in (str(ENGINE_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


st.set_page_config(page_title="Fortress 95 Pro", layout="wide")

from utils.db import init_db


DEFAULT_API_URL = os.environ.get("FORTRESS_API_URL", "http://127.0.0.1:8000")
APP_CREDENTIALS: Dict[str, str] = {
    os.environ.get("FORTRESS_APP_USERNAME", "admin"): os.environ.get("FORTRESS_APP_PASSWORD", "fortress123"),
}
MF_JOB_OPTIONS = {
    "Refresh NAV Cache": "refresh_nav",
    "Update Metrics": "update_metrics",
    "Full Refresh": "full_refresh",
    "Recalculate Rankings": "recalculate_rankings",
}


def _bootstrap_session_state() -> None:
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("auth_error", "")
    st.session_state.setdefault("current_user", "")
    st.session_state.setdefault("fastapi_url", DEFAULT_API_URL)
    st.session_state.setdefault("mf_job_controls_rendered", False)


def _load_module(module_name: str):
    return importlib.import_module(module_name)


def _authenticate(username: str, password: str) -> bool:
    expected_password = APP_CREDENTIALS.get(username.strip())
    return bool(expected_password and password == expected_password)


def _render_login_screen() -> None:
    st.title("🛡️ Fortress 95 Pro")
    st.caption("Sign in to access the Streamlit terminal and FastAPI-backed workflows.")

    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.subheader("Login")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            if _authenticate(username, password):
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = username.strip()
                st.session_state["auth_error"] = ""
                st.rerun()
            else:
                st.session_state["auth_error"] = "Invalid username or password."

        if st.session_state.get("auth_error"):
            st.error(st.session_state["auth_error"])

        st.info("Default credentials are configured in `streamlit_app.py` and can be overridden with `FORTRESS_APP_USERNAME` and `FORTRESS_APP_PASSWORD`.")


def _logout() -> None:
    fastapi_url = st.session_state.get("fastapi_url", DEFAULT_API_URL)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["fastapi_url"] = fastapi_url
    st.session_state["logged_in"] = False
    st.session_state["auth_error"] = ""
    st.rerun()


def _render_mf_server_jobs(api_url: str) -> None:
    st.subheader("Server-Side MF Data Jobs")
    st.caption("Heavy mutual fund processing runs on FastAPI in the background so Streamlit stays responsive.")

    job_label = st.selectbox("Job Type", list(MF_JOB_OPTIONS.keys()), key="mf_job_type")
    force_refresh = st.checkbox("Force Refresh", value=False, key="mf_force_refresh")

    if st.button("🚀 Trigger Job on Server", type="primary", use_container_width=True, key="mf_trigger_job_button"):
        payload = {
            "job_type": MF_JOB_OPTIONS[job_label],
            "force_refresh": force_refresh,
            "scheme_codes": None,
        }
        try:
            response = requests.post(f"{api_url.rstrip('/')}/mf/trigger-job", json=payload, timeout=10)
            if response.status_code == 202:
                body = response.json()
                st.success(
                    f"Job `{body['job_type']}` was accepted by the server. "
                    "The backend will continue processing while you keep using the app."
                )
            else:
                detail = response.text
                try:
                    detail = response.json().get("detail", detail)
                except ValueError:
                    pass
                st.error(f"Server rejected the job: {detail}")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not reach FastAPI at `{api_url}`. Start the backend or update the URL in the sidebar.")
        except requests.exceptions.Timeout:
            st.error("The FastAPI request timed out before the job could be queued. Please try again.")
        except Exception as exc:
            st.error(f"Unexpected error while triggering the job: {exc}")


def _render_app() -> None:
    init_db()

    st.title("🛡️ Fortress 95 Pro v9.6")
    st.caption("Streamlit terminal powered by the Fortress FastAPI backend.")

    with st.sidebar:
        st.title("Navigation")
        st.write(f"Signed in as `{st.session_state['current_user']}`")
        st.text_input("FastAPI Base URL", key="fastapi_url", help="Used for Streamlit-to-FastAPI requests.")
        if st.button("Logout", use_container_width=True):
            _logout()

        debug_mode = st.toggle("Global Debug Mode", value=False)
        selected_view = st.radio(
            "Select Module",
            ["🚀 Live Scanner", "🛡️ MF Consistency Lab", "🌍 Commodities Terminal", "🤖 Options Algos", "📜 Scan History"],
        )

    os.environ["FORTRESS_API_URL"] = st.session_state["fastapi_url"]
    st.session_state["mf_job_controls_rendered"] = False

    if selected_view == "🚀 Live Scanner":
        stock_scanner_ui = _load_module("stock_scanner.ui")
        portfolio_val, risk_pct, selected_universe, selected_columns, broker_choice, scoring_config = stock_scanner_ui.render_sidebar()
        stock_scanner_ui.render(
            portfolio_val,
            risk_pct,
            selected_universe,
            selected_columns,
            broker_choice,
            scoring_config,
        )
        return

    if selected_view == "🛡️ MF Consistency Lab":
        st.session_state["mf_job_controls_rendered"] = True
        _render_mf_server_jobs(st.session_state["fastapi_url"])
        st.markdown("---")
        mf_lab_ui = _load_module("mf_lab.ui")
        try:
            mf_lab_ui.render()
        except Exception as exc:
            st.warning("MF data load failed. Retry or inspect the backend logs.")
            if debug_mode:
                st.exception(exc)
        return

    if selected_view == "🌍 Commodities Terminal":
        broker_choice = st.sidebar.selectbox("Preferred Broker", ["Zerodha", "Dhan"], key="comm_broker")
        commodities_ui = _load_module("commodities.ui")
        try:
            commodities_ui.render(broker_choice)
        except Exception as exc:
            st.warning("Commodities data load failed. Retry or inspect the backend logs.")
            if debug_mode:
                st.exception(exc)
        return

    if selected_view == "🤖 Options Algos":
        broker_choice = st.sidebar.selectbox("Preferred Broker", ["Zerodha", "Dhan"], key="algo_broker")
        options_algo_ui = _load_module("options_algo.ui")
        try:
            options_algo_ui.render(broker_choice)
        except Exception as exc:
            st.warning("Options algo load failed. Retry or inspect the backend logs.")
            if debug_mode:
                st.exception(exc)
        return

    history_ui = _load_module("history.ui")
    try:
        history_ui.render()
    except Exception as exc:
        st.warning("History load failed. Retry or inspect the backend logs.")
        if debug_mode:
            st.exception(exc)


_bootstrap_session_state()

if not st.session_state["logged_in"]:
    _render_login_screen()
    st.stop()

_render_app()

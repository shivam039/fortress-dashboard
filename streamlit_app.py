import importlib
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parent
ENGINE_DIR = ROOT_DIR / "engine"

for path in (str(ENGINE_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

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


st.set_page_config(page_title="Fortress 95 Pro", layout="wide")


ENABLE_NEW_FEATURES = False


DEFAULT_API_URL = os.environ.get("FORTRESS_API_URL", "").strip() or "http://127.0.0.1:8000"
MF_JOB_OPTIONS = {
    "Refresh NAV Cache": "refresh_nav",
    "Update Metrics": "update_metrics",
    "Full Refresh": "full_refresh",
    "Recalculate Rankings": "recalculate_rankings",
}
ORDER_STATUS_OPTIONS = ["Pending", "Executed", "Rejected", "Cancelled"]
BROKER_OPTIONS = ["Zerodha", "Dhan"]

# Broker OAuth / Login URLs
BROKER_LOGIN_URLS = {
    "Zerodha": "https://kite.zerodha.com/connect/login?api_key={api_key}&v=3",
    "Dhan": "https://api.dhan.co/v2/login",
}
# Zerodha login generates a request_token in the redirect URL.
# We read it from st.query_params and exchange or store it.

BASE_MODULES = [
    "🏠 Dashboard",
    "📊 Stock Screener",
    "📈 MF Lab",
    "📋 Orders",
    "🌍 Commodities",
    "⚡ Options",
    "🕐 Scan History",
]


def _available_modules() -> List[str]:
    modules = list(BASE_MODULES)
    if st.session_state.get("ENABLE_NEW_FEATURES", False):
        modules.insert(1, "👤 Profile")
    return modules


def _configured_users() -> Dict[str, Dict[str, str]]:
    username = os.environ.get("FORTRESS_APP_USERNAME", "admin")
    return {
        username: {
            "password": os.environ.get("FORTRESS_APP_PASSWORD", "fortress123"),
            "full_name": os.environ.get("FORTRESS_APP_FULL_NAME", "Fortress Admin"),
            "email": os.environ.get("FORTRESS_APP_EMAIL", "admin@fortress.local"),
            "phone": os.environ.get("FORTRESS_APP_PHONE", "+91 99999 99999"),
            "account_status": os.environ.get("FORTRESS_APP_STATUS", "Active"),
        }
    }


def _bootstrap_session_state() -> None:
    st.session_state.setdefault("ENABLE_NEW_FEATURES", ENABLE_NEW_FEATURES)
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("auth_error", "")
    st.session_state.setdefault("current_user", "")
    st.session_state.setdefault("current_user_profile", {})
    # Use setdefault first, then actively repair blank/None values that may have
    # been persisted from earlier sessions (setdefault won't overwrite them).
    st.session_state.setdefault("fastapi_url", DEFAULT_API_URL)
    if not str(st.session_state.get("fastapi_url", "")).strip():
        st.session_state["fastapi_url"] = DEFAULT_API_URL
    st.session_state.setdefault("mf_job_controls_rendered", False)
    st.session_state.setdefault("screener_results", [])
    st.session_state.setdefault("screener_selected_broker", BROKER_OPTIONS[0])
    st.session_state.setdefault("active_tab", "login")
    st.session_state.setdefault("signup_notice", "")
    st.session_state.setdefault("show_delete_confirm", False)
    st.session_state.setdefault("active_module", BASE_MODULES[0])


def _load_module(module_name: str):
    return importlib.import_module(module_name)


def _format_timestamp(value: Any) -> str:
    if value in (None, "", pd.NaT):
        return "N/A"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    if parsed.tzinfo is None:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return parsed.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST")


def _authenticate(username: str, password: str) -> bool:
    from utils.db import verify_user_credentials

    username = username.strip()

    # Admin shortcut — uses FORTRESS_APP_PASSWORD env var
    if username == "admin":
        admin_pwd = os.environ.get("FORTRESS_APP_PASSWORD", "fortress123")
        if not admin_pwd:
            st.error(
                "⚠️ Admin login is disabled: the **FORTRESS_APP_PASSWORD** "
                "environment variable is not set. Contact the administrator.",
                icon="🔐",
            )
            return False
        return password == admin_pwd

    return verify_user_credentials(username.strip(), password)



def _sync_user_profile(username: str) -> Dict[str, Any]:
    from utils.db import get_app_user, upsert_app_user

    user_config = _configured_users().get(username, {})
    upsert_app_user(
        username=username,
        full_name=user_config.get("full_name", ""),
        email=user_config.get("email", ""),
        phone=user_config.get("phone", ""),
        account_status=user_config.get("account_status", "Active"),
    )
    return get_app_user(username)


def _render_login_screen() -> None:
    st.title("🏹 Fortress Terminal")
    st.caption("Professional quantitative dashboard and execution engine.")
    if st.session_state.get("ENABLE_NEW_FEATURES", False):
        st.info("Enhanced workspace mode is enabled for this session.", icon="✨")

    _, center, _ = st.columns([1, 1.5, 1])
    with center:
        active_tab = st.session_state.get("active_tab", "login")
        if active_tab == "login":
            tab_login, tab_signup, tab_guest = st.tabs(["🔐 Login", "📝 Sign Up", "👤 Guest"])
        elif active_tab == "signup":
            tab_signup, tab_login, tab_guest = st.tabs(["📝 Sign Up", "🔐 Login", "👤 Guest"])
        else:
            tab_login, tab_signup, tab_guest = st.tabs(["🔐 Login", "📝 Sign Up", "👤 Guest"])

        with tab_login:
            if st.session_state.get("signup_notice"):
                st.success(st.session_state["signup_notice"])
                st.session_state["signup_notice"] = ""
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submitted:
                if _authenticate(username, password):
                    from utils.db import record_user_login
                    username = username.strip()
                    profile = _sync_user_profile(username)
                    record_user_login(username)
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = username
                    st.session_state["current_user_profile"] = profile
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

        with tab_signup:
            with st.form("signup_form"):
                new_user = st.text_input("Username*")
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                new_pass = st.text_input("Password*", type="password")
                signup_btn = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if signup_btn:
                from utils.db import get_app_user, upsert_app_user

                clean_user = new_user.strip()
                clean_pass = new_pass.strip()
                existing_user = get_app_user(clean_user) if clean_user else {}

                if not clean_user or not clean_pass:
                    st.error("Username and Password are required.")
                elif existing_user and existing_user.get("password_hash"):
                    st.error("Username already exists. Please choose a different username.")
                else:
                    upsert_app_user(username=clean_user, full_name=full_name, email=email, password=clean_pass)
                    st.session_state["active_tab"] = "login"
                    st.session_state["signup_notice"] = "Account created successfully. Please sign in."
                    st.rerun()

        with tab_guest:
            st.write("Explore the Fortress terminal with a temporary guest session. Note: Broker connections are saved per account.")
            if st.button("Continue as Guest", type="secondary", use_container_width=True):
                from utils.db import record_user_login, upsert_app_user
                guest_username = "guest_user"
                upsert_app_user(username=guest_username, full_name="Guest Explorer", account_status="Trial")
                profile = _sync_user_profile(guest_username)
                record_user_login(guest_username)
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = guest_username
                st.session_state["current_user_profile"] = profile
                st.rerun()


def _logout() -> None:
    fastapi_url = st.session_state.get("fastapi_url", DEFAULT_API_URL)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _bootstrap_session_state()
    st.session_state["fastapi_url"] = fastapi_url
    st.rerun()


@st.dialog("Confirm Logout")
def _logout_dialog() -> None:
    st.write("Are you sure you want to log out of the Fortress Terminal?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Logout", type="primary", use_container_width=True):
            _logout()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("⚠️ Delete Account")
def _delete_account_dialog(username: str) -> None:
    from utils.db import delete_app_user

    st.error("This will **permanently delete** your account, all broker connections, and order history. This action cannot be undone.")
    confirm_text = st.text_input("Type your username to confirm", placeholder=username)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Delete My Account", type="primary", use_container_width=True):
            if confirm_text.strip() == username:
                delete_app_user(username)
                _logout()
            else:
                st.error("Username does not match. Please try again.")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def _get_active_broker_names(username: str) -> List[str]:
    """Returns cached active broker names. Refreshes from DB only on first call per session or after invalidation."""
    if "active_brokers_cache" not in st.session_state:
        from utils.db import list_user_broker_connections
        df = list_user_broker_connections(username)
        st.session_state["brokers_df_cache"] = df
        st.session_state["active_brokers_cache"] = (
            df[df["is_active"].astype(bool)]["broker_name"].dropna().astype(str).tolist()
            if not df.empty else []
        )
    return st.session_state["active_brokers_cache"]


def _invalidate_broker_cache() -> None:
    """Call after any broker connect/disconnect to force a fresh fetch on next render."""
    st.session_state.pop("active_brokers_cache", None)
    st.session_state.pop("brokers_df_cache", None)



def _render_profile_section(profile: Dict[str, Any]) -> None:
    st.markdown("### 👤 Profile")
    with st.container(border=True):
        st.markdown(f"#### {profile.get('full_name') or 'Fortress User'}")
        st.caption(f"📧 {profile.get('email') or 'N/A'}")
        st.caption(f"📱 {profile.get('phone') or 'N/A'}")
        st.divider()
        st.caption("Account Details")
        st.write(f"**Status:** {profile.get('account_status') or 'Active'}")
        st.write(f"**Joined:** {_format_timestamp(profile.get('created_at'))}")
        st.write(f"**Last Login:** {_format_timestamp(profile.get('last_login_at'))}")


def _render_profile_page(profile: Dict[str, Any], username: str) -> None:
    st.subheader("👤 Profile")
    st.caption("Professional account summary and sign-in details.")
    card_data = [
        ("Name", profile.get("full_name") or username),
        ("Email", profile.get("email") or "N/A"),
        ("Account Created", _format_timestamp(profile.get("created_at"))),
        ("Last Login", _format_timestamp(profile.get("last_login_at"))),
        ("Status", profile.get("account_status") or "Active"),
    ]
    row1 = st.columns(2)
    row2 = st.columns(3)
    card_cols = [row1[0], row1[1], row2[0], row2[1], row2[2]]
    for idx, (label, value) in enumerate(card_data):
        with card_cols[idx]:
            with st.container(border=True):
                st.caption(label)
                st.markdown(f"**{value}**")


def _render_enhanced_orders_table(username: str) -> None:
    from utils.db import fetch_fortress_orders

    orders_df = fetch_fortress_orders(username=username)
    if orders_df.empty:
        st.info("No orders placed from Fortress yet.")
        return
    display_cols = [
        col for col in ["symbol", "order_type", "quantity", "price", "status", "broker_name", "created_at"]
        if col in orders_df.columns
    ]
    if "created_at" in display_cols:
        orders_df["created_at"] = orders_df["created_at"].apply(_format_timestamp)
    st.dataframe(
        orders_df[display_cols].rename(
            columns={
                "symbol": "Symbol",
                "order_type": "Type",
                "quantity": "Qty",
                "price": "Price",
                "status": "Status",
                "broker_name": "Broker",
                "created_at": "Timestamp",
            }
        ),
        width="stretch",
        hide_index=True,
    )

@st.dialog("Connect Broker")
def _connect_broker_dialog(username: str):
    from utils.db import upsert_user_broker_connection
    st.write("Link your Zerodha or Dhan account by providing an access token. All tokens are encrypted before storage.")

    with st.form("broker_connection_form", clear_on_submit=True):
        broker_name = st.selectbox("Broker", BROKER_OPTIONS)
        broker_client_id = st.text_input("Client ID / User ID", placeholder="e.g. AB1234 or DHAN_ID")
        access_token = st.text_area("Access Token", placeholder="Paste your permanent or session access token here...", height=120)
        col1, col2 = st.columns(2)
        with col1:
             expires_on = st.text_input("Expiry (Optional)", placeholder="YYYY-MM-DD")
        with col2:
             refresh_token = st.text_input("Refresh Token (Optional)", type="password")
        submitted = st.form_submit_button("💾 Save & Connect", type="primary", use_container_width=True)

    if submitted:
        if not access_token.strip():
            st.error("Access token is required.")
        else:
            upsert_user_broker_connection(
                username=username,
                broker_name=broker_name,
                broker_client_id=broker_client_id.strip(),
                access_token=access_token.strip(),
                refresh_token=refresh_token.strip(),
                expires_at=expires_on.strip() or None,
            )
            _invalidate_broker_cache()
            st.success(f"✅ {broker_name} connection saved successfully.")
            st.rerun()


@st.dialog("🔗 Broker Login")
def _broker_login_dialog(username: str) -> None:
    """Guides user through broker OAuth login and captures the returned token."""
    from utils.db import upsert_user_broker_connection

    st.write("Choose your broker and follow the steps to authenticate via their official login page.")
    broker_name = st.selectbox("Broker", BROKER_OPTIONS, key="broker_login_dialog_broker")

    st.divider()

    if broker_name == "Zerodha":
        api_key = st.text_input(
            "Your Kite API Key",
            placeholder="Enter your Zerodha Kite API Key",
            help="Get this from your Kite Connect developer account at https://developers.kite.trade",
        )
        if api_key:
            login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key.strip()}&v=3"
            st.info("Click the button below to open Zerodha Kite login. After authentication you will be redirected back with a `request_token` in the URL. Copy and paste it below.")
            st.link_button("🔐 Login via Zerodha Kite", login_url, use_container_width=True)

            st.divider()
            st.markdown("**Step 2: Paste the `request_token` from redirect URL**")
            st.caption("After login, Zerodha redirects you to your redirect URL with `?request_token=XXXXX&status=success`. Copy the token value.")
            request_token = st.text_input("Request Token / Access Token", type="password", placeholder="Paste token here...")
            client_id = st.text_input("Client ID (optional)", placeholder="Your Zerodha User ID e.g. AB1234")

            if st.button("✅ Save Zerodha Token", type="primary", use_container_width=True):
                if request_token.strip():
                    upsert_user_broker_connection(
                        username=username,
                        broker_name="Zerodha",
                        broker_client_id=client_id.strip(),
                        access_token=request_token.strip(),
                    )
                    _invalidate_broker_cache()
                    st.success("✅ Zerodha token saved! You can now use it for order placement.")
                    st.rerun()
                else:
                    st.error("Please paste the token before saving.")
        else:
            st.warning("Enter your API key to get the login URL.")

    elif broker_name == "Dhan":
        st.info("Dhan uses a permanent access token. Generate it from your Dhan developer console and paste it below.")
        st.link_button("🔐 Open Dhan Console", "https://login.dhan.co", use_container_width=True)
        st.divider()
        client_id = st.text_input("Dhan Client ID", placeholder="Your Dhan User ID")
        access_token = st.text_input("Access Token", type="password", placeholder="Paste your Dhan access token")

        if st.button("✅ Save Dhan Token", type="primary", use_container_width=True):
            if access_token.strip():
                upsert_user_broker_connection(
                    username=username,
                    broker_name="Dhan",
                    broker_client_id=client_id.strip(),
                    access_token=access_token.strip(),
                )
                _invalidate_broker_cache()
                st.success("✅ Dhan token saved! You can now use it for order placement.")
                st.rerun()
            else:
                st.error("Please paste the access token before saving.")


def _handle_broker_oauth_callback(username: str) -> None:
    """Reads query params after a broker OAuth redirect and auto-saves the token."""
    from utils.db import upsert_user_broker_connection

    params = st.query_params
    request_token = params.get("request_token", "")
    status = params.get("status", "")

    if request_token and status == "success":
        st.success("✅ Zerodha login successful! Saving your access token...")
        upsert_user_broker_connection(
            username=username,
            broker_name="Zerodha",
            access_token=request_token,
        )
        _invalidate_broker_cache()
        st.query_params.clear()
        st.rerun()


def _check_token_expiry(expires_at_raw) -> tuple[str, str]:
    """Returns (badge_text, color) based on token expiry date."""
    if not expires_at_raw or str(expires_at_raw).strip() in ("", "None", "nan"):
        return ("", "")
    try:
        import datetime as dt
        exp = pd.to_datetime(expires_at_raw, errors="coerce")
        if pd.isna(exp):
            return ("", "")
        now = pd.Timestamp.now(tz=exp.tzinfo)
        days_left = (exp - now).days
        if days_left < 0:
            return ("🔴 Token Expired", "#ff4b4b")
        elif days_left <= 2:
            return (f"🟠 Expires in {days_left}d", "#ffa500")
        elif days_left <= 7:
            return (f"🟡 Expires in {days_left}d", "#f0c040")
        else:
            return (f"🟢 Valid ({days_left}d)", "#00c851")
    except Exception:
        return ("", "")


def _render_broker_settings_section(username: str) -> None:
    from utils.db import delete_user_broker_connection

    st.markdown("### 🔑 Broker Connections")
    # Use cached brokers_df from session state if available
    brokers_df = st.session_state.get("brokers_df_cache", pd.DataFrame())
    if brokers_df.empty:
        _get_active_broker_names(username)  # populates cache
        brokers_df = st.session_state.get("brokers_df_cache", pd.DataFrame())

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🔗 Login via Broker", use_container_width=True, type="primary"):
            _broker_login_dialog(username)
    with btn_col2:
        if st.button("➕ Manual Token", use_container_width=True):
            _connect_broker_dialog(username)

    if not brokers_df.empty:
        st.divider()
        for _, row in brokers_df.iterrows():
            expiry_badge, expiry_color = _check_token_expiry(row.get("expires_at"))
            is_expired = expiry_badge.startswith("🔴")

            with st.container(border=True):
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    st.write(f"**{row['broker_name']}** ({row.get('broker_client_id') or 'N/A'})")
                    status = "❌ Inactive (Expired)" if is_expired else ("✅ Active" if bool(row.get("is_active")) else "❌ Inactive")
                    st.caption(f"{status} | Connected: {_format_timestamp(row.get('connected_at'))}")
                    if expiry_badge:
                        st.markdown(
                            f'<span style="color:{expiry_color};font-size:0.8em;font-weight:600;">{expiry_badge}</span>',
                            unsafe_allow_html=True,
                        )
                    if is_expired:
                        st.warning("⚠️ Token expired — reconnect to restore broker access.", icon=None)
                with col_btn:
                    if st.button("Delete", key=f"del_{row['broker_name']}", type="secondary", use_container_width=True):
                        delete_user_broker_connection(username, row['broker_name'])
                        _invalidate_broker_cache()
                        st.rerun()
    else:
        st.caption("No broker connections yet. Use the buttons above to connect.")


def _render_broker_settings_section_enhanced(username: str) -> None:
    st.markdown("### Broker Settings")
    _render_broker_settings_section(username)
    active_count = len(_get_active_broker_names(username))
    st.caption(f"Connection Status: {'🟢 Connected' if active_count else '🔴 Not Connected'}")




def _run_mf_job_directly(payload: dict) -> None:
    """Run an MF job in-process using the engine modules.
    
    Uses threading to keep Streamlit responsive since MF jobs can be heavy.
    """
    from mf_lab.jobs import _run_job_sync
    
    def _thread_target():
        try:
            _run_job_sync(
                job_type=payload["job_type"],
                force_refresh=payload.get("force_refresh", False),
                scheme_codes=payload.get("scheme_codes")
            )
        except Exception as e:
            logging.error(f"In-process MF job failed: {e}")

    thread = threading.Thread(target=_thread_target, daemon=True)
    thread.start()


def _render_mf_job_controls(api_url: str, key_prefix: str, sidebar: bool = False) -> None:
    target = st.sidebar if sidebar else st
    target.markdown("**MF Data Jobs**" if sidebar else "### Server-Side MF Data Jobs")
    target.caption("Trigger heavy MF processing on FastAPI while Streamlit stays responsive.")

    job_label = target.selectbox("Job Type", list(MF_JOB_OPTIONS.keys()), key=f"{key_prefix}_job_type")
    force_refresh = target.checkbox("Force Refresh", value=False, key=f"{key_prefix}_force_refresh")
    scheme_code_text = target.text_input(
        "Scheme Codes (optional)",
        key=f"{key_prefix}_scheme_codes",
        placeholder="e.g. 120503, 120716",
    )

    if target.button("🚀 Trigger Job", type="primary", use_container_width=True, key=f"{key_prefix}_trigger_button"):
        scheme_codes = [code.strip() for code in scheme_code_text.split(",") if code.strip()]
        payload = {
            "job_type": MF_JOB_OPTIONS[job_label],
            "force_refresh": force_refresh,
            "scheme_codes": scheme_codes or None,
        }
        
        # ── Try FastAPI first ────────────────────────────────────────────────
        _used_direct = False
        try:
            if api_url and api_url.strip().startswith("http"):
                response = requests.post(f"{api_url.rstrip('/')}/mf/trigger-job", json=payload, timeout=10)
                if response.status_code == 202:
                    target.success(f"Job `{payload['job_type']}` accepted (FastAPI).")
                else:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    target.error(f"Server rejected: {detail}")
            else:
                raise requests.exceptions.ConnectionError("No valid API URL — running in-process.")
        except requests.exceptions.ConnectionError:
            _used_direct = True
        except requests.exceptions.RequestException as exc:
            target.error(f"Request failed: {exc}")

        if _used_direct:
            try:
                _run_mf_job_directly(payload)
                target.info(f"Job `{payload['job_type']}` started in-process (background thread).")
                target.caption("Check server logs/audit for completion status.")
            except Exception as e:
                target.error(f"Direct job failure: {e}")


def _render_sidebar_module_filters(module: str, username: str, api_url: str) -> dict:
    """Renders contextual filter controls in the sidebar for the active module."""
    filters: dict = {}
    is_guest = (username == "guest_user")

    if module == "📊 Stock Screener":
        st.markdown("**Scan Controls**")
        universes = _fetch_universes(api_url)
        active_brokers = _get_active_broker_names(username)
        broker_choices = active_brokers or BROKER_OPTIONS
        default_broker = st.session_state.get("screener_selected_broker", broker_choices[0])
        if default_broker not in broker_choices:
            default_broker = broker_choices[0]
        
        filters["universe"] = st.selectbox("Universe", universes, key="sb_universe")
        filters["portfolio_val"] = st.number_input(
            "Portfolio (₹)", min_value=100_000.0, value=1_000_000.0, step=50_000.0, key="sb_portfolio_val"
        )
        filters["risk_pct"] = st.number_input(
            "Risk %", min_value=0.1, value=1.0, step=0.1, format="%.1f", key="sb_risk_pct"
        )
        if not is_guest:
            filters["broker"] = st.selectbox(
                "Broker", broker_choices, index=broker_choices.index(default_broker), key="sb_broker"
            )
        else:
            filters["broker"] = broker_choices[0] # Hidden default for guest

    elif module == "📈 MF Lab":
        _render_mf_job_controls(api_url, key_prefix="mf_sidebar", sidebar=True)

    elif module == "📋 Orders":
        st.markdown("**Order Filters**")
        active_brokers = _get_active_broker_names(username)
        broker_filter_options = ["All"] + sorted(set(active_brokers + BROKER_OPTIONS))
        filters["status"] = st.selectbox(
            "Status", ["All"] + ORDER_STATUS_OPTIONS, key="sb_order_status"
        )
        if not is_guest:
            filters["broker"] = st.selectbox("Broker", broker_filter_options, key="sb_order_broker")
        else:
            filters["broker"] = "All"
            
        filters["date_from"] = st.text_input(
            "From Date", key="sb_date_from", placeholder="2026-04-01"
        )
        filters["date_to"] = st.text_input(
            "To Date", key="sb_date_to", placeholder="2026-04-30"
        )

    elif module in ("🌍 Commodities", "⚡ Options"):
        active_brokers = _get_active_broker_names(username)
        broker_choices = active_brokers or BROKER_OPTIONS
        safe_key = module.replace(" ", "_").replace("🌍", "com").replace("⚡", "opt")
        if not is_guest:
            filters["broker"] = st.selectbox("Broker", broker_choices, key=f"sb_{safe_key}_broker")
        else:
            filters["broker"] = broker_choices[0]

    return filters



@st.cache_data(ttl=300)
def _fetch_universes(api_url: str) -> List[str]:
    try:
        response = requests.get(f"{api_url.rstrip('/')}/api/universes", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        from fortress_config import TICKER_GROUPS
        return list(TICKER_GROUPS.keys())



def _build_order_link(symbol: str, quantity: float, price: float, broker_name: str) -> str:
    from utils.broker_mappings import generate_dhan_url, generate_zerodha_url

    if broker_name == "Dhan":
        return generate_dhan_url(symbol, quantity, price) or ""
    return generate_zerodha_url(symbol, quantity) or ""


def _render_dashboard_tab(profile: Dict[str, Any], username: str) -> None:
    from utils.db import fetch_fortress_orders, list_user_broker_connections

    brokers_df = list_user_broker_connections(username)
    orders_df = fetch_fortress_orders(username)
    is_guest = (username == "guest_user")

    st.subheader("Dashboard")
    st.caption("Quick overview of your Fortress workspace, broker connectivity, and recent order flow.")

    col1, col2, col3, col4 = st.columns(4)
    if not is_guest:
        col1.metric("Active Brokers", int(brokers_df["is_active"].astype(bool).sum()) if not brokers_df.empty else 0)
    else:
        col1.metric("Account Type", "Guest Explorer")
    col2.metric("Total Orders", len(orders_df))
    col3.metric("Pending Orders", int((orders_df["status"] == "Pending").sum()) if not orders_df.empty else 0)
    col4.metric("Account Status", profile.get("account_status", "Active"))

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Profile Snapshot")
        summary_df = pd.DataFrame(
            [
                {"Field": "Full Name", "Value": profile.get("full_name") or "N/A"},
                {"Field": "Email", "Value": profile.get("email") or "N/A"},
                {"Field": "Phone", "Value": profile.get("phone") or "N/A"},
                {"Field": "Last Login", "Value": _format_timestamp(profile.get("last_login_at"))},
            ]
        )
        st.dataframe(summary_df, width="stretch", hide_index=True)

    with right:
        st.markdown("#### Recent Orders")
        if orders_df.empty:
            st.info("No orders recorded yet.")
        else:
            preview_cols = [c for c in ["order_id", "symbol", "order_type", "quantity", "status", "broker_name", "created_at"] if c in orders_df.columns]
            st.dataframe(orders_df[preview_cols].head(8), width="stretch", hide_index=True)


def _run_scan_directly(payload: dict) -> list:
    """Run the stock scan in-process using the engine modules directly.

    This mirrors the logic in engine/main.py POST /api/scan so the screener
    works on Streamlit Cloud where no separate FastAPI process is running.
    """
    import logging
    from stock_scanner.logic import (
        check_institutional_fortress,
        apply_advanced_scoring,
        get_stock_data,
        DEFAULT_SCORING_CONFIG,
    )
    from stock_scanner.ui import generate_action_link
    from stock_scanner.pulse import get_current_regime
    from fortress_config import TICKER_GROUPS

    logger = logging.getLogger("fortress-direct-scan")

    universe = payload["universe"]
    tickers = TICKER_GROUPS.get(universe)
    if not tickers:
        raise ValueError(f"Universe '{universe}' not found.")

    try:
        regime_data = get_current_regime()
    except Exception as e:
        logger.warning(f"Regime fetch failed, defaulting to Range: {e}")
        regime_data = {"Market_Regime": "Range", "Regime_Multiplier": 1.0, "VIX": 20.0}

    batch_data = get_stock_data(tickers, period="1y", interval="1d", group_by="ticker")
    results = []
    for ticker in tickers:
        try:
            hist = batch_data[ticker].dropna() if len(tickers) > 1 else batch_data.dropna()
            if not hist.empty and len(hist) >= 210:
                res = check_institutional_fortress(
                    ticker,
                    hist,
                    None,
                    payload["portfolio_val"],
                    payload["risk_pct"],
                    selected_universe=universe,
                    regime_data=regime_data,
                )
                if res:
                    results.append(res)
        except Exception as e:
            logger.warning(f"Error scanning {ticker}: {e}")

    if not results:
        return []

    df = pd.DataFrame(results)
    scoring_config = DEFAULT_SCORING_CONFIG.copy()
    scoring_config.update({
        "enable_regime": payload.get("enable_regime", True),
        "liquidity_cr_min": payload.get("liquidity_cr_min", 8.0),
        "market_cap_cr_min": payload.get("market_cap_cr_min", 1500.0),
        "price_min": payload.get("price_min", 80.0),
        "regime": regime_data,
    })
    if payload.get("weights"):
        scoring_config["weights"] = payload["weights"]

    df = apply_advanced_scoring(df, scoring_config)
    broker = payload.get("broker", "Zerodha")
    df["Actions"] = df.apply(lambda row: generate_action_link(row, broker), axis=1)
    return df.to_dict(orient="records")


def _render_stock_screener_tab(username: str, api_url: str, sidebar_filters: dict = None) -> None:
    from utils.db import create_fortress_order

    f = sidebar_filters or {}
    universe = f.get("universe", "NIFTY50")
    portfolio_val = f.get("portfolio_val", 1_000_000.0)
    risk_pct = f.get("risk_pct", 1.0)
    active_brokers = _get_active_broker_names(username)
    broker_choices = active_brokers or BROKER_OPTIONS
    broker_name = f.get("broker", broker_choices[0])

    st.subheader("📊 Stock Screener")
    st.caption("Scan controls are in the sidebar. Use Advanced Settings below for fine-tuning.")

    # Defaults (overridden by widgets inside the expander)
    enable_regime = True
    liquidity_cr_min = 8.0
    market_cap_cr_min = 1500.0
    price_min = 80.0
    technical = 50
    fundamental = 25
    sentiment = 15
    context_w = 10

    with st.expander("⚙️ Advanced Scan Settings", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            enable_regime = st.checkbox("Enable Regime Scaling", value=True)
        with col_b:
            liquidity_cr_min = st.number_input("Liquidity Gate (₹ Cr)", min_value=0.0, value=8.0, step=0.5)
        with col_c:
            market_cap_cr_min = st.number_input("Market Cap Gate (₹ Cr)", min_value=0.0, value=1500.0, step=50.0)
        with col_d:
            price_min = st.number_input("Min Price Gate (₹)", min_value=0.0, value=80.0, step=5.0)

        w1, w2, w3, w4 = st.columns(4)
        with w1:
            technical = st.slider("Technical", 0, 100, 50, 1)
        with w2:
            fundamental = st.slider("Fundamental", 0, 100, 25, 1)
        with w3:
            sentiment = st.slider("Sentiment", 0, 100, 15, 1)
        with w4:
            context_w = st.slider("Context", 0, 100, 10, 1)


    if st.button("🔍 Run Screener", type="primary", use_container_width=True):
        total = max(technical + fundamental + sentiment + context_w, 1)
        payload = {
            "universe": universe,
            "portfolio_val": portfolio_val,
            "risk_pct": risk_pct / 100.0,
            "weights": {
                "technical": technical / total,
                "fundamental": fundamental / total,
                "sentiment": sentiment / total,
                "context": context_w / total,
            },
            "enable_regime": enable_regime,
            "liquidity_cr_min": liquidity_cr_min,
            "market_cap_cr_min": market_cap_cr_min,
            "price_min": price_min,
            "broker": broker_name,
        }
        # ── Try FastAPI first (local dev / external deployment) ─────────────
        # Fall back to direct in-process engine call when no server is running
        # (e.g. Streamlit Cloud where only streamlit_app.py is executed).
        _used_direct = False
        try:
            if api_url and api_url.strip().startswith("http"):
                with st.spinner("Running scan via FastAPI..."):
                    response = requests.post(
                        f"{api_url.rstrip('/')}/api/scan", json=payload, timeout=180
                    )
                    response.raise_for_status()
                    st.session_state["screener_results"] = response.json()
                    st.session_state["screener_selected_broker"] = broker_name
                st.success("Stock scan completed (FastAPI).")
            else:
                raise requests.exceptions.ConnectionError("No valid API URL — running in-process.")
        except requests.exceptions.ConnectionError:
            # FastAPI not reachable — run the scan directly inside Streamlit.
            _used_direct = True
        except requests.exceptions.RequestException as exc:
            st.error(f"Scan failed: {exc}")

        if _used_direct:
            try:
                with st.spinner("Running scan in-process (engine)..."):
                    records = _run_scan_directly(payload)
                st.session_state["screener_results"] = records
                st.session_state["screener_selected_broker"] = broker_name
                st.success(f"Stock scan completed — {len(records)} results.")
            except Exception as exc:
                st.error(f"Direct scan failed: {exc}")

    results = pd.DataFrame(st.session_state.get("screener_results", []))
    if results.empty:
        st.info("Run a scan to see actionable stock setups here.")
        return

    # ── Strategic Splits ──────────────────────────────────────────────────
    momentum_picks = results[results["Strategy"] == "Momentum Pick"].copy()
    lt_picks = results[results["Strategy"] == "Long-Term Pick"].copy()

    if not momentum_picks.empty:
        st.markdown(f"#### 🚀 Momentum Picks ({len(momentum_picks)})")
        if "Actions" in momentum_picks.columns: momentum_picks.drop(columns=["Actions"], inplace=True)
        st.dataframe(momentum_picks, width="stretch", hide_index=True)

    if not lt_picks.empty:
        st.markdown(f"#### 💎 Long-Term Picks ({len(lt_picks)})")
        if "Actions" in lt_picks.columns: lt_picks.drop(columns=["Actions"], inplace=True)
        st.dataframe(lt_picks, width="stretch", hide_index=True)

    # ── Full Results ─────────────────────────────────────────────────────
    st.markdown("#### 📋 All Actionable Setups")
    display_df = results.copy()
    if "Actions" in display_df.columns:
        display_df = display_df.drop(columns=["Actions"])
    st.dataframe(display_df, width="stretch", hide_index=True)

    symbol_options = display_df["Symbol"].dropna().astype(str).tolist() if "Symbol" in display_df.columns else []
    if not symbol_options:
        return

    st.markdown("#### Record Order From Fortress")
    with st.form("fortress_order_form"):
        selected_symbol = st.selectbox("Symbol", symbol_options)
        selected_row = display_df[display_df["Symbol"] == selected_symbol].iloc[0]
        order_cols = st.columns(4)
        with order_cols[0]:
            order_type = st.selectbox("Order Type", ["Buy", "Sell"])
        with order_cols[1]:
            quantity = st.number_input("Quantity", min_value=1.0, value=float(selected_row.get("Position_Qty", 1) or 1), step=1.0)
        with order_cols[2]:
            price = st.number_input("Price", min_value=0.0, value=float(selected_row.get("Price", 0) or 0), step=1.0)
        with order_cols[3]:
            status = st.selectbox("Status", ORDER_STATUS_OPTIONS)
        notes = st.text_input("Notes", value=str(selected_row.get("Strategy", "")))
        submitted = st.form_submit_button("Save Order", type="primary", use_container_width=True)

    broker_link = _build_order_link(selected_symbol, quantity, price, broker_name)
    if broker_link and username != "guest_user":
        st.link_button("Open Broker Order Page", broker_link, use_container_width=False)

    # ── Conviction Heatmap ──────────────────────────────────────────────
    if not results.empty and "Score" in results.columns:
        st.subheader("📊 Conviction Heatmap")
        # Ensure Score is numeric
        results["Score"] = pd.to_numeric(results["Score"], errors="coerce").fillna(0)
        
        plt.figure(figsize=(10, max(4, len(results) / 3)))
        # Create conviction bands
        def get_band(x):
            if x >= 85: return "🔥 High (85+)"
            if x >= 60: return "🚀 Pass (60-85)"
            return "🟡 Watch (<60)"
            
        heatmap_df = results[["Symbol", "Score"]].copy()
        heatmap_df["Conviction_Band"] = heatmap_df["Score"].apply(get_band)
        
        # Pivot for heatmap
        pivot = heatmap_df.pivot_table(index="Symbol", columns="Conviction_Band", values="Score", fill_value=0)
        
        # Ensure all columns exist for consistent layout
        for col in ["🔥 High (85+)", "🚀 Pass (60-85)", "🟡 Watch (<60)"]:
            if col not in pivot.columns:
                pivot[col] = 0.0
        
        # Reorder columns
        pivot = pivot[["🔥 High (85+)", "🚀 Pass (60-85)", "🟡 Watch (<60)"]]
        
        sns.heatmap(pivot, annot=True, cmap="Greens", cbar=False, linewidths=0.5, linecolor='grey')
        st.pyplot(plt)

    if submitted:
        create_fortress_order(
            username=username,
            symbol=selected_symbol,
            stock_name=str(selected_row.get("Company", selected_symbol)),
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=status,
            broker_name=broker_name,
            notes=notes,
        )
        st.success(f"Order for {selected_symbol} saved to Fortress order history.")


def _render_mf_lab_tab(api_url: str) -> None:
    mf_lab_ui = _load_module("mf_lab.ui")
    st.subheader("📈 MF Lab")
    st.caption("Use the sidebar to trigger server-side MF jobs. Analysis results appear below.")
    if st.session_state.get("ENABLE_NEW_FEATURES", False):
        _render_mf_job_controls(api_url, key_prefix="mf_main", sidebar=False)
    st.session_state["mf_job_controls_rendered"] = True
    st.markdown("---")
    mf_lab_ui.render()



def _render_orders_tab(username: str, sidebar_filters: dict = None) -> None:
    from utils.db import fetch_fortress_orders

    f = sidebar_filters or {}
    status_filter = f.get("status", "All")
    broker_filter = f.get("broker", "All")
    date_from = f.get("date_from", "").strip()
    date_to = f.get("date_to", "").strip()

    st.subheader("📋 Orders")
    st.caption("Filters are in the sidebar. Showing Fortress orders for your account.")

    orders_df = fetch_fortress_orders(
        username=username,
        status=status_filter,
        broker_name=broker_filter,
        date_from=date_from or None,
        date_to=date_to or None,
    )

    if orders_df.empty:
        st.info("No orders match the current filters.")
        return

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Orders", len(orders_df))
    metric_col2.metric("Executed", int((orders_df["status"] == "Executed").sum()))
    metric_col3.metric("Pending", int((orders_df["status"] == "Pending").sum()))

    if "created_at" in orders_df.columns:
        orders_df["created_at"] = orders_df["created_at"].apply(_format_timestamp)
    if "updated_at" in orders_df.columns:
        orders_df["updated_at"] = orders_df["updated_at"].apply(_format_timestamp)

    st.dataframe(
        orders_df.rename(
            columns={
                "order_id": "Order ID", "symbol": "Symbol", "stock_name": "Stock Name",
                "order_type": "Order Type", "quantity": "Quantity", "price": "Price",
                "status": "Status", "broker_name": "Broker", "broker_order_id": "Broker Order ID",
                "notes": "Notes", "created_at": "Timestamp",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def _render_commodities_tab(username: str, broker: str = None) -> None:
    commodities_ui = _load_module("commodities.ui")
    active_brokers = _get_active_broker_names(username)
    broker_name = broker or (active_brokers[0] if active_brokers else BROKER_OPTIONS[0])
    st.subheader("🌍 Commodities")
    if username != "guest_user":
        st.caption(f"Broker: **{broker_name}** — change in the sidebar.")
    commodities_ui.render(broker_name)


def _render_options_tab(username: str, broker: str = None) -> None:
    options_ui = _load_module("options_algo.ui")
    active_brokers = _get_active_broker_names(username)
    broker_name = broker or (active_brokers[0] if active_brokers else BROKER_OPTIONS[0])
    st.subheader("⚡ Options")
    if username != "guest_user":
        st.caption(f"Broker: **{broker_name}** — change in the sidebar.")
    options_ui.render(broker_name)



def _render_history_tab() -> None:
    history_ui = _load_module("history.ui")
    st.subheader("🕐 Scan History")
    history_ui.render()


def _render_authenticated_app() -> None:
    username = st.session_state["current_user"]

    # Handle broker OAuth callback (Zerodha request_token in URL params)
    if username != "guest_user":
        _handle_broker_oauth_callback(username)

    # Use cached profile — only re-sync on first load after login, not every render
    profile = st.session_state.get("current_user_profile") or {}
    if not profile:
        profile = _sync_user_profile(username)
        st.session_state["current_user_profile"] = profile

    api_url = st.session_state["fastapi_url"]
    os.environ["FORTRESS_API_URL"] = api_url

    modules = _available_modules()
    if st.session_state.get("active_module") not in modules:
        st.session_state["active_module"] = modules[0]

    with st.sidebar:
        st.markdown("## 🏹 Fortress")
        st.divider()

        # ── Navigation ──────────────────────────────────────────────────────
        module = st.radio(
            "Navigate",
            modules,
            key="active_module",
            label_visibility="collapsed",
        )
        st.divider()

        # ── Contextual Controls ─────────────────────────────────────────────
        filters = _render_sidebar_module_filters(module, username, api_url)
        st.divider()

        # ── Account ─────────────────────────────────────────────────────────
        with st.expander(f"👤 {profile.get('full_name') or username}", expanded=False):
            _render_profile_section(profile)

        if username != "guest_user":
            if st.session_state.get("ENABLE_NEW_FEATURES", False):
                with st.expander("🔑 Broker Settings", expanded=False):
                    _render_broker_settings_section_enhanced(username)
            else:
                with st.expander("🔑 Broker Connections", expanded=False):
                    _render_broker_settings_section(username)

        with st.expander("⚙️ Settings", expanded=False):
            st.text_input("API URL", key="fastapi_url", help="Backend FastAPI endpoint.")
        if st.session_state.get("ENABLE_NEW_FEATURES", False):
            with st.expander("🛠️ Setup", expanded=False):
                st.caption("One-time development helpers.")
                if st.button("Seed 5 Dummy Users", use_container_width=True):
                    from utils.db import seed_dummy_users

                    added_count = seed_dummy_users()
                    st.success(f"Dummy user setup complete. Added {added_count} user(s).")

        st.divider()
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            _logout_dialog()
        if username != "guest_user":
            if st.button("🗑️ Delete Account", use_container_width=True, type="secondary"):
                _delete_account_dialog(username)

    # ── Main Content ─────────────────────────────────────────────────────────
    st.session_state["mf_job_controls_rendered"] = False
    if module == "🏠 Dashboard":
        if st.session_state.get("ENABLE_NEW_FEATURES", False):
            tab_overview, tab_orders = st.tabs(["Overview", "Orders"])
            with tab_overview:
                _render_dashboard_tab(profile, username)
            with tab_orders:
                st.subheader("📋 Orders")
                _render_enhanced_orders_table(username)
        else:
            _render_dashboard_tab(profile, username)
    elif module == "👤 Profile":
        _render_profile_page(profile, username)
    elif module == "📊 Stock Screener":
        _render_stock_screener_tab(username, api_url, filters)
    elif module == "📈 MF Lab":
        _render_mf_lab_tab(api_url)
    elif module == "📋 Orders":
        _render_orders_tab(username, filters)
    elif module == "🌍 Commodities":
        _render_commodities_tab(username, filters.get("broker"))
    elif module == "⚡ Options":
        _render_options_tab(username, filters.get("broker"))
    elif module == "🕐 Scan History":
        _render_history_tab()


_bootstrap_session_state()

# Guard init_db() so it only runs ONCE per browser session, not on every rerender.
# Without this guard, 15+ SQL statements fire on every single page interaction.
# Import is deferred here (not at module top-level) so a transient import failure
# during Streamlit's pre-load phase never causes a NameError at startup.
if not st.session_state.get("_db_initialized"):
    from utils.db import init_db
    init_db()
    st.session_state["_db_initialized"] = True

if not st.session_state["logged_in"]:
    _render_login_screen()
    st.stop()

_render_authenticated_app()

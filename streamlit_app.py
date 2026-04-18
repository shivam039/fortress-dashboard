import importlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
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
MF_JOB_OPTIONS = {
    "Refresh NAV Cache": "refresh_nav",
    "Update Metrics": "update_metrics",
    "Full Refresh": "full_refresh",
    "Recalculate Rankings": "recalculate_rankings",
}
ORDER_STATUS_OPTIONS = ["Pending", "Executed", "Rejected", "Cancelled"]
BROKER_OPTIONS = ["Zerodha", "Dhan"]


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
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("auth_error", "")
    st.session_state.setdefault("current_user", "")
    st.session_state.setdefault("current_user_profile", {})
    st.session_state.setdefault("fastapi_url", DEFAULT_API_URL)
    st.session_state.setdefault("mf_job_controls_rendered", False)
    st.session_state.setdefault("screener_results", [])
    st.session_state.setdefault("screener_selected_broker", BROKER_OPTIONS[0])


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
    user = _configured_users().get(username.strip())
    return bool(user and password == user["password"])


def _sync_user_profile(username: str) -> Dict[str, Any]:
    from utils.db import get_app_user, upsert_app_user

    user_config = _configured_users()[username]
    upsert_app_user(
        username=username,
        full_name=user_config.get("full_name", ""),
        email=user_config.get("email", ""),
        phone=user_config.get("phone", ""),
        account_status=user_config.get("account_status", "Active"),
    )
    return get_app_user(username)


def _render_login_screen() -> None:
    st.title("Fortress Dashboard")
    st.caption("Secure Streamlit terminal for Fortress screening, mutual funds, brokers, and order tracking.")

    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        st.subheader("Login")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            if _authenticate(username, password):
                from utils.db import record_user_login

                username = username.strip()
                profile = _sync_user_profile(username)
                record_user_login(username)
                profile = _sync_user_profile(username)

                st.session_state["logged_in"] = True
                st.session_state["current_user"] = username
                st.session_state["current_user_profile"] = profile
                st.session_state["auth_error"] = ""
                st.rerun()
            else:
                st.session_state["auth_error"] = "Invalid username or password."

        if st.session_state.get("auth_error"):
            st.error(st.session_state["auth_error"])

        st.info("Credentials remain simple for now, but broker tokens are stored encrypted with Fernet in the database.")


def _logout() -> None:
    fastapi_url = st.session_state.get("fastapi_url", DEFAULT_API_URL)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _bootstrap_session_state()
    st.session_state["fastapi_url"] = fastapi_url
    st.rerun()


def _get_active_broker_names(username: str) -> List[str]:
    from utils.db import list_user_broker_connections

    df = list_user_broker_connections(username)
    if df.empty:
        return []
    return df[df["is_active"].astype(bool)]["broker_name"].dropna().astype(str).tolist()


def _render_profile_section(profile: Dict[str, Any]) -> None:
    st.markdown("### Profile")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Full Name")
            st.write(profile.get("full_name") or "N/A")
            st.caption("Email")
            st.write(profile.get("email") or "N/A")
            st.caption("Phone")
            st.write(profile.get("phone") or "N/A")
        with col2:
            st.caption("Account Created")
            st.write(_format_timestamp(profile.get("created_at")))
            st.caption("Last Login")
            st.write(_format_timestamp(profile.get("last_login_at")))
            st.caption("Account Status")
            st.write(profile.get("account_status") or "Active")


def _render_broker_settings_section(username: str) -> None:
    from utils.db import deactivate_user_broker_connection, list_user_broker_connections, upsert_user_broker_connection

    st.markdown("### Broker Settings")
    brokers_df = list_user_broker_connections(username)

    with st.expander("Connected Brokers", expanded=True):
        if brokers_df.empty:
            st.caption("No broker connections yet.")
        else:
            for _, row in brokers_df.iterrows():
                metadata = row.get("metadata_json") or {}
                status = "Connected" if bool(row.get("is_active")) else "Disconnected"
                expires_label = _format_timestamp(row.get("expires_at"))
                with st.container(border=True):
                    st.write(f"**{row['broker_name']}**")
                    st.caption(
                        f"Status: {status} | Connected: {_format_timestamp(row.get('connected_at'))} | "
                        f"Expires: {expires_label}"
                    )
                    if metadata:
                        st.caption(f"Metadata: {metadata}")

    with st.expander("Add or Update Broker", expanded=False):
        with st.form("broker_connection_form", clear_on_submit=True):
            broker_name = st.selectbox("Broker", BROKER_OPTIONS)
            access_token = st.text_input("Access Token", type="password")
            refresh_token = st.text_input("Refresh Token", type="password")
            expires_on = st.text_input("Token Expiry (optional, ISO format)", placeholder="2026-12-31T23:59:59")
            broker_note = st.text_input("Notes", placeholder="Optional note or client id")
            submitted = st.form_submit_button("Save Broker Connection", type="primary", use_container_width=True)

        if submitted:
            if not access_token.strip():
                st.error("Access token is required.")
            else:
                upsert_user_broker_connection(
                    username=username,
                    broker_name=broker_name,
                    access_token=access_token.strip(),
                    refresh_token=refresh_token.strip(),
                    expires_at=expires_on.strip() or None,
                    metadata={"note": broker_note.strip()},
                )
                st.success(f"{broker_name} connection saved.")
                st.rerun()

    with st.expander("Disconnect Broker", expanded=False):
        active_brokers = brokers_df[brokers_df["is_active"].astype(bool)]["broker_name"].tolist() if not brokers_df.empty else []
        if not active_brokers:
            st.caption("No active brokers to disconnect.")
        else:
            broker_to_remove = st.selectbox("Active Broker", active_brokers, key="disconnect_broker_name")
            if st.button("Disconnect Broker", use_container_width=True):
                deactivate_user_broker_connection(username, broker_to_remove)
                st.success(f"{broker_to_remove} disconnected.")
                st.rerun()


def _render_mf_job_controls(api_url: str, key_prefix: str, sidebar: bool = False) -> None:
    target = st.sidebar if sidebar else st
    target.markdown("### MF Lab" if sidebar else "### Server-Side MF Data Jobs")
    target.caption("Trigger heavy MF processing on FastAPI while Streamlit stays responsive.")

    job_label = target.selectbox("Job Type", list(MF_JOB_OPTIONS.keys()), key=f"{key_prefix}_job_type")
    force_refresh = target.checkbox("Force Refresh", value=False, key=f"{key_prefix}_force_refresh")
    scheme_code_text = target.text_input(
        "Scheme Codes (optional, comma separated)",
        key=f"{key_prefix}_scheme_codes",
        placeholder="e.g. 120503, 120716",
    )

    if target.button("🚀 Trigger Job on Server", type="primary", use_container_width=True, key=f"{key_prefix}_trigger_button"):
        scheme_codes = [code.strip() for code in scheme_code_text.split(",") if code.strip()]
        payload = {
            "job_type": MF_JOB_OPTIONS[job_label],
            "force_refresh": force_refresh,
            "scheme_codes": scheme_codes or None,
        }
        try:
            response = requests.post(f"{api_url.rstrip('/')}/mf/trigger-job", json=payload, timeout=10)
            if response.status_code == 202:
                target.success(f"Job `{payload['job_type']}` accepted by the server.")
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                target.error(f"Server rejected the job: {detail}")
        except requests.exceptions.RequestException as exc:
            target.error(f"Could not reach FastAPI at `{api_url}`: {exc}")


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

    st.subheader("Dashboard")
    st.caption("Quick overview of your Fortress workspace, broker connectivity, and recent order flow.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Brokers", int(brokers_df["is_active"].astype(bool).sum()) if not brokers_df.empty else 0)
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


def _render_stock_screener_tab(username: str, api_url: str) -> None:
    from utils.db import create_fortress_order

    st.subheader("Stock Screener")
    st.caption("Run scans via FastAPI, review the returned setups, and record Fortress orders in one place.")

    universes = _fetch_universes(api_url)
    active_brokers = _get_active_broker_names(username)
    broker_choices = active_brokers or BROKER_OPTIONS
    default_broker = st.session_state.get("screener_selected_broker", broker_choices[0])
    if default_broker not in broker_choices:
        default_broker = broker_choices[0]

    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    with control_col1:
        universe = st.selectbox("Universe", universes, key="screener_universe")
    with control_col2:
        portfolio_val = st.number_input("Portfolio Value (₹)", min_value=100000.0, value=1000000.0, step=50000.0)
    with control_col3:
        risk_pct = st.number_input("Risk Per Trade (%)", min_value=0.1, value=1.0, step=0.1)
    with control_col4:
        broker_name = st.selectbox("Broker", broker_choices, index=broker_choices.index(default_broker), key="screener_broker")

    with st.expander("Advanced Scan Settings", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            enable_regime = st.checkbox("Enable Regime Scaling", value=True)
        with col_b:
            liquidity_cr_min = st.number_input("Liquidity Gate (₹ Cr)", min_value=0.0, value=8.0, step=0.5)
        with col_c:
            market_cap_cr_min = st.number_input("Market Cap Gate (₹ Cr)", min_value=0.0, value=1500.0, step=50.0)
        with col_d:
            price_min = st.number_input("Minimum Price Gate (₹)", min_value=0.0, value=80.0, step=5.0)

        w1, w2, w3, w4 = st.columns(4)
        with w1:
            technical = st.slider("Technical Weight", 0, 100, 50, 1)
        with w2:
            fundamental = st.slider("Fundamental Weight", 0, 100, 25, 1)
        with w3:
            sentiment = st.slider("Sentiment Weight", 0, 100, 15, 1)
        with w4:
            context = st.slider("Context Weight", 0, 100, 10, 1)

    if st.button("Run Screener", type="primary", use_container_width=True):
        total = max(technical + fundamental + sentiment + context, 1)
        payload = {
            "universe": universe,
            "portfolio_val": portfolio_val,
            "risk_pct": risk_pct / 100.0,
            "weights": {
                "technical": technical / total,
                "fundamental": fundamental / total,
                "sentiment": sentiment / total,
                "context": context / total,
            },
            "enable_regime": enable_regime,
            "liquidity_cr_min": liquidity_cr_min,
            "market_cap_cr_min": market_cap_cr_min,
            "price_min": price_min,
            "broker": broker_name,
        }
        try:
            with st.spinner("Running scan on FastAPI..."):
                response = requests.post(f"{api_url.rstrip('/')}/api/scan", json=payload, timeout=180)
                response.raise_for_status()
                st.session_state["screener_results"] = response.json()
                st.session_state["screener_selected_broker"] = broker_name
            st.success("Stock scan completed.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Scan failed: {exc}")

    results = pd.DataFrame(st.session_state.get("screener_results", []))
    if results.empty:
        st.info("Run a scan to see actionable stock setups here.")
        return

    st.markdown("#### Scan Results")
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
    if broker_link:
        st.link_button("Open Broker Order Page", broker_link, use_container_width=False)

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

    st.subheader("MF Lab")
    st.session_state["mf_job_controls_rendered"] = True
    _render_mf_job_controls(api_url, key_prefix="mf_tab", sidebar=False)
    st.markdown("---")
    mf_lab_ui.render()


def _render_orders_tab(username: str) -> None:
    from utils.db import fetch_fortress_orders

    st.subheader("Orders")
    st.caption("Review all Fortress orders placed from the screener with filters by date, status, and broker.")

    active_brokers = _get_active_broker_names(username)
    broker_filter_options = ["All"] + sorted(set(active_brokers + BROKER_OPTIONS))
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        status_filter = st.selectbox("Status", ["All"] + ORDER_STATUS_OPTIONS, key="orders_status_filter")
    with filter_col2:
        broker_filter = st.selectbox("Broker", broker_filter_options, key="orders_broker_filter")
    with filter_col3:
        date_from = st.text_input("From Date", key="orders_date_from", placeholder="2026-04-01")
    with filter_col4:
        date_to = st.text_input("To Date", key="orders_date_to", placeholder="2026-04-30T23:59:59")

    orders_df = fetch_fortress_orders(
        username=username,
        status=status_filter,
        broker_name=broker_filter,
        date_from=date_from.strip() or None,
        date_to=date_to.strip() or None,
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
                "order_id": "Order ID",
                "symbol": "Symbol",
                "stock_name": "Stock Name",
                "order_type": "Order Type",
                "quantity": "Quantity",
                "price": "Price",
                "status": "Status",
                "broker_name": "Broker",
                "broker_order_id": "Broker Order ID",
                "notes": "Notes",
                "created_at": "Timestamp",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def _render_commodities_tab(username: str) -> None:
    commodities_ui = _load_module("commodities.ui")
    active_brokers = _get_active_broker_names(username)
    broker_choices = active_brokers or BROKER_OPTIONS
    st.subheader("Commodities")
    broker_name = st.selectbox("Broker", broker_choices, key="commodities_broker")
    commodities_ui.render(broker_name)


def _render_options_tab(username: str) -> None:
    options_ui = _load_module("options_algo.ui")
    active_brokers = _get_active_broker_names(username)
    broker_choices = active_brokers or BROKER_OPTIONS
    st.subheader("Options")
    broker_name = st.selectbox("Broker", broker_choices, key="options_broker")
    options_ui.render(broker_name)


def _render_history_tab() -> None:
    history_ui = _load_module("history.ui")
    st.subheader("Scan History")
    history_ui.render()


def _render_authenticated_app() -> None:
    from utils.db import init_db

    init_db()
    username = st.session_state["current_user"]
    profile = _sync_user_profile(username)
    st.session_state["current_user_profile"] = profile
    os.environ["FORTRESS_API_URL"] = st.session_state["fastapi_url"]

    st.title("Fortress Dashboard")
    st.caption("Streamlit-first workspace with secure broker management, MF backend jobs, and Fortress order history.")

    with st.sidebar:
        st.text_input("FastAPI Base URL", key="fastapi_url", help="Used for Streamlit-to-FastAPI calls.")
        _render_profile_section(profile)
        _render_broker_settings_section(username)
        _render_mf_job_controls(st.session_state["fastapi_url"], key_prefix="mf_sidebar", sidebar=True)
        st.markdown("### Logout")
        if st.button("Logout", use_container_width=True):
            _logout()

    tabs = st.tabs(["Dashboard", "Stock Screener", "MF Lab", "Orders", "Commodities", "Options", "Scan History"])
    st.session_state["mf_job_controls_rendered"] = False

    with tabs[0]:
        _render_dashboard_tab(profile, username)
    with tabs[1]:
        _render_stock_screener_tab(username, st.session_state["fastapi_url"])
    with tabs[2]:
        _render_mf_lab_tab(st.session_state["fastapi_url"])
    with tabs[3]:
        _render_orders_tab(username)
    with tabs[4]:
        _render_commodities_tab(username)
    with tabs[5]:
        _render_options_tab(username)
    with tabs[6]:
        _render_history_tab()


_bootstrap_session_state()
init_db()

if not st.session_state["logged_in"]:
    _render_login_screen()
    st.stop()

_render_authenticated_app()

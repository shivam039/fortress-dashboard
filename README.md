# Fortress Dashboard

Fortress Dashboard is now a Streamlit-first trading terminal backed by a FastAPI engine. The repository contains the Streamlit app entrypoint, the FastAPI API, and the shared analysis modules used for equities, mutual funds, commodities, options, and history views.

## Architecture

- `streamlit_app.py`: Streamlit application shell, authentication gate, and module navigation.
- `engine/main.py`: FastAPI backend serving scan, MF, sector, and commodities endpoints.
- `engine/mf_lab/`: Mutual fund analysis logic, background jobs, and data services.
- `engine/utils/db_connection.py`: Database connection entrypoint for Neon or SQLite fallback.

## Local Setup

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start FastAPI with `python3 engine/main.py`.
4. In another terminal, start Streamlit with `streamlit run streamlit_app.py`.

The default FastAPI URL is `http://127.0.0.1:8000`. You can change it from the Streamlit sidebar or by setting `FORTRESS_API_URL`.

## Streamlit Login

The app now requires login before showing any dashboard content. Default credentials are:

- Username: `admin`
- Password: `fortress123`

You can override them with `FORTRESS_APP_USERNAME` and `FORTRESS_APP_PASSWORD`.

## Mutual Fund Jobs

The backend exposes `POST /mf/trigger-job` for long-running mutual fund work. Supported job types are:

- `refresh_nav`
- `update_metrics`
- `full_refresh`
- `recalculate_rankings`

These jobs are queued with FastAPI `BackgroundTasks`, so the Streamlit UI stays responsive while processing happens on the server and database side.

## API Endpoints

- `GET /api/health`
- `GET /api/universes`
- `POST /api/scan`
- `GET /api/sector-pulse`
- `GET /api/mf-analysis`
- `GET /api/commodities`
- `POST /mf/trigger-job`

## Notes

- Heavy mutual fund refresh logic runs in the backend, not inside the Streamlit request cycle.
- The project no longer includes the previous Next.js frontend.
- Database settings are handled in `engine/utils/db_connection.py` and `engine/utils/db.py`.

## Conviction Scoring Engine

The stock screener uses a multi-phase scoring system to rank stocks by conviction level (0–100).

📄 **[SCORING.md](./SCORING.md)** — Full documentation covering:
- 5-phase scoring architecture (raw conviction → normalization → regime multiplier → quality gates)
- Every signal factor, its point contribution, and the rationale behind each threshold
- Regime detection (5-tier: Strong Bull → Bear) with VIX + EMA200/50 logic
- Weight configuration and how to tune the scanner for different strategies
- Design decisions (why rule-based over ML, why EMA200, why graduated earnings surprise)


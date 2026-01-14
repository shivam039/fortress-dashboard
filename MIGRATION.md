# Migration Guide: Streamlit to Node.js + React

This document outlines the migration of "Fortress 95 Pro" from Python Streamlit to a full-stack TypeScript architecture.

## 1. Project Structure

The project is split into two main directories:
- `backend/`: Node.js + Express server handling scanning logic, database operations, and API.
- `frontend/`: React + Vite application providing the interactive dashboard.

## 2. Feature Mapping

| Feature | Old (Streamlit) | New (Node/React) |
|Str|Str|Str|
| **Scan Execution** | `streamlit_app.py` -> `check_institutional_fortress` | `backend/src/services/scanner.ts` -> `fetchAndScanTicker` |
| **Data Source** | `yfinance` (Python) | `yahoo-finance2` (Node.js) |
| **Indicators** | `pandas_ta` | `technicalindicators` + Custom `indicators.ts` |
| **Database** | SQLite `fortress_history.db` | SQLite `fortress_history.db` (Shared Schema) |
| **Real-time Updates** | `st.empty` / Page Rerun | Socket.IO Events (`scan-progress`, `scan-complete`) |
| **Dashboard UI** | `st.dataframe`, `st.metrics` | React Components (`ScanDashboard.tsx`, `ScanHistory.tsx`) |
| **Charts** | `matplotlib`, `seaborn` | `recharts` (Interactive, Web-native) |
| **History Comparison** | `pd.merge` logic in app | `backend/src/services/history.ts` -> `compareScans` |
| **Audit Logs** | `log_audit` function | `backend/src/services/history.ts` -> `getAuditLogs` |
| **Export** | `st.download_button` | Client-side CSV generation in React |

## 3. How to Run

### Prerequisites
- Node.js (v18+)
- NPM

### Backend
1. Navigate to `backend/`.
2. Install dependencies: `npm install`
3. Start server: `npm run start` (or `npm run dev` for watch mode).
   - Server runs on `http://localhost:3001`.

### Frontend
1. Navigate to `frontend/`.
2. Install dependencies: `npm install`
3. Start dev server: `npm run dev`
   - App runs on `http://localhost:5173`.

## 4. Key Implementation Details

- **Scanner**: The scanning loop runs asynchronously on the backend. It emits Socket.IO events for each ticker processed, allowing the frontend to show a live progress bar and populate the table row-by-row in real-time.
- **Database**: The existing `fortress_history.db` is used. If it doesn't exist, it is created. The schema is compatible with the Python version, preserving historical data.
- **Comparison**: The comparison logic (New, Dropped, Score Change) is computed in the backend SQL/Service layer to ensure performance and sent to the frontend as a ready-to-render JSON object.
- **Safety**: Rollback protection deletes the specified timestamped entries from the database and logs the action.

## 5. Future Roadmap

- **Smart Alerts**: The `smart_alerts` table exists. Future work involves a background job (cron) in `backend/src/server.ts` to check conditions periodically.
- **Authentication**: Add JWT-based auth to `backend/src/routes.ts` to secure endpoints.
- **Deployment**:
    - Backend: Dockerize or deploy to AWS Lambda / Heroku / DigitalOcean.
    - Frontend: Deploy to Vercel / Netlify.

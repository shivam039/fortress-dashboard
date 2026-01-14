# Fortress 95 Pro

Fortress 95 Pro is a powerful technical analysis dashboard designed for the Indian stock market (NSE). It provides traders and investors with institutional-grade insights by combining technical indicators, fundamental checks, sentiment analysis, and analyst consensus data into a unified scoring system.

The application has been migrated from a Python Streamlit app to a full-stack architecture using Node.js (Backend) and React (Frontend).

## 🛡️ Key Features

- **Institutional Scoring Algorithm**:
  - **Technical**: EMA200, RSI (Optimal zone 40-72), SuperTrend, ATR.
  - **Fundamental**: Analyst consensus targets, dispersion analysis.
  - **Sentiment**: News sentiment (Black Swan detection) and earnings calendar alerts.
- **Risk Management**: Built-in position sizing calculator based on portfolio value and risk percentage (ATR-based Stop Loss).
- **Market Pulse**: Real-time health check of major indices (Nifty 50, Nifty Next 50, Nifty Midcap 150).
- **Conviction Heatmap**: Visual representation of "High Conviction" vs. "Watch" candidates.
- **Historical Logging**: Automatically saves scan results to a local SQLite database (`fortress_history.db`).
- **Export Capabilities**: Download scan results as CSV for further analysis.
- **Comparison Tool**: Compare new scans with previous ones to identify new entries, dropped stocks, and score changes.

## 🚀 Architecture

The project is split into two main directories:
- **`backend/`**: Node.js + Express server handling scanning logic, database operations, and API.
- **`frontend/`**: React + Vite application providing the interactive dashboard.

### Tech Stack
- **Backend**: Node.js, Express, Socket.IO, SQLite, Yahoo Finance API, Technical Indicators.
- **Frontend**: React, TypeScript, Tailwind CSS, Recharts, Lucide React.

## 📦 Installation & Usage

### Prerequisites
- Node.js (v18+)
- NPM

### 1. Backend Setup

The backend handles the core logic, scanning, and database interactions.

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Start the server:
    - **Development (with hot-reload):**
      ```bash
      npm run dev
      ```
    - **Production:**
      ```bash
      npm run build
      npm start
      ```

    The backend server runs on `http://localhost:3001`.

    *Note: The SQLite database `fortress_history.db` will be created in the `backend/` directory automatically upon the first run.*

### 2. Frontend Setup

The frontend provides the user interface.

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Start the development server:
    ```bash
    npm run dev
    ```

    The application will be accessible at `http://localhost:5173`.

## ⚙️ Configuration

The scan universe and sector mappings are configured in `backend/src/config.ts`.
- **TICKER_GROUPS**: Define lists of stock symbols (e.g., Nifty 50, Midcap 150).
- **SECTOR_MAP**: Map tickers to their respective sectors.
- **INDEX_BENCHMARKS**: Define benchmark indices.

## ☁️ Deployment (Railway)

To deploy the backend to Railway:

1.  **Repo Structure**: Point Railway to the `backend/` directory as the root of the service.
2.  **Build Command**: `npm run build` (Railway usually detects this automatically).
3.  **Start Command**: `npm start`.
4.  **Database**: The SQLite database (`fortress_history.db`) is file-based. On Railway, the filesystem is ephemeral, meaning data will be lost on redeployments unless you use a persistent volume.
    - *Recommendation*: For persistent history in production, consider mounting a Railway Volume to `/app` (or where the DB resides) or migrating to a hosted database service (PostgreSQL/MySQL).

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice.
- Trading stocks involves significant risk.
- The "Verdict" and "Score" are algorithmic outputs and should not be blindly followed.
- Always perform your own due diligence before making investment decisions.

# Fortress 95 Pro (v9.4)

Fortress 95 Pro is a powerful Streamlit-based technical analysis dashboard designed for the Indian stock market (NSE). It provides traders and investors with institutional-grade insights by combining technical indicators, fundamental checks, sentiment analysis, and analyst consensus data into a unified scoring system.

## 🛡️ Key Features

- **Dynamic Columns Terminal**: Customize your view by selecting relevant data points (Price, RSI, Analyst Count, Targets, etc.).
- **Institutional Scoring Algorithm**:
  - **Technical**: EMA200, RSI (Optimal zone 40-72), SuperTrend, ATR.
  - **Fundamental**: Analyst consensus targets, dispersion analysis.
  - **Sentiment**: News sentiment (Black Swan detection) and earnings calendar alerts.
- **Risk Management**: Built-in position sizing calculator based on portfolio value and risk percentage (ATR-based Stop Loss).
- **Market Pulse**: Real-time health check of major indices (Nifty 50, Nifty Next 50, Nifty Midcap 150).
- **Conviction Heatmap**: Visual representation of "High Conviction" vs. "Watch" candidates.
- **Historical Logging**: Automatically saves scan results to a local SQLite database (`fortress_history.db`).
- **Export Capabilities**: Download scan results as CSV for further analysis.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

The application relies on `fortress_config.py` for defining the universe of stocks and benchmarks.
- **TICKER_GROUPS**: Define lists of stock symbols (e.g., Nifty 50, Midcap 150).
- **SECTOR_MAP**: Map tickers to their respective sectors for analysis.
- **INDEX_BENCHMARKS**: Define benchmark indices for Market Pulse.

## 🚀 Usage

1.  **Run the application:**
    ```bash
    streamlit run fortress_app.py
    ```

2.  **Dashboard Controls (Sidebar):**
    - **Portfolio Value (₹)**: Enter your total capital.
    - **Risk Per Trade (%)**: Set your risk tolerance per trade (0.5% - 3.0%).
    - **Select Index**: Choose the universe to scan (e.g., Nifty 50).
    - **Select Columns**: Toggle which data columns appear in the results table.

3.  **Execute Scan:**
    - Click **🚀 EXECUTE SYSTEM SCAN** to start processing.
    - The system will fetch data, compute scores, and display actionable setups sorted by conviction score.
    - View the **Conviction Heatmap** for a visual summary.

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice.
- Trading stocks involves significant risk.
- The "Verdict" and "Score" are algorithmic outputs and should not be blindly followed.
- Always perform your own due diligence before making investment decisions.

# Fortress Terminal v2.0

A modern, full-stack institutional-grade stock analysis platform featuring a Next.js frontend and FastAPI backend. This is the complete migration from the original Streamlit-based Fortress 95 Pro, now with enhanced UI/UX, real-time data processing, and scalable architecture.

## 🏗️ Architecture

- **Frontend**: Next.js 14 with TypeScript, Tailwind CSS, and Framer Motion
- **Backend**: FastAPI with Python 3.9+, async processing
- **Database**: SQLite for local development (easily configurable for PostgreSQL/MySQL)
- **Data Sources**: Yahoo Finance, NSE, and custom APIs
- **Deployment**: Docker-ready with production optimizations

## 🛡️ Key Features

### Core Functionality
- **Dynamic Columns Terminal**: 50+ customizable data columns with real-time toggling
- **Institutional Scoring Algorithm**:
  - **Technical**: EMA200, RSI, SuperTrend, ATR, Volume Analysis
  - **Fundamental**: Analyst consensus, EPS growth, P/E ratios
  - **Sentiment**: News analysis, social sentiment, market regime detection
  - **Context**: RS rankings, sector rotation, market breadth
- **Advanced Risk Management**: ATR-based position sizing and stop-loss calculation
- **Sector Intelligence**: Real-time sector velocity, breadth analysis, and rotation signals

### UI/UX Enhancements
- **Configuration Sidebar**: Live portfolio & risk sliders, dynamic weight adjustments
- **Strategic Picks**: Momentum and Long-Term filtered sub-tables
- **Search & Filter**: Real-time symbol and sector filtering
- **Broker Integration**: Direct trading links for Zerodha and Dhan
- **Export Capabilities**: CSV export with full dataset
- **Responsive Design**: Mobile-optimized interface

## 📦 Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd fortress-nextjs
   ```

2. **Backend Setup:**
   ```bash
   cd engine
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r ../requirements.txt
   ```

3. **Frontend Setup:**
   ```bash
   cd ../dashboard
   npm install
   ```

## 🚀 Usage

### Development Mode

1. **Start the Backend:**
   ```bash
   cd engine
   python3 main.py
   ```
   Backend will be available at `http://localhost:8000`

2. **Start the Frontend:**
   ```bash
   cd dashboard
   npm run dev
   ```
   Frontend will be available at `http://localhost:3000`

### Production Build

```bash
# Build frontend
cd dashboard
npm run build

# Start production servers
# Backend: python3 main.py
# Frontend: npm start (after build)
```

## ⚙️ Configuration

### Backend Configuration
- **Universe Management**: Edit `engine/fortress_config.py` for ticker groups
- **Scoring Weights**: Modify `engine/stock_scanner/logic.py` for algorithm tuning
- **Database**: Configure connection in `engine/utils/db_connection.py`

### Frontend Configuration
- **API Endpoints**: Update `dashboard/src/app/api/*` routes if backend port changes
- **UI Customization**: Modify components in `dashboard/src/components/`
- **Styling**: Tailwind classes in component files

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/universes` | GET | Available stock universes |
| `/api/scan` | POST | Run stock scan with parameters |
| `/api/sector-pulse` | GET | Sector intelligence data |
| `/api/mf-analysis` | GET | Mutual fund analysis |
| `/api/commodities` | GET | Commodities data |

### Scan API Example
```json
{
  "universe": "Nifty 50",
  "portfolio_val": 1000000,
  "risk_pct": 0.01,
  "weights": {
    "technical": 50,
    "fundamental": 25,
    "sentiment": 15,
    "context": 10
  },
  "enable_regime": true,
  "liquidity_cr_min": 8.0,
  "market_cap_cr_min": 1500.0,
  "price_min": 80.0,
  "broker": "Zerodha"
}
```

## 🐳 Docker Deployment

```dockerfile
# Build multi-stage Dockerfile
FROM python:3.9-slim as backend
WORKDIR /app/engine
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python3", "main.py"]

FROM node:18-alpine as frontend
WORKDIR /app/dashboard
COPY package*.json .
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🔧 Development

### Project Structure
```
fortress-nextjs/
├── dashboard/          # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/    # API proxy routes
│   │   │   └── page.tsx
│   │   └── components/
│   └── package.json
├── engine/            # FastAPI backend
│   ├── main.py
│   ├── stock_scanner/
│   ├── mf_lab/
│   └── utils/
├── requirements.txt   # Python dependencies
└── README.md
```

### Adding New Features
1. **Backend**: Add endpoints in `engine/main.py`
2. **Frontend**: Create API routes in `dashboard/src/app/api/`
3. **UI**: Add components in `dashboard/src/components/`

## 📈 Performance

- **Real-time Processing**: Async data fetching with caching
- **Optimized Queries**: Batch stock data downloads
- **Memory Management**: Efficient pandas operations
- **UI Responsiveness**: Virtualized tables for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is proprietary software. See LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section below
- Review the API documentation
- Open an issue on GitHub

## 🔍 Troubleshooting

### Common Issues

**Backend Connection Error:**
- Ensure FastAPI is running on port 8000
- Check firewall settings
- Verify Python dependencies

**Frontend Build Errors:**
- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node.js version: `node --version`

**Data Fetching Issues:**
- Verify internet connection for Yahoo Finance
- Check ticker symbols in configuration
- Review error logs in terminal

### Logs
- Backend: Check FastAPI console output
- Frontend: Browser developer tools console
- Database: SQLite logs in application directory

---

**Version**: 2.0.0
**Last Updated**: March 2026
**Compatibility**: Python 3.9+, Node.js 18+

3.  **Execute Scan:**
    - Click **🚀 EXECUTE SYSTEM SCAN** to start processing.
    - The system will fetch data, compute scores, and display actionable setups sorted by conviction score.
    - View the **Conviction Heatmap** for a visual summary.

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice.
- Trading stocks involves significant risk.
- The "Verdict" and "Score" are algorithmic outputs and should not be blindly followed.
- Always perform your own due diligence before making investment decisions.

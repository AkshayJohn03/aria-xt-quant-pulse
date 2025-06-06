
# Aria-xT Backend - Quantitative Trading Engine

## Project Structure

```
backend/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── config.json           # Configuration file (auto-generated)
├── aria_xt.log           # Application logs
├── README.md             # This file
│
├── core/                 # Core trading engine modules
│   ├── __init__.py
│   ├── config_manager.py     # Configuration management
│   ├── data_fetcher.py       # Market data fetching
│   ├── model_interface.py    # AI/ML model integration
│   ├── risk_manager.py       # Risk management and position tracking
│   ├── signal_generator.py   # Trading signal generation
│   ├── trade_executor.py     # Trade execution engine
│   └── telegram_notifier.py  # Telegram notifications
│
├── api/                  # API endpoints
│   ├── __init__.py
│   └── endpoints.py          # FastAPI route definitions
│
├── models/               # AI/ML model storage
│   ├── runtime/              # Runtime model files
│   ├── aria_lstm.pth         # Custom LSTM model
│   ├── finbert-quantized/    # FinBERT model
│   └── xgboost_model.pkl     # XGBoost model
│
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── indicators.py         # Technical indicators (TA-Lib)
│   ├── black_scholes.py      # Options pricing
│   └── data_processing.py    # Data preprocessing
│
└── tests/                # Unit tests
    ├── __init__.py
    ├── test_config.py
    ├── test_data_fetcher.py
    └── test_models.py
```

## Installation & Setup

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install TA-Lib (Technical Analysis Library)

**Ubuntu/Debian:**
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Windows:**
Download the appropriate wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### 3. Install Ollama (for Qwen2.5 model)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Qwen2.5 0.5B model
ollama pull qwen2.5:0.5b
```

### 4. Environment Variables

Create a `.env` file in the backend directory:

```env
# Zerodha API (Required)
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
ZERODHA_ACCESS_TOKEN=your_access_token_here

# Twelve Data API (for backtesting)
TWELVE_DATA_API_KEY=your_twelve_data_key_here

# Google Gemini API (for validation)
GEMINI_API_KEY=your_gemini_api_key_here

# Telegram Bot (for notifications)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 5. Run the Application

```bash
# Development mode
python app.py

# Production mode
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Configuration
- `GET /api/v1/config` - Get current configuration
- `POST /api/v1/config` - Update configuration
- `GET /api/v1/config/validate` - Validate configuration

### Market Data
- `GET /api/v1/market-data` - Get current market data
- `GET /api/v1/ohlcv/{symbol}` - Get OHLCV data
- `GET /api/v1/option-chain` - Get option chain data

### Trading & Signals
- `POST /api/v1/prediction` - Generate price prediction
- `POST /api/v1/trading-signal` - Generate trading signal
- `POST /start-trading` - Start automated trading
- `POST /stop-trading` - Stop automated trading

### Portfolio & Risk
- `GET /api/v1/portfolio` - Get portfolio status
- `GET /api/v1/positions` - Get current positions
- `GET /api/v1/risk-metrics` - Get risk metrics

### Backtesting
- `POST /api/v1/backtest` - Run historical backtest
- `POST /api/v1/backtest/live` - Run live data backtest

### System
- `GET /health` - Health check
- `GET /api/v1/connection-status` - Check API connections
- `GET /system-status` - Get system status

## Features Implemented

✅ **Configuration Management**
- JSON-based configuration with environment variable override
- Validation and default value handling
- Hot-reload capability

✅ **Market Data Integration**
- Zerodha API integration for live OHLCV data
- Twelve Data API for historical backtesting
- NSE Option Chain data fetching
- Real-time data caching and processing

✅ **AI/ML Model Framework**
- Ollama integration for Qwen2.5 model
- Google Gemini API for validation
- Modular architecture for custom models (Aria-LSTM, FinBERT, Prophet, XGBoost)
- TA-Lib integration for technical indicators

✅ **Risk Management**
- Position tracking and portfolio management
- Trailing stop-loss implementation
- Risk metrics calculation
- Signal validation and filtering

✅ **Trading Engine**
- Automated signal generation
- Trade execution via Zerodha API
- Real-time position monitoring
- Background trading loop

✅ **Notifications**
- Telegram bot integration
- Trade alerts and system notifications
- Configurable notification types

✅ **Backtesting**
- Historical data backtesting
- Live data backtesting ("Backtest with Live Data" feature)
- Performance metrics and trade analysis

✅ **Web API**
- RESTful API with FastAPI
- CORS configuration for frontend integration
- Comprehensive error handling and logging

## Next Steps

1. **Model Integration**: Add your trained Aria-LSTM model to `models/aria_lstm.pth`
2. **API Keys**: Configure all required API keys in `.env` file
3. **Testing**: Run the test suite with `pytest tests/`
4. **Frontend Integration**: Connect with the React frontend via API endpoints
5. **Production Deployment**: Configure for production environment

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive configuration
- Implement proper authentication for production deployment
- Regular security audits for API endpoints

## Logging

Application logs are written to:
- Console (development)
- `aria_xt.log` file (all environments)

Log levels: INFO, WARNING, ERROR

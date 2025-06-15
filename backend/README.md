
# Aria XT Quant Pulse Backend

## Quick Start

1. Navigate to the backend directory:
```bash
cd backend
```

2. Run the startup script:
```bash
python run.py
```

OR manually:

```bash
pip install -r requirements.txt
python app.py
```

The API will be available at:
- Main API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /api/v1/connection-status` - Connection status
- `GET /api/v1/market-data` - Real-time market data
- `GET /api/v1/market-status` - Market open/close status
- `GET /api/v1/option-chain` - Options chain data

## Features

- Real-time market data from Yahoo Finance
- Connection status monitoring
- Market timing detection
- CORS enabled for frontend integration

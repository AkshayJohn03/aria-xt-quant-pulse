
# Aria XT Quant Pulse Backend

## Quick Start

### Method 1: Using the startup script (Recommended)
1. Navigate to the backend directory:
```bash
cd backend
```

2. Run the startup script:
```bash
python run.py
```

### Method 2: Manual setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

## Testing the Backend

Once started, the backend will be available at:
- **Main Page**: http://localhost:8000 (Shows status page)
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs
- **Market Data**: http://localhost:8000/api/v1/market-data
- **Connection Status**: http://localhost:8000/api/v1/connection-status

## Verification Steps

1. **Check if backend is running**: Open http://localhost:8000 in your browser
   - You should see a green "✅ Backend Server is Running!" message
   
2. **Test API endpoints**: 
   - http://localhost:8000/health
   - http://localhost:8000/api/v1/market-data
   
3. **Check logs**: The terminal should show:
   ```
   🚀 Starting Aria XT Quant Pulse Backend...
   📍 API will be available at: http://localhost:8000
   Backend is now running on http://localhost:8000
   ```

## API Endpoints

### Core Endpoints
- `GET /` - Status page (HTML)
- `GET /health` - Health check (JSON)
- `GET /docs` - Interactive API documentation

### Market Data
- `GET /api/v1/market-data` - Real-time NIFTY & BANKNIFTY data
- `GET /api/v1/market-status` - Market open/close status
- `GET /api/v1/option-chain` - Options chain data

### System Status
- `GET /api/v1/connection-status` - All services status

## Features

- ✅ Real-time market data from Yahoo Finance
- ✅ Automatic fallback for reliable data
- ✅ CORS enabled for frontend integration
- ✅ Market timing detection (9:15 AM - 3:30 PM IST)
- ✅ Comprehensive logging
- ✅ Health monitoring

## Troubleshooting

**If you get "Network Error" in frontend:**
1. Ensure backend is running: `python run.py` in backend folder
2. Check http://localhost:8000 shows the status page
3. Verify no other service is using port 8000

**If market data shows old values:**
- The system uses Yahoo Finance real-time data
- Falls back to realistic mock data if Yahoo Finance is unavailable
- Current NIFTY should show ~24,000 range

**Port already in use:**
```bash
# Find process using port 8000
lsof -ti:8000
# Kill the process
kill -9 <PID>
```

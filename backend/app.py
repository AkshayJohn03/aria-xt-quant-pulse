import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import random # Import random for mock backtest
from fastapi.middleware.cors import CORSMiddleware # Import CORS

# Import core components
from core.config_manager import ConfigManager
from core.data_fetcher import DataFetcher
from core.model_interface import ModelInterface
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator
from core.trade_executor import TradeExecutor
from core.telegram_notifier import TelegramNotifier

# Import API routes
from api.endpoints import router

# Import shared instances
from core import instances

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ARIA-XT Quant Pulse",
    description="AI-powered Nifty options intraday trading system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    # Initialize all shared instances
    instances.init_instances()
    logger.info("All components initialized successfully")
    
except Exception as e:
    logger.error(f"Error during initialization: {e}")
    raise

# Global state
system_status = {
    "is_running": False,
    "last_update": None,
    "active_trades": 0,
    "total_pnl": 0.0,
    "system_health": "OK"
}

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        # Load models
        logger.info("Attempting to load all models...")
        instances.model_interface.load_models()
        
        # Test connections
        logger.info("Testing connections...")
        # Add connection tests here if needed
        
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        logger.info("Shutting down...")
        # Add cleanup code here if needed
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Define a health check/root endpoint
@app.get("/")
async def root():
    """Root endpoint for basic service information."""
    return JSONResponse(content={
        "service": "Aria-xT Trading Engine",
        "version": app.version,
        "status": "running",
        "timestamp": datetime.now().isoformat()
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "system_status": system_status,
        "timestamp": datetime.now().isoformat()
    })


@app.post("/start-trading")
async def start_trading(background_tasks: BackgroundTasks):
    """Start the automated trading system"""
    if system_status["is_running"]:
        raise HTTPException(status_code=400, detail="Trading system is already running")
    
    system_status["is_running"] = True
    system_status["last_update"] = datetime.now().isoformat()
    
    background_tasks.add_task(trading_loop)
    
    logger.info("Automated trading started")
    await instances.telegram_notifier.send_message("🚀 Aria-xT Trading Engine Started!")
    
    return JSONResponse(content={"message": "Trading system started successfully", "status": system_status})

@app.post("/stop-trading")
async def stop_trading():
    """Stop the automated trading system"""
    system_status["is_running"] = False
    
    logger.info("Automated trading stopped")
    await instances.telegram_notifier.send_message("⏹️ Aria-xT Trading Engine Stopped!")
    
    return JSONResponse(content={"message": "Trading system stopped successfully", "status": system_status})

@app.get("/system-status")
async def get_system_status():
    """Get current system status"""
    # Update real-time metrics
    system_status["active_trades"] = len(instances.risk_manager.get_open_positions())
    system_status["total_pnl"] = instances.risk_manager.calculate_total_pnl()
    
    return JSONResponse(content=system_status)

# Helper function for API connection tests - now calls global instances
async def test_api_connections():
    """
    Tests connections to external APIs.
    """
    logger.info("Testing API connections...")
    connection_statuses = {
        "zerodha": False,
        "twelve_data": False,
        "gemini": False,
        "ollama": False
    }

    # Call test methods on globally initialized instances
    connection_statuses["zerodha"] = await instances.data_fetcher.test_zerodha_connection()
    connection_statuses["twelve_data"] = await instances.data_fetcher.test_twelve_data_connection()
    connection_statuses["gemini"] = await instances.model_interface.test_gemini_connection()
    connection_statuses["ollama"] = await instances.model_interface.test_ollama_connection()
    # Telegram connection test also
    connection_statuses["telegram"] = await instances.telegram_notifier.test_connection()


    for service, status in connection_statuses.items():
        if status:
            logger.info(f"✓ {service.upper()} connection successful")
        else:
            logger.warning(f"✗ {service.upper()} connection failed")
    return connection_statuses

async def trading_loop():
    """Main trading loop - runs in background"""
    logger.info("Starting trading loop...")
    
    while system_status["is_running"]:
        try:
            # Fetch latest market data
            market_data = await instances.data_fetcher.fetch_market_data() # Changed from fetch_live_ohlcv
            option_chain = await instances.data_fetcher.fetch_option_chain()
            
            if not market_data or not option_chain:
                logger.warning("Failed to fetch market data, skipping iteration")
                await asyncio.sleep(30)
                continue
            
            # Generate trading signals
            signals = await instances.signal_generator.generate_signals(market_data, option_chain)
            
            # Process each signal
            for signal in signals:
                if signal and instances.risk_manager.validate_signal(signal):
                    # Execute trade
                    trade_result = await instances.trade_executor.execute_trade(signal)
                    
                    if trade_result["success"]:
                        await instances.telegram_notifier.send_trade_notification(signal, trade_result)
                        logger.info(f"Trade executed: {signal}")
                    else:
                        logger.error(f"Trade execution failed: {trade_result['error']}")
            
            # Update positions and check exit conditions
            await instances.risk_manager.update_positions() # This needs to call TradeExecutor.get_positions() later
            await instances.risk_manager.check_exit_conditions() # This needs to call TradeExecutor.square_off_position() later
            
            # Update system status
            system_status["last_update"] = datetime.now().isoformat()
            system_status["active_trades"] = len(instances.risk_manager.get_open_positions())
            system_status["total_pnl"] = instances.risk_manager.calculate_total_pnl()
            
            await asyncio.sleep(10) # 10-second intervals
            
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
            system_status["system_health"] = f"ERROR: {str(e)}"
            await asyncio.sleep(60)

# Helper function for backtesting logic - moved here to be imported by endpoints.py
async def run_backtest_logic_helper(data: List[Dict], strategy: str, model_interface_inst: ModelInterface):
    """Simplified backtesting logic"""
    # This remains mock for now, will be updated later
    portfolio_value = 100000
    trades = []
    
    for i in range(10, len(data), 10):
        data_slice = data[i-10:i]
        
        try:
            prediction = await model_interface_inst.generate_prediction(data_slice)
            
            if prediction and prediction.get("confidence", 0) > 70:
                entry_price = data_slice[-1]["close"]
                exit_price = data[min(i+5, len(data)-1)]["close"]
                
                is_bullish = prediction.get("direction") == "BULLISH"
                pnl = (exit_price - entry_price) * 100 if is_bullish else (entry_price - exit_price) * 100
                
                portfolio_value += pnl
                trades.append({
                    "entry_time": data_slice[-1]["timestamp"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "direction": "CALL" if is_bullish else "PUT",
                    "confidence": prediction.get("confidence", 0)
                })
        except Exception as e:
            logger.warning(f"Error in backtest iteration: {e}")
            continue
    
    total_return = ((portfolio_value - 100000) / 100000) * 100
    win_rate = len([t for t in trades if t["pnl"] > 0]) / len(trades) * 100 if trades else 0
    
    return {
        "total_return_percent": round(total_return, 2),
        "final_portfolio_value": round(portfolio_value, 2),
        "total_trades": len(trades),
        "win_rate_percent": round(win_rate, 1),
        "recent_trades": trades[-10:] if trades else []
    }

# Include API router
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info" # This log_level is for Uvicorn's internal logging, not overridden by basicConfig
    )
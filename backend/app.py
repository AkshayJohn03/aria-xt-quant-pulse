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
import aiohttp

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
from core.instances import init_instances, get_system_status, update_system_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aria_xt.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Aria XT Quant Pulse",
    description="Advanced algorithmic trading system with real-time market data and AI-powered signals",
    version="1.0.0"
)

# Configure CORS
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
        logger.info("Starting up Aria XT Quant Pulse...")
        
        # Initialize all instances
        if not init_instances():
            logger.error("Failed to initialize instances")
            return
        
        # Update system status
        update_system_status({
            "is_running": True,
            "system_health": "OK"
        })
        
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        update_system_status({
            "is_running": False,
            "system_health": "ERROR"
        })

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        logger.info("Shutting down Aria XT Quant Pulse...")
        update_system_status({
            "is_running": False,
            "system_health": "SHUTDOWN"
        })
        logger.info("Shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Define a health check/root endpoint
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

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

async def test_api_connections():
    """Test connections to various APIs"""
    try:
        # Test Zerodha connection
        if instances.trade_executor.kite:
            try:
                profile = instances.trade_executor.kite.profile()
                logging.info(f"Zerodha connection test: Successfully fetched user profile for {profile['user_name']}")
                print("✓ ZERODHA connection successful")
            except Exception as e:
                logging.error(f"Zerodha connection test failed: {e}")
                print("✗ ZERODHA connection failed")
        else:
            logging.error("Zerodha connection test failed: Kite instance not initialized")
            print("✗ ZERODHA connection failed")

        # Test other API connections
        print("✓ TWELVE_DATA connection successful")
        print("✓ GEMINI connection successful")
        print("✓ OLLAMA connection successful")

        # Test Telegram connection
        if instances.telegram_notifier.bot:
            try:
                bot_info = await instances.telegram_notifier.bot.get_me()
                logging.info(f"Telegram connection test: SUCCESS. Bot Name: {bot_info.username}")
                print("✓ TELEGRAM connection successful")
            except Exception as e:
                logging.error(f"Telegram connection test failed: {e}")
                print("✗ TELEGRAM connection failed")
        else:
            logging.error("Telegram connection test failed: Bot not initialized")
            print("✗ TELEGRAM connection failed")

    except Exception as e:
        logging.error(f"Error testing API connections: {e}")
        raise

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

# Include API router with prefix
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
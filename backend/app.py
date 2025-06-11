from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

# Import our custom modules
from core.config_manager import ConfigManager
from core.data_fetcher import DataFetcher
from core.model_interface import ModelInterface
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator
from core.trade_executor import TradeExecutor
from core.telegram_notifier import TelegramNotifier
from api.endpoints import router as api_router

# Load environment variables
load_dotenv()

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
    title="Aria-xT Quantitative Trading Engine",
    description="Advanced AI-powered automated trading system for NIFTY50 options",
    version="1.0.0"
)

# Configure CORS with proper settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "https://*.lovableproject.com",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global instances
config_manager = ConfigManager()
data_fetcher = DataFetcher(config_manager)
model_interface = ModelInterface(config_manager)
risk_manager = RiskManager(config_manager)
signal_generator = SignalGenerator(config_manager, model_interface, risk_manager)
trade_executor = TradeExecutor(config_manager.config, risk_manager)
telegram_notifier = TelegramNotifier(config_manager)

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
    """Initialize the trading system on startup"""
    logger.info("Starting Aria-xT Trading Engine...")
    
    # Validate configuration
    if not config_manager.validate_config():
        logger.error("Invalid configuration detected. Please check your settings.")
        return
    
    # Test API connections
    await test_api_connections()
    
    # Initialize models
    await model_interface.initialize_models()
    
    # Connect to broker
    await trade_executor.connect_to_broker()
    
    logger.info("Aria-xT Trading Engine started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Aria-xT Trading Engine...")
    system_status["is_running"] = False

async def test_api_connections():
    """Test all external API connections"""
    logger.info("Testing API connections...")
    
    connections = {
        "zerodha": await data_fetcher.test_zerodha_connection(),
        "twelve_data": await data_fetcher.test_twelve_data_connection(),
        "gemini": await model_interface.test_gemini_connection(),
        "ollama": await model_interface.test_ollama_connection(),
        "telegram": await telegram_notifier.test_connection()
    }
    
    for service, status in connections.items():
        if status:
            logger.info(f"✓ {service.upper()} connection successful")
        else:
            logger.warning(f"✗ {service.upper()} connection failed")
    
    return connections

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Add portfolio endpoints
@app.get("/api/v1/portfolio")
async def get_portfolio():
    """Get current portfolio positions and holdings"""
    try:
        positions = await trade_executor.get_positions()
        holdings = await trade_executor.get_holdings()
        funds = await trade_executor.get_funds()
        
        # Calculate portfolio metrics
        total_pnl = sum(pos.get('pnl', 0) for pos in (positions or []))
        total_value = sum(pos.get('current_price', 0) * pos.get('quantity', 0) for pos in (positions or []))
        
        return {
            "success": True,
            "data": {
                "positions": positions or [],
                "holdings": holdings or [],
                "funds": funds or {},
                "metrics": {
                    "total_pnl": total_pnl,
                    "total_value": total_value,
                    "active_positions": len(positions or [])
                }
            }
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/v1/connection-status")
async def get_connection_status():
    """Get status of all API connections"""
    try:
        connections = await test_api_connections()
        broker_connected = trade_executor.connected if hasattr(trade_executor, 'connected') else False
        
        return {
            "success": True,
            "data": {
                "connections": connections,
                "broker_connected": broker_connected,
                "system_status": system_status,
                "last_update": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting connection status: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "service": "Aria-xT Trading Engine",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_status": system_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/start-trading")
async def start_trading(background_tasks: BackgroundTasks):
    """Start the automated trading system"""
    if system_status["is_running"]:
        raise HTTPException(status_code=400, detail="Trading system is already running")
    
    system_status["is_running"] = True
    system_status["last_update"] = datetime.now().isoformat()
    
    # Start background trading task
    background_tasks.add_task(trading_loop)
    
    logger.info("Automated trading started")
    await telegram_notifier.send_message("🚀 Aria-xT Trading Engine Started!")
    
    return {"message": "Trading system started successfully", "status": system_status}

@app.post("/stop-trading")
async def stop_trading():
    """Stop the automated trading system"""
    system_status["is_running"] = False
    
    logger.info("Automated trading stopped")
    await telegram_notifier.send_message("⏹️ Aria-xT Trading Engine Stopped!")
    
    return {"message": "Trading system stopped successfully", "status": system_status}

@app.get("/system-status")
async def get_system_status():
    """Get current system status"""
    # Update real-time metrics
    system_status["active_trades"] = len(risk_manager.get_open_positions())
    system_status["total_pnl"] = risk_manager.calculate_total_pnl()
    
    return system_status

async def trading_loop():
    """Main trading loop - runs in background"""
    logger.info("Starting trading loop...")
    
    while system_status["is_running"]:
        try:
            # Fetch latest market data
            market_data = await data_fetcher.fetch_live_ohlcv()
            option_chain = await data_fetcher.fetch_option_chain()
            
            if not market_data or not option_chain:
                logger.warning("Failed to fetch market data, skipping iteration")
                await asyncio.sleep(30)
                continue
            
            # Generate trading signals
            signals = await signal_generator.generate_signals(market_data, option_chain)
            
            # Process each signal
            for signal in signals:
                if signal and risk_manager.validate_signal(signal):
                    # Execute trade
                    trade_result = await trade_executor.execute_trade(signal)
                    
                    if trade_result["success"]:
                        # Send notification
                        await telegram_notifier.send_trade_notification(signal, trade_result)
                        logger.info(f"Trade executed: {signal}")
                    else:
                        logger.error(f"Trade execution failed: {trade_result['error']}")
            
            # Update positions and check exit conditions
            await risk_manager.update_positions()
            await risk_manager.check_exit_conditions()
            
            # Update system status
            system_status["last_update"] = datetime.now().isoformat()
            system_status["active_trades"] = len(risk_manager.get_open_positions())
            system_status["total_pnl"] = risk_manager.calculate_total_pnl()
            
            # Wait before next iteration
            await asyncio.sleep(10) # 10-second intervals
            
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
            system_status["system_health"] = f"ERROR: {str(e)}"
            await asyncio.sleep(60) # Wait longer on error

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info" # This log_level is for Uvicorn's internal logging, not overridden by basicConfig
    )

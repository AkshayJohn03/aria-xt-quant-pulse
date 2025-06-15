
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import random
from fastapi.middleware.cors import CORSMiddleware

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
    title="Aria XT Quant Pulse API",
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
            logger.warning("Some instances failed to initialize, continuing with available services")
        
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

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "status": "healthy",
        "message": "Aria XT Quant Pulse API is running",
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

# Include API router with prefix
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    print("Starting Aria XT Quant Pulse Backend...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

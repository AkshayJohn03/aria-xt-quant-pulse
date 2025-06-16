import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
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

# Configure CORS for development: allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        
        # Warm up Ollama model
        try:
            model_name = instances.config_manager.get('ollama.model', 'dolphin3:latest')
            if hasattr(instances.model_interface, 'warmup_ollama_model'):
                await instances.model_interface.warmup_ollama_model(model_name)
                logger.info(f"Ollama model '{model_name}' warmed up on startup.")
        except Exception as e:
            logger.warning(f"Ollama warmup failed: {e}")
        
        logger.info("Startup completed successfully")
        logger.info("Backend is now running on http://localhost:8000")
        logger.info("API documentation: http://localhost:8000/docs")
        
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

# Root endpoint with HTML response for easy testing
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint for testing - returns HTML page"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Aria XT Quant Pulse API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .status {{ color: green; font-weight: bold; }}
            .endpoint {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>🚀 Aria XT Quant Pulse API</h1>
        <p class="status">✅ Backend Server is Running!</p>
        <p><strong>Version:</strong> 1.0.0</p>
        <p><strong>Time:</strong> {datetime.now().isoformat()}</p>
        <h2>Available Endpoints:</h2>
        <div class="endpoint"><strong>GET /health</strong> - Health check</div>
        <div class="endpoint"><strong>GET /api/v1/market-data</strong> - Real-time market data</div>
        <div class="endpoint"><strong>GET /api/v1/connection-status</strong> - Connection status</div>
        <div class="endpoint"><strong>GET /api/v1/option-chain</strong> - Options chain data</div>
        <div class="endpoint"><strong>GET /docs</strong> - API documentation</div>
        <h2>Quick Test:</h2>
        <p><a href="/health" target="_blank">Test Health Endpoint</a></p>
        <p><a href="/api/v1/market-data" target="_blank">Test Market Data</a></p>
        <p><a href="/docs" target="_blank">View API Documentation</a></p>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "message": "Aria XT Quant Pulse API is running",
        "system_status": system_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# Include API router with prefix
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Aria XT Quant Pulse Backend...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🏥 Health check at: http://localhost:8000/health")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

# Import shared instances
from core import instances
from core.config_manager import ConfigManager
from core.instances import (
    config_manager,
    data_fetcher,
    model_interface,
    risk_manager,
    signal_generator,
    trade_executor,
    telegram_notifier,
    init_instances
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection - using shared instances
def get_config_manager_dep():
    return instances.config_manager

def get_model_interface_dep():
    return instances.model_interface

def get_risk_manager_dep():
    return instances.risk_manager

def get_trade_executor_dep():
    return instances.trade_executor

def create_api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Helper function to create consistent API responses"""
    response_data = {
        "success": success,
        "data": data,
        "error": error
    }
    return JSONResponse(content=response_data, status_code=status_code)

# --- Configuration endpoints ---
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/config")
async def get_config():
    """Get current configuration"""
    try:
        return config_manager.config
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config")
async def update_config(updates: Dict[str, Any], config: ConfigManager = Depends(get_config_manager_dep)):
    """Update configuration"""
    try:
        success = config.update_config(updates)
        if success:
            return create_api_response(True, {"message": "Configuration updated successfully"})
        else:
            return create_api_response(False, error="Failed to update configuration", status_code=500)
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/config/validate")
async def validate_config(config: ConfigManager = Depends(get_config_manager_dep)):
    """Validate current configuration"""
    try:
        is_valid = config.validate_config()
        if is_valid:
            return create_api_response(True, {"message": "Configuration is valid"})
        else:
            return create_api_response(False, error="Configuration validation failed", status_code=400)
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.post("/reload-config")
async def reload_config():
    """Reload .env and config.json, re-initialize all shared instances."""
    try:
        init_instances()
        return create_api_response(True, {"message": "Configuration and environment reloaded successfully."})
    except Exception as e:
        return create_api_response(False, error=str(e), status_code=500)

# --- System status endpoints ---
@router.get("/status")
async def get_status() -> Dict[str, bool]:
    """Get connection status for all services"""
    try:
        return {
            "zerodha": instances.trade_executor.kite is not None,
            "telegram": instances.telegram_notifier.bot is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connection-status")
async def get_connection_status():
    """Get connection status for all services"""
    try:
        status = {
            "zerodha": False,
            "telegram": False,
            "market_data": False,
            "portfolio": False,
            "options": False,
            "risk_metrics": False,
            "models": False,
            "ollama": False,
            "gemini": False,
            "timestamp": datetime.now().isoformat()
        }
        # Zerodha
        try:
            if instances.data_fetcher:
                status["zerodha"] = await instances.data_fetcher.test_zerodha_connection()
        except Exception as e:
            logger.warning(f"Zerodha connection check failed: {e}")
        # Telegram
        try:
            if instances.telegram_notifier:
                status["telegram"] = await instances.telegram_notifier.test_connection()
        except Exception as e:
            logger.warning(f"Telegram connection check failed: {e}")
        # Market Data
        try:
            if instances.data_fetcher:
                data = await instances.data_fetcher.fetch_market_data("NIFTY50")
                status["market_data"] = bool(data)
        except Exception as e:
            logger.warning(f"Market data connection check failed: {e}")
        # Portfolio
        try:
            if instances.data_fetcher:
                portfolio = await instances.data_fetcher.fetch_portfolio()
                status["portfolio"] = bool(portfolio)
        except Exception as e:
            logger.warning(f"Portfolio connection check failed: {e}")
        # Options
        try:
            if instances.data_fetcher:
                chain = await instances.data_fetcher.fetch_option_chain("NIFTY")
                status["options"] = bool(chain)
        except Exception as e:
            logger.warning(f"Options connection check failed: {e}")
        # Risk Metrics
        try:
            if instances.risk_manager:
                metrics = instances.risk_manager.calculate_risk_metrics()
                status["risk_metrics"] = bool(metrics)
        except Exception as e:
            logger.warning(f"Risk metrics connection check failed: {e}")
        # Models
        try:
            if instances.model_interface:
                status["models"] = True
                # Ollama
                status["ollama"] = await instances.model_interface.test_ollama_connection()
                # Gemini
                status["gemini"] = await instances.model_interface.test_gemini_connection()
        except Exception as e:
            logger.warning(f"Model interface connection check failed: {e}")
        return create_api_response(True, status)
    except Exception as e:
        logger.error(f"Error getting connection status: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/market-data")
async def get_market_data(symbol: str = "NIFTY50"):
    """Get current market data"""
    try:
        if not instances.data_fetcher:
            logger.error("Data fetcher not initialized")
            return create_api_response(False, error="Data fetcher not initialized", status_code=503)

        # Always fetch both NIFTY50 and BANKNIFTY for dashboard
        symbols = ["NIFTY50", "BANKNIFTY"]
        result = {}
        for sym in symbols:
            try:
                data = await instances.data_fetcher.fetch_market_data(sym)
                if not data:
                    raise Exception(f"No data available for {sym}")
                result[sym.lower()] = {
                    "value": data.get('current_price', 0.0) or 0.0,
                    "change": data.get('change', 0.0) or 0.0,
                    "percentChange": data.get('change_percent', 0.0) or 0.0,
                    "high": data['data'][0]['high'] if data.get('data') else 0.0,
                    "low": data['data'][0]['low'] if data.get('data') else 0.0,
                    "volume": data['data'][0]['volume'] if data.get('data') else 0,
                    "timestamp": data.get('timestamp'),
                    "source": data.get('source', 'unknown')
                }
            except Exception as e:
                logger.error(f"Error fetching market data for {sym}: {e}")
                result[sym.lower()] = {
                    "value": 0.0,
                    "change": 0.0,
                    "percentChange": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "volume": 0,
                    "timestamp": None,
                    "source": "error"
                }
        return create_api_response(True, result)
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/market-status")
async def get_market_status() -> Dict[str, bool]:
    """Get current market status"""
    try:
        return {
            "is_open": instances.data_fetcher.is_market_open()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio")
async def get_portfolio():
    """Get current portfolio status"""
    try:
        if not instances.data_fetcher:
            return create_api_response(False, error="Data fetcher not initialized", status_code=503)
        portfolio = await instances.data_fetcher.fetch_portfolio()
        return create_api_response(True, portfolio)
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return create_api_response(False, error=str(e), status_code=500)

# --- Trading endpoints ---
@router.get("/positions")
async def get_positions(risk_manager = Depends(get_risk_manager_dep)):
    """Get current positions"""
    try:
        positions = risk_manager.get_positions()
        return create_api_response(True, positions)
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/holdings")
async def get_holdings(risk_manager = Depends(get_risk_manager_dep)):
    """Get current holdings"""
    try:
        holdings = risk_manager.get_holdings()
        return create_api_response(True, holdings)
    except Exception as e:
        logger.error(f"Error fetching holdings: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/funds")
async def get_funds(risk_manager = Depends(get_risk_manager_dep)):
    """Get available funds"""
    try:
        funds = risk_manager.get_funds()
        return create_api_response(True, funds)
    except Exception as e:
        logger.error(f"Error fetching funds: {e}")
        return create_api_response(False, error=str(e), status_code=500)

# --- Model endpoints ---
@router.get("/models/status")
async def get_model_status(model_interface = Depends(get_model_interface_dep)):
    """Get status of all models"""
    try:
        status = {
            model: model_interface.is_model_loaded(model)
            for model in ["aria_lstm", "xgboost", "finbert", "prophet"]
        }
        return create_api_response(True, status)
    except Exception as e:
        logger.error(f"Error fetching model status: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/signals")
async def get_signals():
    """Get current trading signals"""
    try:
        if not instances.signal_generator:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        signals = await instances.signal_generator.generate_signals()
        return signals
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute-trade")
async def execute_trade(signal: Dict[str, Any]):
    """Execute a trade based on the provided signal"""
    try:
        if not instances.trade_executor:
            raise HTTPException(status_code=503, detail="Trade executor not initialized")

        result = await instances.trade_executor.execute_trade(signal)
        return result
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-metrics")
async def get_risk_metrics():
    """Get current risk metrics"""
    try:
        if not instances.risk_manager:
            return create_api_response(False, error="Risk manager not initialized", status_code=503)
        metrics = instances.risk_manager.calculate_risk_metrics()
        if not metrics:
            # Return default structure
            metrics = {
                'total_investment': 0.0,
                'portfolio_value': 0.0,
                'total_pnl': 0.0,
                'day_pnl': 0.0,
                'portfolio_exposure_percent': 0.0,
                'risk_score': 'LOW',
                'sector_exposure': {},
                'correlations': {},
                'portfolio_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
                'available_balance': 0.0
            }
        return create_api_response(True, metrics)
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.post("/notify")
async def send_notification(message: Dict[str, Any]):
    """Send a notification"""
    try:
        if not instances.telegram_notifier:
            raise HTTPException(status_code=503, detail="Telegram notifier not initialized")

        result = await instances.telegram_notifier.send_message(message)
        return result
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/option-chain")
async def get_option_chain(symbol: str = "NIFTY", expiry: Optional[str] = None):
    """Get real option chain data for a symbol and expiry"""
    try:
        if not instances.data_fetcher:
            return create_api_response(False, error="Data fetcher not initialized", status_code=503)
        chain = await instances.data_fetcher.fetch_option_chain(symbol, expiry)
        return create_api_response(True, chain)
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return create_api_response(False, error=str(e), status_code=500)

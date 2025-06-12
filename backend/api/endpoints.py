from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

# Import global instances from app.py to ensure singletons are used
from app import config_manager, data_fetcher, model_interface, risk_manager, signal_generator, trade_executor, telegram_notifier, system_status, run_backtest_logic_helper

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection - now using the global instances initialized in app.py
def get_config_manager_dep():
    return config_manager

def get_data_fetcher_dep():
    return data_fetcher

def get_model_interface_dep():
    return model_interface

def get_risk_manager_dep():
    return risk_manager

# Helper function to standardize API responses
def create_api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Create standardized API response"""
    response_data = {
        "success": success,
        "data": data,
        "error": error
    }
    return JSONResponse(content=response_data, status_code=status_code)

# --- Configuration endpoints ---
@router.get("/config")
async def get_config(config: ConfigManager = Depends(get_config_manager_dep)):
    """Get current configuration"""
    try:
        return create_api_response(True, config.config)
    except Exception as e:
        logger.error(f"Error fetching configuration: {e}")
        return create_api_response(False, error=str(e), status_code=500)

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
        return create_api_response(True, {"valid": is_valid})
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return create_api_response(False, error=str(e), status_code=500)

# --- Market data endpoints ---
@router.get("/market-data")
async def get_market_data():
    """Get current market data (NIFTY 50, SENSEX) from Zerodha."""
    try:
        data = await data_fetcher.fetch_market_data()
        
        # Ensure proper structure for frontend
        result = {
            "nifty": data.get("nifty", {"value": 0, "change": 0, "percentChange": 0}) if data else {"value": 0, "change": 0, "percentChange": 0},
            "sensex": data.get("sensex", {"value": 0, "change": 0, "percentChange": 0}) if data else {"value": 0, "change": 0, "percentChange": 0},
            "marketStatus": "OPEN",  # TODO: Get real market status
            "lastUpdate": datetime.now().isoformat(),
            "aiSentiment": {
                "direction": "NEUTRAL",
                "confidence": 50
            }
        }
        
        return create_api_response(True, result)
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        # Return default structure on error
        result = {
            "nifty": {"value": 0, "change": 0, "percentChange": 0},
            "sensex": {"value": 0, "change": 0, "percentChange": 0},
            "marketStatus": "CLOSED",
            "lastUpdate": datetime.now().isoformat(),
            "aiSentiment": {"direction": "NEUTRAL", "confidence": 0}
        }
        return create_api_response(False, result, str(e), status_code=500)

@router.get("/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str,
    timeframe: str = "1min",
    limit: int = 100,
    data_fetcher_inst: DataFetcher = Depends(get_data_fetcher_dep)
):
    """Get OHLCV data for a symbol"""
    try:
        ohlcv_data = await data_fetcher_inst.fetch_live_ohlcv(symbol, timeframe, limit)
        return create_api_response(True, {"symbol": symbol, "timeframe": timeframe, "data": ohlcv_data})
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/option-chain")
async def get_option_chain(
    symbol: str = "NIFTY",
    expiry: Optional[str] = None,
    data_fetcher_inst: DataFetcher = Depends(get_data_fetcher_dep)
):
    """Get option chain data"""
    try:
        option_chain = await data_fetcher_inst.fetch_option_chain(symbol, expiry)
        return create_api_response(True, {"symbol": symbol, "expiry": expiry, "data": option_chain})
    except Exception as e:
        logger.error(f"Error fetching option chain for {symbol}: {e}")
        return create_api_response(False, error=str(e), status_code=500)

# --- Model and prediction endpoints ---
@router.post("/prediction")
async def generate_prediction(
    symbol: str,
    timeframe: str = "1min",
    model_interface_inst: ModelInterface = Depends(get_model_interface_dep),
    data_fetcher_inst: DataFetcher = Depends(get_data_fetcher_dep)
):
    """Generate price prediction"""
    try:
        ohlcv_data = await data_fetcher_inst.fetch_live_ohlcv(symbol, timeframe, 100)
        prediction = await model_interface_inst.generate_prediction(ohlcv_data)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

@router.post("/trading-signal")
async def generate_trading_signal(
    symbol: str = "NIFTY",
    model_interface_inst: ModelInterface = Depends(get_model_interface_dep),
    data_fetcher_inst: DataFetcher = Depends(get_data_fetcher_dep),
    risk_manager_inst: RiskManager = Depends(get_risk_manager_dep)
):
    """Generate trading signal"""
    try:
        ohlcv_data = await data_fetcher_inst.fetch_live_ohlcv(symbol)
        option_chain = await data_fetcher_inst.fetch_option_chain(symbol)
        
        signal = await model_interface_inst.generate_trading_signal(ohlcv_data, option_chain)
        
        if signal:
            is_valid = risk_manager_inst.validate_signal(signal)
            signal["validated"] = is_valid
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "symbol": symbol,
                "signal": signal,
                "timestamp": datetime.now().isoformat()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error generating trading signal for {symbol}: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

# --- Portfolio and risk management endpoints ---
@router.get("/portfolio")
async def get_portfolio():
    """Get live portfolio overview (positions, holdings, funds) from Zerodha."""
    try:
        data = await trade_executor.get_portfolio_overview()
        return create_api_response(True, data)
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/positions")
async def get_positions(
    status: Optional[str] = None,
    risk_manager_inst: RiskManager = Depends(get_risk_manager_dep)
):
    """Get positions with optional status filter"""
    try:
        all_positions = risk_manager_inst.get_positions()
        if status:
            positions = [p for p in all_positions if p.get("status", "").upper() == status.upper()]
        else:
            positions = all_positions
        
        return create_api_response(True, {"positions": positions, "count": len(positions)})
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/risk-metrics")
async def get_risk_metrics(risk_manager_inst: RiskManager = Depends(get_risk_manager_dep)):
    """Get current risk metrics"""
    try:
        risk_metrics = risk_manager_inst.calculate_risk_metrics()
        return create_api_response(True, risk_metrics)
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        return create_api_response(False, error=str(e), status_code=500)

# --- Backtesting endpoints ---
@router.post("/backtest")
async def run_backtest(
    start_date: str,
    end_date: str,
    symbol: str = "NIFTY",
    strategy: str = "aria-lstm",
    data_fetcher_inst: DataFetcher = Depends(get_data_fetcher_dep),
    model_interface_inst: ModelInterface = Depends(get_model_interface_dep)
):
    """Run backtesting for a specific period"""
    try:
        historical_data = await data_fetcher_inst.fetch_historical_ohlcv(
            symbol, start_date, end_date
        )
        
        # Use the imported run_backtest_logic_helper
        backtest_results = await run_backtest_logic_helper(
            historical_data, strategy, model_interface_inst
        )
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "strategy": strategy,
                "period": f"{start_date} to {end_date}",
                "results": backtest_results,
                "timestamp": datetime.now().isoformat()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

@router.post("/backtest/live")
async def run_live_backtest(
    symbol: str = "NIFTY",
    hours: int = 1,
    strategy: str = "aria-lstm",
    data_fetcher_inst: DataFetcher = Depends(get_data_fetcher_dep),
    model_interface_inst: ModelInterface = Depends(get_model_interface_dep)
):
    """Run backtest with live data from the last N hours"""
    try:
        
        live_data = await data_fetcher_inst.fetch_live_ohlcv(
            symbol, "1min", hours * 60
        )
        
        backtest_results = await run_backtest_logic_helper(
            live_data, strategy, model_interface_inst
        )
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "strategy": strategy,
                "period": f"Last {hours} hours",
                "data_points": len(live_data),
                "results": backtest_results,
                "timestamp": datetime.now().isoformat()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error running live backtest for {symbol}: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

# --- Connection status endpoint ---
@router.get("/connection-status")
async def get_connection_status():
    """Get status of all external connections"""
    try:
        connections = {}
        
        # Test each connection safely
        try:
            connections["zerodha"] = await data_fetcher.test_zerodha_connection()
        except Exception as e:
            logger.error(f"Zerodha connection test failed: {e}")
            connections["zerodha"] = False
            
        try:
            connections["twelve_data"] = await data_fetcher.test_twelve_data_connection()
        except Exception as e:
            logger.error(f"Twelve Data connection test failed: {e}")
            connections["twelve_data"] = False
            
        try:
            connections["gemini"] = await model_interface.test_gemini_connection()
        except Exception as e:
            logger.error(f"Gemini connection test failed: {e}")
            connections["gemini"] = False
            
        try:
            connections["ollama"] = await model_interface.test_ollama_connection()
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            connections["ollama"] = False
            
        try:
            connections["telegram"] = await telegram_notifier.test_connection()
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            connections["telegram"] = False

        broker_connected = trade_executor.connected if trade_executor else False
        
        result = {
            "connections": connections,
            "broker_connected": broker_connected,
            "system_status": {
                "is_running": system_status.get("is_running", False),
                "active_trades": len(risk_manager.get_open_positions()) if risk_manager else 0,
                "total_pnl": risk_manager.calculate_total_pnl() if risk_manager else 0.0,
                "system_health": system_status.get("system_health", "OK")
            },
            "last_update": datetime.now().isoformat()
        }
        
        return create_api_response(True, result)
    except Exception as e:
        logger.error(f"Error getting connection status: {e}")
        return create_api_response(False, error=str(e), status_code=500)

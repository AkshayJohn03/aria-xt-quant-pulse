from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from core.config_manager import ConfigManager
from core.data_fetcher import DataFetcher
from core.model_interface import ModelInterface
from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor

# Import global instances from app.py to ensure singletons are used
# IMPORTANT: These imports must be after config_manager, data_fetcher, etc. are defined in app.py
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

# --- Configuration endpoints ---
@router.get("/config")
async def get_config(config: ConfigManager = Depends(get_config_manager_dep)):
    """Get current configuration"""
    try:
        return JSONResponse(content={"success": True, "data": config.config, "error": None})
    except Exception as e:
        logger.error(f"Error fetching configuration: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

@router.post("/config")
async def update_config(
    updates: Dict[str, Any],
    config: ConfigManager = Depends(get_config_manager_dep)
):
    """Update configuration"""
    try:
        success = config.update_config(updates)
        if success:
            return JSONResponse(content={"success": True, "data": {"message": "Configuration updated successfully"}, "error": None})
        else:
            return JSONResponse(content={"success": False, "data": None, "error": "Failed to update configuration"}, status_code=500)
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)


@router.get("/config/validate")
async def validate_config(config: ConfigManager = Depends(get_config_manager_dep)):
    """Validate current configuration"""
    try:
        is_valid = config.validate_config()
        return JSONResponse(content={"success": True, "data": {"valid": is_valid}, "error": None})
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

# --- Market data endpoints ---
@router.get("/market-data")
async def get_market_data():
    """Get current market data (NIFTY 50, SENSEX) from Zerodha."""
    try:
        from app import data_fetcher
        data = await data_fetcher.fetch_market_data()
        # Ensure both keys are present, even if one or both are missing
        result = {
            "nifty": data.get("nifty") if data and "nifty" in data else {"value": None, "change": None, "percentChange": None},
            "sensex": data.get("sensex") if data and "sensex" in data else {"value": None, "change": None, "percentChange": None},
            "lastUpdate": datetime.now().isoformat()
        }
        return JSONResponse(content={"success": True, "data": result, "error": None})
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        # Always return both keys with None values on error
        result = {
            "nifty": {"value": None, "change": None, "percentChange": None},
            "sensex": {"value": None, "change": None, "percentChange": None},
            "lastUpdate": datetime.now().isoformat()
        }
        return JSONResponse(content={"success": False, "data": result, "error": str(e)}, status_code=500)

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
        return JSONResponse(content={"success": True, "data": {"symbol": symbol, "timeframe": timeframe, "data": ohlcv_data}, "error": None})
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

@router.get("/option-chain")
async def get_option_chain(
    symbol: str = "NIFTY",
    expiry: Optional[str] = None,
    data_fetcher_inst: DataFetcher = Depends(get_data_fetcher_dep)
):
    """Get option chain data"""
    try:
        option_chain = await data_fetcher_inst.fetch_option_chain(symbol, expiry)
        return JSONResponse(content={"success": True, "data": {"symbol": symbol, "expiry": expiry, "data": option_chain}, "error": None})
    except Exception as e:
        logger.error(f"Error fetching option chain for {symbol}: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

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
        from app import trade_executor
        data = await trade_executor.get_portfolio_overview()
        return JSONResponse(content={"success": True, "data": data, "error": None})
    except Exception as e:
        logging.error(f"Error fetching portfolio: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

@router.get("/positions")
async def get_positions(
    status: Optional[str] = None,
    risk_manager_inst: RiskManager = Depends(get_risk_manager_dep)
):
    """Get positions with optional status filter"""
    try:
        all_positions = risk_manager_inst.get_positions() # Use the new get_positions
        if status:
            positions = [p for p in all_positions if p.get("status", "").upper() == status.upper()]
        else:
            positions = all_positions
        
        return JSONResponse(content={"success": True, "data": {"positions": positions, "count": len(positions)}, "error": None})
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

@router.get("/risk-metrics")
async def get_risk_metrics(risk_manager_inst: RiskManager = Depends(get_risk_manager_dep)):
    """Get current risk metrics"""
    try:
        risk_metrics = risk_manager_inst.calculate_risk_metrics()
        return JSONResponse(content={"success": True, "data": risk_metrics, "error": None})
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        return JSONResponse(content={"success": False, "data": None, "error": str(e)}, status_code=500)

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
async def get_connection_status(
    data_fetcher_inst: DataFetcher = Depends(get_data_fetcher_dep),
    model_interface_inst: ModelInterface = Depends(get_model_interface_dep)
):
    """Get status of all external connections (robust to partial failures)"""
    status = {}
    # Zerodha
    try:
        status["zerodha"] = await data_fetcher_inst.test_zerodha_connection()
    except Exception as e:
        logger.error(f"Zerodha connection check failed: {e}")
        status["zerodha"] = False
    # Twelve Data
    try:
        status["twelve_data"] = await data_fetcher_inst.test_twelve_data_connection()
    except Exception as e:
        logger.error(f"Twelve Data connection check failed: {e}")
        status["twelve_data"] = False
    # Gemini
    try:
        status["gemini"] = await model_interface_inst.test_gemini_connection()
    except Exception as e:
        logger.error(f"Gemini connection check failed: {e}")
        status["gemini"] = False
    # Ollama
    try:
        status["ollama"] = await model_interface_inst.test_ollama_connection()
    except Exception as e:
        logger.error(f"Ollama connection check failed: {e}")
        status["ollama"] = False
    status["last_update"] = datetime.now().isoformat()
    # Ensure all values are boolean for all()
    overall_health = all(s for key, s in status.items() if key != "last_update")
    return JSONResponse(content={"success": True, "data": {"connections": status, "overall_health": overall_health}, "error": None})
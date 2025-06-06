
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from core.config_manager import ConfigManager
from core.data_fetcher import DataFetcher
from core.model_interface import ModelInterface
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection
def get_config_manager():
    return ConfigManager()

def get_data_fetcher(config: ConfigManager = Depends(get_config_manager)):
    return DataFetcher(config)

def get_model_interface(config: ConfigManager = Depends(get_config_manager)):
    return ModelInterface(config)

def get_risk_manager(config: ConfigManager = Depends(get_config_manager)):
    return RiskManager(config)

# Configuration endpoints
@router.get("/config")
async def get_config(config: ConfigManager = Depends(get_config_manager)):
    """Get current configuration"""
    return config.config

@router.post("/config")
async def update_config(
    updates: Dict[str, Any],
    config: ConfigManager = Depends(get_config_manager)
):
    """Update configuration"""
    success = config.update_config(updates)
    if success:
        return {"message": "Configuration updated successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update configuration")

@router.get("/config/validate")
async def validate_config(config: ConfigManager = Depends(get_config_manager)):
    """Validate current configuration"""
    is_valid = config.validate_config()
    return {"valid": is_valid}

# Market data endpoints
@router.get("/market-data")
async def get_market_data(data_fetcher: DataFetcher = Depends(get_data_fetcher)):
    """Get current market data"""
    try:
        market_data = await data_fetcher.fetch_market_data()
        return market_data
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str,
    timeframe: str = "1min",
    limit: int = 100,
    data_fetcher: DataFetcher = Depends(get_data_fetcher)
):
    """Get OHLCV data for a symbol"""
    try:
        ohlcv_data = await data_fetcher.fetch_live_ohlcv(symbol, timeframe, limit)
        return {"symbol": symbol, "timeframe": timeframe, "data": ohlcv_data}
    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/option-chain")
async def get_option_chain(
    symbol: str = "NIFTY",
    expiry: Optional[str] = None,
    data_fetcher: DataFetcher = Depends(get_data_fetcher)
):
    """Get option chain data"""
    try:
        option_chain = await data_fetcher.fetch_option_chain(symbol, expiry)
        return {"symbol": symbol, "expiry": expiry, "data": option_chain}
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model and prediction endpoints
@router.post("/prediction")
async def generate_prediction(
    symbol: str,
    timeframe: str = "1min",
    model_interface: ModelInterface = Depends(get_model_interface),
    data_fetcher: DataFetcher = Depends(get_data_fetcher)
):
    """Generate price prediction"""
    try:
        # Fetch recent data
        ohlcv_data = await data_fetcher.fetch_live_ohlcv(symbol, timeframe, 100)
        
        # Generate prediction
        prediction = await model_interface.generate_prediction(ohlcv_data)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trading-signal")
async def generate_trading_signal(
    symbol: str = "NIFTY",
    model_interface: ModelInterface = Depends(get_model_interface),
    data_fetcher: DataFetcher = Depends(get_data_fetcher),
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    """Generate trading signal"""
    try:
        # Fetch market data
        ohlcv_data = await data_fetcher.fetch_live_ohlcv(symbol)
        option_chain = await data_fetcher.fetch_option_chain(symbol)
        
        # Generate signal
        signal = await model_interface.generate_trading_signal(ohlcv_data, option_chain)
        
        if signal:
            # Validate signal
            is_valid = risk_manager.validate_signal(signal)
            signal["validated"] = is_valid
        
        return {
            "symbol": symbol,
            "signal": signal,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating trading signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Portfolio and risk management endpoints
@router.get("/portfolio")
async def get_portfolio(risk_manager: RiskManager = Depends(get_risk_manager)):
    """Get current portfolio status"""
    try:
        portfolio = {
            "positions": risk_manager.get_positions(),
            "risk_metrics": risk_manager.calculate_risk_metrics(),
            "total_pnl": risk_manager.calculate_total_pnl(),
            "open_positions": len(risk_manager.get_open_positions())
        }
        return portfolio
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions")
async def get_positions(
    status: Optional[str] = None,
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    """Get positions with optional status filter"""
    try:
        if status:
            positions = [p for p in risk_manager.get_positions() if p["status"] == status.upper()]
        else:
            positions = risk_manager.get_positions()
        
        return {"positions": positions, "count": len(positions)}
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-metrics")
async def get_risk_metrics(risk_manager: RiskManager = Depends(get_risk_manager)):
    """Get current risk metrics"""
    try:
        risk_metrics = risk_manager.calculate_risk_metrics()
        return risk_metrics
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtesting endpoints
@router.post("/backtest")
async def run_backtest(
    start_date: str,
    end_date: str,
    symbol: str = "NIFTY",
    strategy: str = "aria-lstm",
    data_fetcher: DataFetcher = Depends(get_data_fetcher),
    model_interface: ModelInterface = Depends(get_model_interface)
):
    """Run backtesting for a specific period"""
    try:
        # Fetch historical data
        historical_data = await data_fetcher.fetch_historical_ohlcv(
            symbol, start_date, end_date
        )
        
        # Run backtest logic (simplified)
        backtest_results = await run_backtest_logic(
            historical_data, strategy, model_interface
        )
        
        return {
            "strategy": strategy,
            "period": f"{start_date} to {end_date}",
            "results": backtest_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest/live")
async def run_live_backtest(
    symbol: str = "NIFTY",
    hours: int = 1,
    strategy: str = "aria-lstm",
    data_fetcher: DataFetcher = Depends(get_data_fetcher),
    model_interface: ModelInterface = Depends(get_model_interface)
):
    """Run backtest with live data from the last N hours"""
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Fetch recent live data
        live_data = await data_fetcher.fetch_live_ohlcv(
            symbol, "1min", hours * 60
        )
        
        # Run backtest logic
        backtest_results = await run_backtest_logic(
            live_data, strategy, model_interface
        )
        
        return {
            "strategy": strategy,
            "period": f"Last {hours} hours",
            "data_points": len(live_data),
            "results": backtest_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running live backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Connection status endpoint
@router.get("/connection-status")
async def get_connection_status(
    data_fetcher: DataFetcher = Depends(get_data_fetcher),
    model_interface: ModelInterface = Depends(get_model_interface)
):
    """Get status of all external connections"""
    try:
        status = {
            "zerodha": await data_fetcher.test_zerodha_connection(),
            "twelve_data": await data_fetcher.test_twelve_data_connection(),
            "gemini": await model_interface.test_gemini_connection(),
            "ollama": await model_interface.test_ollama_connection(),
            "last_update": datetime.now().isoformat()
        }
        
        return {"connections": status, "overall_health": all(status.values())}
    except Exception as e:
        logger.error(f"Error checking connection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function for backtesting logic
async def run_backtest_logic(data: List[Dict], strategy: str, model_interface: ModelInterface):
    """Simplified backtesting logic"""
    portfolio_value = 100000
    trades = []
    
    for i in range(10, len(data), 10):  # Every 10 data points
        data_slice = data[i-10:i]
        
        try:
            prediction = await model_interface.generate_prediction(data_slice)
            
            if prediction and prediction.get("confidence", 0) > 70:
                # Simulate trade
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

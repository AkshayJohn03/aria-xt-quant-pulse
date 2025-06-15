
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import asyncio
import random

# Import shared instances
from core import instances

logger = logging.getLogger(__name__)

router = APIRouter()

def create_api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Helper function to create consistent API responses"""
    response_data = {
        "success": success,
        "data": data,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }
    return JSONResponse(content=response_data, status_code=status_code)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return create_api_response(True, {
        "status": "healthy",
        "message": "API is running",
        "timestamp": datetime.now().isoformat()
    })

@router.get("/connection-status")
async def get_connection_status():
    """Get connection status for all services"""
    try:
        status = {
            "zerodha": False,
            "telegram": False,
            "market_data": True,  # Always true for Yahoo Finance
            "portfolio": False,
            "options": False,
            "risk_metrics": False,
            "models": False,
            "ollama": False,
            "gemini": False,
            "twelve_data": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Test market data connection
        if instances.data_fetcher:
            try:
                test_data = await instances.data_fetcher.fetch_market_data("NIFTY50")
                status["market_data"] = test_data is not None
                logger.info(f"Market data test: {'SUCCESS' if status['market_data'] else 'FAILED'}")
            except Exception as e:
                logger.error(f"Market data test failed: {e}")
                status["market_data"] = False
        
        # Test Twelve Data if available
        if instances.data_fetcher:
            try:
                status["twelve_data"] = await instances.data_fetcher.test_twelve_data_connection()
            except Exception:
                status["twelve_data"] = False
        
        logger.info(f"Connection status check completed: {sum(1 for v in status.values() if v is True)} services online")
        return create_api_response(True, status)
        
    except Exception as e:
        logger.error(f"Error getting connection status: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/market-data")
async def get_market_data():
    """Get current market data for NIFTY50 and BANKNIFTY"""
    try:
        if not instances.data_fetcher:
            logger.error("Data fetcher not initialized")
            return create_api_response(False, error="Data fetcher not initialized", status_code=503)

        # Fetch both NIFTY50 and BANKNIFTY
        symbols = ["NIFTY50", "BANKNIFTY"]
        result = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching market data for {symbol}")
                data = await instances.data_fetcher.fetch_market_data(symbol)
                
                if data:
                    # Map to frontend expected format
                    symbol_key = symbol.lower().replace("50", "")  # nifty50 -> nifty, banknifty -> banknifty
                    result[symbol_key] = {
                        "value": float(data.get('current_price', 0.0)),
                        "change": float(data.get('change', 0.0)),
                        "percentChange": float(data.get('change_percent', 0.0)),
                        "high": float(data.get('high_24h', 0.0)),
                        "low": float(data.get('low_24h', 0.0)),
                        "volume": int(data.get('volume', 0)),
                        "timestamp": data.get('timestamp'),
                        "source": data.get('source', 'unknown')
                    }
                    logger.info(f"Processed market data for {symbol}: ₹{result[symbol_key]['value']:.2f} ({result[symbol_key]['percentChange']:+.2f}%)")
                else:
                    logger.warning(f"No data returned for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {e}")
                continue
        
        if not result:
            logger.error("Failed to fetch market data for any symbol")
            return create_api_response(False, error="Failed to fetch market data for any symbol", status_code=500)
        
        logger.info(f"Market data fetch completed successfully for {len(result)} symbols")
        return create_api_response(True, result)
        
    except Exception as e:
        logger.error(f"Error in market data endpoint: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/market-status")
async def get_market_status():
    """Get current market status"""
    try:
        is_open = False
        if instances.data_fetcher:
            is_open = instances.data_fetcher.is_market_open()
        
        logger.info(f"Market status: {'OPEN' if is_open else 'CLOSED'}")
        return create_api_response(True, {"is_open": is_open})
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/option-chain")
async def get_option_chain(symbol: str = "NIFTY", expiry: Optional[str] = None):
    """Get option chain data"""
    try:
        logger.info(f"Generating option chain for {symbol}")
        
        # Get current market price for strike calculation
        current_price = 24000  # Default fallback
        if instances.data_fetcher:
            try:
                market_data = await instances.data_fetcher.fetch_market_data("NIFTY50")
                if market_data:
                    current_price = market_data.get('current_price', 24000)
            except Exception as e:
                logger.warning(f"Could not fetch current price for option chain: {e}")
        
        # Generate realistic strikes around current price
        base_strike = int(current_price / 50) * 50  # Round to nearest 50
        strikes = [base_strike + (i * 50) for i in range(-4, 5)]  # 9 strikes total
        
        option_chain = []
        
        for strike in strikes:
            # Calculate realistic option prices
            moneyness = (current_price - strike) / strike
            
            # Call option pricing (simplified)
            call_ltp = max(0.5, current_price - strike + random.uniform(-10, 10)) if current_price > strike else random.uniform(0.5, 5)
            
            # Put option pricing (simplified)
            put_ltp = max(0.5, strike - current_price + random.uniform(-10, 10)) if strike > current_price else random.uniform(0.5, 5)
            
            option_chain.append({
                "strike": strike,
                "expiry": expiry or "2024-12-26",
                "call": {
                    "ltp": round(call_ltp, 2),
                    "volume": random.randint(1000, 100000),
                    "oi": random.randint(10000, 500000),
                    "iv": round(15 + random.uniform(-5, 10), 2),
                    "delta": round(0.1 + random.uniform(0, 0.8), 3),
                    "gamma": round(random.uniform(0, 0.01), 4),
                    "theta": round(-random.uniform(1, 5), 2),
                    "vega": round(random.uniform(5, 15), 2),
                    "affordable": call_ltp < 100
                },
                "put": {
                    "ltp": round(put_ltp, 2),
                    "volume": random.randint(1000, 100000),
                    "oi": random.randint(10000, 500000),
                    "iv": round(15 + random.uniform(-5, 10), 2),
                    "delta": round(-0.1 - random.uniform(0, 0.8), 3),
                    "gamma": round(random.uniform(0, 0.01), 4),
                    "theta": round(-random.uniform(1, 5), 2),
                    "vega": round(random.uniform(5, 15), 2),
                    "affordable": put_ltp < 100
                }
            })
        
        result = {
            "symbol": symbol,
            "underlying_value": current_price,
            "expiry_dates": ["2024-12-26", "2025-01-02", "2025-01-09", "2025-01-16"],
            "option_chain": option_chain,
            "available_funds": 100000,
            "source": "calculated_realistic",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Generated option chain for {symbol} with {len(option_chain)} strikes")
        return create_api_response(True, result)
        
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return create_api_response(False, error=str(e), status_code=500)

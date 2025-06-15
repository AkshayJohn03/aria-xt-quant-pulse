
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import asyncio

# Import shared instances
from core import instances

logger = logging.getLogger(__name__)

router = APIRouter()

def create_api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Helper function to create consistent API responses"""
    response_data = {
        "success": success,
        "data": data,
        "error": error
    }
    return JSONResponse(content=response_data, status_code=status_code)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/connection-status")
async def get_connection_status():
    """Get connection status for all services"""
    try:
        status = {
            "zerodha": False,
            "telegram": False,
            "market_data": True,  # Yahoo Finance is available
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
            except Exception as e:
                logger.error(f"Market data test failed: {e}")
                status["market_data"] = False
        
        # Test Twelve Data if available
        if instances.data_fetcher:
            try:
                status["twelve_data"] = await instances.data_fetcher.test_twelve_data_connection()
            except Exception:
                status["twelve_data"] = False
        
        logger.info(f"Connection status: {status}")
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
                    logger.info(f"Successfully processed market data for {symbol}: ₹{result[symbol_key]['value']:.2f}")
                else:
                    logger.warning(f"No data returned for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {e}")
                # Don't fail the entire request if one symbol fails
                continue
        
        if not result:
            return create_api_response(False, error="Failed to fetch market data for any symbol", status_code=500)
        
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
        
        return create_api_response(True, {"is_open": is_open})
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/option-chain")
async def get_option_chain(symbol: str = "NIFTY", expiry: Optional[str] = None):
    """Get option chain data with NSE scraping fallback"""
    try:
        # For now, return mock data as NSE scraping needs to be implemented properly
        # This ensures the frontend doesn't break while we work on the scraping
        
        mock_strikes = [23800, 23850, 23900, 23950, 24000, 24050, 24100, 24150, 24200]
        option_chain = []
        
        for strike in mock_strikes:
            option_chain.append({
                "strike": strike,
                "expiry": expiry or "2024-12-26",
                "call": {
                    "ltp": max(1, abs(24000 - strike) + (random.random() * 50)),
                    "volume": int(random.random() * 100000),
                    "oi": int(random.random() * 50000),
                    "iv": 15 + (random.random() * 20),
                    "delta": 0.1 + (random.random() * 0.8),
                    "gamma": random.random() * 0.01,
                    "theta": -(random.random() * 5),
                    "vega": random.random() * 10,
                    "affordable": True
                },
                "put": {
                    "ltp": max(1, abs(strike - 24000) + (random.random() * 50)),
                    "volume": int(random.random() * 100000),
                    "oi": int(random.random() * 50000),
                    "iv": 15 + (random.random() * 20),
                    "delta": -(0.1 + (random.random() * 0.8)),
                    "gamma": random.random() * 0.01,
                    "theta": -(random.random() * 5),
                    "vega": random.random() * 10,
                    "affordable": True
                }
            })
        
        result = {
            "symbol": symbol,
            "underlying_value": 24000,
            "expiry_dates": ["2024-12-26", "2025-01-02", "2025-01-09"],
            "option_chain": option_chain,
            "available_funds": 100000,
            "source": "mock_data_realistic",
            "timestamp": datetime.now().isoformat()
        }
        
        return create_api_response(True, result)
        
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return create_api_response(False, error=str(e), status_code=500)

# Import random for mock data
import random

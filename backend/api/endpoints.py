from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import asyncio
import random
import httpx
import re
import pandas as pd
import yfinance as yf

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
            "market_data": False,
            "portfolio": False,
            "options": False,
            "risk_metrics": False,
            "models": False,
            "ollama": False,
            "gemini": False,
            "twelve_data": False,
            "timestamp": datetime.now().isoformat()
        }

        # Market data
        if instances.data_fetcher:
            try:
                test_data = await instances.data_fetcher.fetch_market_data("NIFTY50")
                status["market_data"] = test_data is not None
                logger.info(f"Market data test: {'SUCCESS' if status['market_data'] else 'FAILED'}")
            except Exception as e:
                logger.error(f"Market data test failed: {e}")
                status["market_data"] = False

        # Twelve Data
        if instances.data_fetcher:
            try:
                status["twelve_data"] = await instances.data_fetcher.test_twelve_data_connection()
            except Exception:
                status["twelve_data"] = False

        # Zerodha
        if hasattr(instances, "trade_executor") and instances.trade_executor:
            try:
                status["zerodha"] = await instances.trade_executor.test_connection()
            except Exception as e:
                logger.error(f"Zerodha connection test failed: {e}")
                status["zerodha"] = False

        # Portfolio
        if hasattr(instances, "trade_executor") and instances.trade_executor:
            try:
                portfolio = await instances.trade_executor.get_portfolio()
                status["portfolio"] = bool(portfolio and (portfolio.get("positions") or portfolio.get("holdings")))
            except Exception as e:
                logger.error(f"Portfolio test failed: {e}")
                status["portfolio"] = False

        # Telegram
        if hasattr(instances, "telegram_notifier") and instances.telegram_notifier:
            try:
                status["telegram"] = await instances.telegram_notifier.test_connection()
                logger.info(f"Telegram connection status: {status['telegram']}")
            except Exception as e:
                logger.error(f"Telegram connection test failed: {e}")
                status["telegram"] = False

        # Ollama
        if hasattr(instances, "model_interface") and instances.model_interface:
            try:
                status["ollama"] = await instances.model_interface.test_ollama_connection()
            except Exception as e:
                logger.error(f"Ollama connection test failed: {e}")
                status["ollama"] = False

        # Gemini
        if hasattr(instances, "model_interface") and instances.model_interface:
            try:
                status["gemini"] = await instances.model_interface.test_gemini_connection()
            except Exception as e:
                logger.error(f"Gemini connection test failed: {e}")
                status["gemini"] = False

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
    """Get option chain data from NSE India with session and headers, retry and fallback to Yahoo if blocked."""
    try:
        nse_symbol = symbol.upper()
        base_url = "https://www.nseindia.com"
        option_chain_url = f"{base_url}/api/option-chain-indices?symbol={nse_symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": f"{base_url}/option-chain",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive"
        }
        async with httpx.AsyncClient(timeout=10) as client:
            await client.get(base_url, headers=headers)
            for attempt in range(2):
                resp = await client.get(option_chain_url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    all_expiries = data['records']['expiryDates']
                    expiry_to_use = expiry or (all_expiries[0] if all_expiries else None)
                    option_data = [item for item in data['records']['data'] if item.get('expiryDate') == expiry_to_use]
                    option_chain = []
                    for item in option_data:
                        strike = item.get('strikePrice')
                        ce = item.get('CE', {})
                        pe = item.get('PE', {})
                        option_chain.append({
                            "strike": strike,
                            "expiry": expiry_to_use,
                            "call": {
                                "ltp": ce.get('lastPrice'),
                                "volume": ce.get('totalTradedVolume'),
                                "oi": ce.get('openInterest'),
                                "chng": ce.get('change'),
                                "iv": ce.get('impliedVolatility'),
                            },
                            "put": {
                                "ltp": pe.get('lastPrice'),
                                "volume": pe.get('totalTradedVolume'),
                                "oi": pe.get('openInterest'),
                                "chng": pe.get('change'),
                                "iv": pe.get('impliedVolatility'),
                            }
                        })
                    result = {
                        "symbol": symbol,
                        "underlying_value": data['records']['underlyingValue'],
                        "expiry_dates": all_expiries,
                        "option_chain": option_chain,
                        "source": "nse_india",
                        "timestamp": datetime.now().isoformat()
                    }
                    return create_api_response(True, result)
                elif resp.status_code == 401 and attempt == 0:
                    await asyncio.sleep(2)  # Wait and retry
                else:
                    break
        # Fallback: Yahoo Finance (mocked structure, no real option chain)
        ticker = yf.Ticker("^NSEI")
        hist = ticker.history(period="5d", interval="1d")
        strikes = [int(hist['Close'].iloc[-1] // 50) * 50 + i * 50 for i in range(-4, 5)]
        option_chain = [
            {
                "strike": strike,
                "expiry": expiry or "N/A",
                "call": {"ltp": None, "volume": None, "oi": None, "chng": None, "iv": None},
                "put": {"ltp": None, "volume": None, "oi": None, "chng": None, "iv": None}
            }
            for strike in strikes
        ]
        return create_api_response(False, {
            "symbol": symbol,
            "underlying_value": hist['Close'].iloc[-1] if not hist.empty else None,
            "expiry_dates": [expiry] if expiry else [],
            "option_chain": option_chain,
            "source": "yahoo_fallback",
            "timestamp": datetime.now().isoformat(),
            "reason": "NSE blocked automated requests. Showing fallback structure."
        }, error="NSE option chain fetch failed", status_code=500)
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/portfolio")
async def get_portfolio():
    """Get current portfolio (positions, holdings, funds)"""
    try:
        if hasattr(instances, "trade_executor") and instances.trade_executor:
            try:
                portfolio = await instances.trade_executor.get_portfolio()
                if portfolio:
                    return create_api_response(True, portfolio)
            except Exception as e:
                logger.error(f"Error fetching portfolio: {e}")
        # Fallback: always return a valid structure
        fallback = {
            "positions": [],
            "holdings": [],
            "funds": {"available": 0.0, "used": 0.0, "total": 0.0},
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }
        return create_api_response(True, fallback)
    except Exception as e:
        logger.error(f"Error in portfolio endpoint: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/risk-metrics")
async def get_risk_metrics():
    """Get current risk metrics for the portfolio"""
    try:
        if hasattr(instances, "risk_manager") and instances.risk_manager:
            try:
                metrics = instances.risk_manager.calculate_risk_metrics()
                if metrics:
                    return create_api_response(True, metrics)
            except Exception as e:
                logger.error(f"Error fetching risk metrics: {e}")
        # Fallback: always return a valid structure
        fallback = {
            "total_pnl": 0.0,
            "portfolio_volatility": 0.0,
            "risk_score": "N/A",
            "portfolio_exposure_percent": 0.0,
            "sector_exposure": {},
            "correlations": {},
            "timestamp": datetime.now().isoformat(),
            "source": "fallback",
            "reason": "No real portfolio data available. Zerodha not connected or portfolio is empty."
        }
        return create_api_response(True, fallback)
    except Exception as e:
        logger.error(f"Error in risk metrics endpoint: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/ohlc")
async def get_ohlc(symbol: str = "NIFTY50", interval: str = "1d", limit: int = 100):
    """Get OHLC data for charting (NSE or Yahoo fallback)"""
    try:
        # Try NSE first
        nse_symbol = {"NIFTY50": "NIFTY", "BANKNIFTY": "BANKNIFTY"}.get(symbol, "NIFTY")
        base_url = "https://www.nseindia.com"
        chart_url = f"{base_url}/api/chart-databyindex?index={nse_symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": f"{base_url}/chart",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive"
        }
        async with httpx.AsyncClient(timeout=10) as client:
            await client.get(base_url, headers=headers)
            resp = await client.get(chart_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                candles = data.get('grapthData', [])[-limit:]
                ohlc = [
                    {
                        "timestamp": c[0],
                        "open": c[1],
                        "high": c[2],
                        "low": c[3],
                        "close": c[4],
                        "volume": c[5] if len(c) > 5 else 0
                    }
                    for c in candles
                ]
                return create_api_response(True, {"symbol": symbol, "ohlc": ohlc, "source": "nse_india"})
        # Yahoo fallback
        ticker_map = {"NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK"}
        ticker = yf.Ticker(ticker_map.get(symbol, "^NSEI"))
        hist = ticker.history(period="max", interval=interval)
        ohlc = [
            {
                "timestamp": str(idx),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            }
            for idx, row in hist.tail(limit).iterrows()
        ]
        return create_api_response(True, {"symbol": symbol, "ohlc": ohlc, "source": "yahoo_finance"})
    except Exception as e:
        logger.error(f"Error fetching OHLC data: {e}")
        return create_api_response(False, error=str(e), status_code=500)

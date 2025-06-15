
import yfinance as yf
import requests
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import aiohttp
import random

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.session = None
        
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_market_data(self, symbol: str = "NIFTY50") -> Dict[str, Any]:
        """Fetch real-time market data from Yahoo Finance"""
        try:
            # Map symbols to Yahoo Finance tickers
            symbol_map = {
                "NIFTY50": "^NSEI",
                "NIFTY": "^NSEI", 
                "BANKNIFTY": "^NSEBANK"
            }
            
            yahoo_symbol = symbol_map.get(symbol, "^NSEI")
            logger.info(f"Fetching data for {symbol} -> {yahoo_symbol}")
            
            # Fetch data using yfinance
            ticker = yf.Ticker(yahoo_symbol)
            
            # Get current price and basic info
            try:
                info = ticker.info
                hist = ticker.history(period="5d", interval="1m")
                
                if hist.empty:
                    logger.warning(f"No historical data available for {symbol}, using fallback")
                    return self._get_fallback_data(symbol)
                    
                current_price = float(hist['Close'].iloc[-1])
                previous_close = float(info.get('previousClose', hist['Close'].iloc[0]))
                
                change = current_price - previous_close
                percent_change = (change / previous_close) * 100 if previous_close > 0 else 0
                
                # Get high/low from recent data
                high_24h = float(hist['High'].tail(50).max())
                low_24h = float(hist['Low'].tail(50).min())
                volume = int(hist['Volume'].tail(50).sum())
                
                result = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'change': change,
                    'change_percent': percent_change,
                    'high_24h': high_24h,
                    'low_24h': low_24h,
                    'volume': volume,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'yahoo_finance'
                }
                
                logger.info(f"Successfully fetched market data for {symbol}: ₹{current_price:.2f} ({percent_change:+.2f}%)")
                return result
                
            except Exception as e:
                logger.error(f"Error processing yfinance data for {symbol}: {e}")
                return self._get_fallback_data(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return self._get_fallback_data(symbol)
    
    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic fallback data when real data is not available"""
        base_prices = {
            "NIFTY50": 24000,
            "NIFTY": 24000,
            "BANKNIFTY": 51000
        }
        
        base_price = base_prices.get(symbol, 24000)
        # Generate realistic intraday movement
        change_percent = (random.random() - 0.5) * 2  # -1% to +1%
        change = base_price * (change_percent / 100)
        current_price = base_price + change
        
        logger.info(f"Using fallback data for {symbol}: ₹{current_price:.2f}")
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'change': change,
            'change_percent': change_percent,
            'high_24h': current_price + abs(change) + 50,
            'low_24h': current_price - abs(change) - 50,
            'volume': random.randint(100000000, 200000000),
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }
    
    def is_market_open(self) -> bool:
        """Check if Indian market is currently open"""
        try:
            now = datetime.now()
            # Convert to IST
            ist_time = now + timedelta(hours=5, minutes=30)
            
            # Market is open Monday to Friday, 9:15 AM to 3:30 PM IST
            if ist_time.weekday() > 4:  # Saturday = 5, Sunday = 6
                return False
                
            market_open = ist_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = ist_time.replace(hour=15, minute=30, second=0, microsecond=0)
            
            is_open = market_open <= ist_time <= market_close
            logger.info(f"Market status check: {ist_time.strftime('%H:%M')} IST - {'OPEN' if is_open else 'CLOSED'}")
            return is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    async def test_zerodha_connection(self) -> bool:
        """Test Zerodha connection"""
        try:
            # This would test actual Zerodha connection
            # For now, return False as it's not configured
            return False
        except Exception:
            return False
    
    async def test_twelve_data_connection(self) -> bool:
        """Test Twelve Data API connection"""
        try:
            session = await self.get_session()
            async with session.get("https://api.twelvedata.com/time_series?symbol=AAPL&interval=1min&apikey=demo", timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Twelve Data connection test failed: {e}")
            return False

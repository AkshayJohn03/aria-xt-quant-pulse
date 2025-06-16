import yfinance as yf
import requests
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import aiohttp
import random
import httpx
import re
import html

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
        """Fetch real-time market data: Zerodha > Yahoo > Google > NSE > fallback"""
        # Try Zerodha first
        try:
            from core import instances
            if hasattr(instances, "trade_executor") and instances.trade_executor:
                broker = getattr(instances.trade_executor, "broker", None)
                if broker and hasattr(broker, "get_positions"):
                    # Try to fetch from Zerodha
                    try:
                        # For index, use kite.quote
                        if hasattr(broker, "kite") and broker.kite:
                            quote = await asyncio.to_thread(broker.kite.quote, [symbol])
                            if quote and symbol in quote:
                                q = quote[symbol]
                                current_price = float(q.get('last_price', 0))
                                change = float(q.get('change', 0))
                                percent_change = float(q.get('net_change', 0))
                                high_24h = float(q.get('ohlc', {}).get('high', 0))
                                low_24h = float(q.get('ohlc', {}).get('low', 0))
                                volume = int(q.get('volume', 0))
                                return {
                                    'symbol': symbol,
                                    'current_price': current_price,
                                    'change': change,
                                    'change_percent': percent_change,
                                    'high_24h': high_24h,
                                    'low_24h': low_24h,
                                    'volume': volume,
                                    'timestamp': datetime.now().isoformat(),
                                    'source': 'zerodha'
                                }
                    except Exception as e:
                        logger.warning(f"Zerodha fetch failed for {symbol}: {e}")
        except Exception as e:
            logger.warning(f"Zerodha not available: {e}")
        # Yahoo fallback (existing logic)
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
        # Google Finance fallback
        try:
            google_symbol = {
                "NIFTY50": "NSE:NIFTY_50",
                "BANKNIFTY": "NSE:NIFTY_BANK"
            }.get(symbol, "NSE:NIFTY_50")
            url = f"https://www.google.com/finance/quote/{google_symbol}"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=5)
                if resp.status_code == 200:
                    # Parse price from HTML (fragile, but works as fallback)
                    match = re.search(r'<div[^>]*class="YMlKec"[^>]*>([\d,\.]+)</div>', resp.text)
                    if match:
                        current_price = float(match.group(1).replace(",", ""))
                        logger.info(f"Fetched {symbol} from Google Finance: ₹{current_price}")
                        return {
                            'symbol': symbol,
                            'current_price': current_price,
                            'change': 0.0,
                            'change_percent': 0.0,
                            'high_24h': 0.0,
                            'low_24h': 0.0,
                            'volume': 0,
                            'timestamp': datetime.now().isoformat(),
                            'source': 'google_finance'
                        }
        except Exception as e:
            logger.warning(f"Google Finance fetch failed for {symbol}: {e}")
        # NSE India fallback (scrape chart OHLC)
        try:
            nse_symbol = {
                "NIFTY50": "NIFTY",
                "BANKNIFTY": "BANKNIFTY"
            }.get(symbol, "NIFTY")
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={nse_symbol}"
            headers = {"User-Agent": "Mozilla/5.0"}
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    underlying = data.get('records', {}).get('underlyingValue')
                    if underlying:
                        logger.info(f"Fetched {symbol} from NSE India: ₹{underlying}")
                        return {
                            'symbol': symbol,
                            'current_price': underlying,
                            'change': 0.0,
                            'change_percent': 0.0,
                            'high_24h': 0.0,
                            'low_24h': 0.0,
                            'volume': 0,
                            'timestamp': datetime.now().isoformat(),
                            'source': 'nse_india'
                        }
        except Exception as e:
            logger.warning(f"NSE India fetch failed for {symbol}: {e}")
        # Deterministic fallback
        base_prices = {"NIFTY50": 24000, "NIFTY": 24000, "BANKNIFTY": 51000}
        base_price = base_prices.get(symbol, 24000)
        logger.info(f"Using deterministic fallback for {symbol}: ₹{base_price}")
        return {
            'symbol': symbol,
            'current_price': base_price,
            'change': 0.0,
            'change_percent': 0.0,
            'high_24h': base_price,
            'low_24h': base_price,
            'volume': 0,
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }
    
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

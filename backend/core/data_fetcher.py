import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.config_manager import ConfigManager
import httpx
from kiteconnect import KiteConnect
import yfinance as yf
import pytz
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.config
        self.config_manager = config_manager
        
        self.zerodha_api_key = self.config_manager.get("apis.zerodha.api_key")
        self.zerodha_api_secret = self.config_manager.get("apis.zerodha.api_secret")
        self.zerodha_access_token = self.config_manager.get("apis.zerodha.access_token")
        self.zerodha_base_url = self.config_manager.get("apis.zerodha.base_url")

        self.twelve_data_api_key = self.config_manager.get("apis.twelve_data.api_key")
        self.twelve_data_base_url = self.config_manager.get("apis.twelve_data.base_url")

        self.kite = None
        self.cache = {}
        self.cache_timeout = 60
        self.market_hours = {
            'start': '09:15:00',
            'end': '15:30:00'
        }
        self.last_update = None
        self.cache_ttl = 60
        
        # Initialize KiteConnect if credentials are available
        if self.zerodha_api_key and self.zerodha_access_token:
            try:
                self.kite = KiteConnect(api_key=self.zerodha_api_key)
                self.kite.set_access_token(self.zerodha_access_token)
                logger.info("KiteConnect client initialized with access token.")
            except Exception as e:
                logger.error(f"Failed to initialize KiteConnect with access token: {e}")
        else:
            logger.warning("Zerodha API key or access token not found. Using Yahoo Finance as primary data source.")
            
        logger.info("DataFetcher initialized.")

    def set_kite(self, kite):
        """Set KiteConnect instance"""
        self.kite = kite

    def initialize(self):
        """Initialize the data fetcher"""
        try:
            if self.zerodha_api_key and self.zerodha_access_token:
                self.kite = KiteConnect(api_key=self.zerodha_api_key)
                self.kite.set_access_token(self.zerodha_access_token)
                logger.info("KiteConnect client initialized with access token.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error initializing DataFetcher: {e}")
            return False

    async def fetch_market_data(self, symbol: str = "NIFTY50") -> Dict[str, Any]:
        """Fetch market data with Yahoo Finance as primary source"""
        try:
            # Check cache first
            if self._is_cache_valid(symbol):
                return self.cache[symbol]

            # Use Yahoo Finance as primary source for free tier
            try:
                logger.info(f"Fetching market data from Yahoo Finance for {symbol}")
                data = await self._fetch_from_yahoo(symbol)
                if data:
                    self._update_cache(symbol, data)
                    return data
            except Exception as e:
                logger.warning(f"Yahoo Finance fetch failed: {e}")

            # Fallback to Zerodha if available and Yahoo fails
            if self.kite:
                try:
                    logger.info(f"Falling back to Zerodha for {symbol}")
                    data = await self._fetch_from_zerodha(symbol)
                    if data:
                        self._update_cache(symbol, data)
                        return data
                except Exception as e:
                    logger.warning(f"Zerodha fetch also failed: {e}")

            # If both fail, return None - let the API handle fallback
            logger.warning(f"All data sources failed for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    async def _fetch_from_yahoo(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Yahoo Finance with improved error handling"""
        try:
            yahoo_symbol = self._map_to_yahoo_symbol(symbol)
            if not yahoo_symbol:
                return None

            logger.info(f"Fetching from Yahoo Finance: {symbol} -> {yahoo_symbol}")
            
            # Fetch data in a separate thread to avoid blocking
            def fetch_yahoo_data():
                ticker = yf.Ticker(yahoo_symbol)
                hist = ticker.history(period='5d', interval='5m')  # Get more data
                info = ticker.info
                return hist, info

            hist, info = await asyncio.to_thread(fetch_yahoo_data)

            if hist.empty:
                logger.warning(f"No historical data available for {yahoo_symbol}")
                return None

            # Get current and previous close prices
            current_price = float(hist['Close'].iloc[-1])
            
            # Try to get previous close from info, fallback to previous data point
            try:
                prev_close = float(info.get('previousClose', hist['Close'].iloc[-2]))
            except (IndexError, KeyError):
                prev_close = current_price  # Fallback if no previous data
            
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0

            # Convert to required format
            formatted_data = []
            for index, row in hist.tail(50).iterrows():  # Last 50 data points
                formatted_data.append({
                    'timestamp': index.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })

            result = {
                'symbol': symbol,
                'data': formatted_data,
                'current_price': current_price,
                'change': change,
                'change_percent': change_percent,
                'source': 'yahoo',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully fetched Yahoo Finance data for {symbol}: {current_price} (change: {change:.2f})")
            return result

        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance for {symbol}: {e}")
            return None

    async def _fetch_from_zerodha(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Zerodha"""
        try:
            if not self.kite:
                return None

            # Get historical data
            to_date = datetime.now()
            from_date = to_date - timedelta(days=1)
            
            instrument_token = self._get_instrument_token(symbol)
            if not instrument_token:
                logger.error(f"No instrument token found for {symbol}")
                return None

            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval='5minute'
            )

            if not data:
                return None

            # Get current quote
            quote = self.kite.quote(instrument_token)
            if quote:
                current_price = quote[instrument_token]['last_price']
                change = quote[instrument_token]['change']
                change_percent = quote[instrument_token]['change_percent']
            else:
                current_price = data[0]['close']
                change = current_price - data[1]['close']
                change_percent = (change / data[1]['close']) * 100

            return {
                'symbol': symbol,
                'data': data,
                'current_price': current_price,
                'change': change,
                'change_percent': change_percent,
                'source': 'zerodha',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching from Zerodha: {e}")
            return None

    def _get_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock data for testing"""
        now = datetime.now()
        data = []
        
        # Generate realistic mock data
        base_price = 19000 if symbol == "NIFTY50" else 45000 if symbol == "BANKNIFTY" else 65000
        current_price = base_price + np.random.normal(0, 100)
        prev_price = current_price - np.random.normal(0, 50)
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100

        # Generate 50 data points
        for i in range(50):
            timestamp = now - timedelta(minutes=5 * (50 - i))
            price_variation = np.random.normal(0, 20)
            
            data.append({
                'timestamp': timestamp.isoformat(),
                'open': float(current_price + price_variation),
                'high': float(current_price + price_variation + abs(np.random.normal(0, 10))),
                'low': float(current_price + price_variation - abs(np.random.normal(0, 10))),
                'close': float(current_price + price_variation + np.random.normal(0, 5)),
                'volume': int(np.random.normal(1000000, 200000))
            })

        return {
            'symbol': symbol,
            'data': data,
            'current_price': current_price,
            'change': change,
            'change_percent': change_percent,
            'source': 'mock',
            'timestamp': now.isoformat()
        }

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self.cache:
            return False
        if not self.last_update:
            return False
        return (datetime.now() - self.last_update).seconds < self.cache_ttl

    def _update_cache(self, symbol: str, data: Dict[str, Any]):
        """Update the cache with new data"""
        self.cache[symbol] = data
        self.last_update = datetime.now()

    def _get_instrument_token(self, symbol: str) -> int:
        """Get instrument token for Zerodha"""
        # This should be implemented based on your instrument mapping
        mapping = {
            "NIFTY50": 256265,
            "BANKNIFTY": 260105
        }
        return mapping.get(symbol)

    def _map_to_yahoo_symbol(self, symbol: str) -> str:
        """Map internal symbol to Yahoo Finance symbol"""
        mapping = {
            "NIFTY50": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "NIFTY": "^NSEI",
            "SENSEX": "^BSESN"
        }
        return mapping.get(symbol, symbol)

    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        try:
            now = datetime.now(pytz.timezone('Asia/Kolkata'))
            if now.weekday() >= 5:  # Weekend
                return False

            market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)

            return market_start <= now <= market_end
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    async def test_zerodha_connection(self) -> bool:
        """Test Zerodha connection"""
        try:
            if not self.kite:
                logger.info("Zerodha connection test: No KiteConnect client")
                return False
            
            profile = await asyncio.to_thread(self.kite.profile)
            logger.info("Zerodha connection test: SUCCESS")
            return bool(profile)
        except Exception as e:
            logger.error(f"Zerodha connection test failed: {e}")
            return False

    async def test_twelve_data_connection(self) -> bool:
        """Tests connection to Twelve Data API."""
        if not self.twelve_data_api_key:
            logger.info("Twelve Data connection test: No API key configured")
            return False
        try:
            url = f"{self.twelve_data_base_url}/quote?symbol=AAPL&apikey={self.twelve_data_api_key}"
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data and data.get("status") == "ok":
                    logger.info("Twelve Data connection test: SUCCESS")
                    return True
                else:
                    logger.error(f"Twelve Data connection test failed with status: {data.get('status', 'N/A')}")
                    return False
        except Exception as e:
            logger.error(f"Twelve Data connection test failed: {e}")
            return False

    async def get_live_price(self, symbol: str) -> float:
        """Get live price for a symbol"""
        try:
            if not self.is_market_open():
                raise ValueError("Market is closed")

            if self.kite:
                try:
                    # Try Zerodha first
                    quote = self.kite.quote(symbol)
                    return quote[symbol]['last_price']
                except Exception as e:
                    logger.error(f"Error getting Zerodha live price: {e}")

            # Fallback to Yahoo Finance
            ticker = yf.Ticker(symbol)
            return ticker.info['regularMarketPrice']

        except Exception as e:
            logger.error(f"Error getting live price: {e}")
            raise

    async def get_option_chain(self, symbol: str) -> Dict[str, Any]:
        """Get option chain for a symbol"""
        try:
            if not self.is_market_open():
                raise ValueError("Market is closed")

            if not self.kite:
                raise ValueError("Zerodha connection not available")

            # Get current price
            current_price = await self.get_live_price(symbol)

            # Get expiry dates
            expiries = self.kite.margins()['equity']['available']['cash']

            # Get option chain
            option_chain = {}
            for expiry in expiries:
                # Get calls
                calls = self.kite.instruments('OPT', symbol, expiry, 'CE')
                # Get puts
                puts = self.kite.instruments('OPT', symbol, expiry, 'PE')

                option_chain[expiry] = {
                    'calls': [{
                        'strike': option['strike'],
                        'expiry': option['expiry'],
                        'type': 'CE',
                        'last_price': option['last_price'],
                        'volume': option['volume'],
                        'oi': option['oi'],
                        'iv': option['iv']
                    } for option in calls],
                    'puts': [{
                        'strike': option['strike'],
                        'expiry': option['expiry'],
                        'type': 'PE',
                        'last_price': option['last_price'],
                        'volume': option['volume'],
                        'oi': option['oi'],
                        'iv': option['iv']
                    } for option in puts]
                }

            return {
                'symbol': symbol,
                'current_price': current_price,
                'option_chain': option_chain
            }

        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            raise

    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            if not self.kite:
                raise ValueError("Zerodha connection not available")

            # Get market status
            status = self.kite.margins()

            return {
                'is_open': self.is_market_open(),
                'nifty': {
                    'last_price': status['equity']['available']['cash'],
                    'change': status['equity']['used']['cash'],
                    'change_percent': status['equity']['used']['cash'] / status['equity']['available']['cash'] * 100
                },
                'banknifty': {
                    'last_price': status['equity']['available']['cash'],
                    'change': status['equity']['used']['cash'],
                    'change_percent': status['equity']['used']['cash'] / status['equity']['available']['cash'] * 100
                }
            }

        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            raise

    async def fetch_live_ohlcv(self, symbol: str = "NIFTY50", timeframe: str = "1min", limit: int = 100) -> List[Dict]:
        """
        Fetches live OHLCV data. This will now attempt to use Zerodha Kite.
        """
        if self.kite:
            try:
                instrument_token = self.config_manager.get(f"data.{symbol.lower().replace(' ', '_')}_instrument_token", None) # Adjusted key for flexibility
                
                if not instrument_token:
                    logger.warning(f"Instrument token for {symbol} not configured. Cannot fetch real OHLCV data from Zerodha. Returning mock data.")
                    return self._generate_mock_ohlcv(symbol, timeframe, limit)
                
                kite_interval_map = {
                    "1min": "minute", "5min": "5minute", "15min": "15minute",
                    "30min": "30minute", "60min": "60minute", "1day": "day"
                }
                kite_interval = kite_interval_map.get(timeframe)

                if not kite_interval:
                    logger.error(f"Unsupported timeframe: {timeframe} for Zerodha. Falling back to mock OHLCV.")
                    return self._generate_mock_ohlcv(symbol, timeframe, limit)

                to_date = datetime.now()
                # Calculate from_date based on limit and timeframe
                if timeframe == "1min":
                    from_date = to_date - timedelta(minutes=limit * 2) # More buffer for 1min
                elif timeframe == "5min":
                    from_date = to_date - timedelta(minutes=limit * 5 * 2)
                elif timeframe == "15min":
                    from_date = to_date - timedelta(minutes=limit * 15 * 2)
                elif timeframe == "60min":
                    from_date = to_date - timedelta(hours=limit * 2)
                elif timeframe == "1day":
                    from_date = to_date - timedelta(days=limit * 2)
                else:
                    from_date = to_date - timedelta(days=limit * 2)

                logger.info(f"Fetching historical OHLCV for {symbol} ({instrument_token}) from {from_date.isoformat()} to {to_date.isoformat()} at {kite_interval} interval.")
                
                raw_ohlcv_data = await asyncio.to_thread(
                    self.kite.historical_data, 
                    instrument_token, 
                    from_date, 
                    to_date, 
                    kite_interval,
                    continuous=False
                )

                parsed_ohlcv = []
                for entry in raw_ohlcv_data:
                    parsed_ohlcv.append({
                        "timestamp": entry.get("date").isoformat(),
                        "open": entry.get("open"),
                        "high": entry.get("high"),
                        "low": entry.get("low"),
                        "close": entry.get("close"),
                        "volume": entry.get("volume")
                    })
                logger.info(f"Fetched {len(parsed_ohlcv)} OHLCV data points for {symbol} from Zerodha.")
                return parsed_ohlcv[-limit:]
            except Exception as e:
                logger.error(f"Error fetching live OHLCV for {symbol} from Zerodha: {e}. Returning mock data.")
                return self._generate_mock_ohlcv(symbol, timeframe, limit)
        else:
            logger.warning("KiteConnect client not initialized. Cannot fetch live OHLCV data from Zerodha. Returning mock data.")
            return self._generate_mock_ohlcv(symbol, timeframe, limit)

    def _generate_mock_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Helper to generate mock OHLCV data."""
        mock_data = []
        base_price = 20000 if "NIFTY" in symbol.upper() else 40000 # Different base for NIFTY/BANKNIFTY
        current_time = datetime.now()
        for i in range(limit + 50): # Generate a bit more than limit for safety
            offset_minutes = (limit + 50) - i
            timestamp = current_time - timedelta(minutes=offset_minutes) # Adjust timestamp based on minutes ago
            
            close = base_price + (i * 0.5) + (i % 5 - 2) * 10
            open_p = close - 5
            high_p = close + 10
            low_p = close - 8
            volume = 100000 + i * 100
            mock_data.append({
                "timestamp": timestamp.isoformat(),
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close,
                "volume": volume
            })
        logger.info(f"Generated {len(mock_data)} mock OHLCV data points for {symbol}.")
        return mock_data[-limit:]

    async def fetch_option_chain(self, symbol: str = "NIFTY", expiry: Optional[str] = None) -> List[Dict]:
        """
        Fetches option chain data. Uses Zerodha as primary, falls back to NSE scraping if needed.
        """
        try:
            # Try Zerodha first
            if self.kite:
                try:
                    # Get expiry dates
                    instruments = await asyncio.to_thread(self.kite.instruments, "NFO")
                    filtered = [i for i in instruments if i["name"].upper() == symbol.upper() and i["segment"] == "NFO-OPT"]
                    if expiry:
                        filtered = [i for i in filtered if i["expiry"] == expiry]
                    strikes = sorted(set(i["strike"] for i in filtered))
                    expiries = sorted(set(i["expiry"] for i in filtered))
                    if not expiry and expiries:
                        expiry = expiries[0]
                    chain = []
                    for strike in strikes:
                        call = next((i for i in filtered if i["strike"] == strike and i["instrument_type"] == "CE" and i["expiry"] == expiry), None)
                        put = next((i for i in filtered if i["strike"] == strike and i["instrument_type"] == "PE" and i["expiry"] == expiry), None)
                        if call or put:
                            chain.append({
                                "strike": strike,
                                "expiry": expiry,
                                "call": {
                                    "ltp": call["last_price"] if call else None,
                                    "volume": call["volume"] if call else None,
                                    "oi": call["open_interest"] if call else None,
                                    "iv": call["implied_volatility"] if call else None,
                                    "delta": call.get("greeks", {}).get("delta") if call and call.get("greeks") else None
                                } if call else {},
                                "put": {
                                    "ltp": put["last_price"] if put else None,
                                    "volume": put["volume"] if put else None,
                                    "oi": put["open_interest"] if put else None,
                                    "iv": put["implied_volatility"] if put else None,
                                    "delta": put.get("greeks", {}).get("delta") if put and put.get("greeks") else None
                                } if put else {}
                            })
                    if chain:
                        return chain
                except Exception as e:
                    logger.warning(f"Zerodha option chain fetch failed: {e}")
            # Fallback to NSE scraping
            try:
                import nsepython
                nse_chain = nsepython.nse_optionchain_scrapper(symbol)
                expiry = expiry or nse_chain['records']['expiryDates'][0]
                data = nse_chain['records']['data']
                chain = []
                for entry in data:
                    if entry['expiryDate'] != expiry:
                        continue
                    strike = entry['strikePrice']
                    call = entry.get('CE', {})
                    put = entry.get('PE', {})
                    chain.append({
                        "strike": strike,
                        "expiry": expiry,
                        "call": {
                            "ltp": call.get('lastPrice'),
                            "volume": call.get('totalTradedVolume'),
                            "oi": call.get('openInterest'),
                            "iv": call.get('impliedVolatility'),
                            "delta": call.get('delta')
                        } if call else {},
                        "put": {
                            "ltp": put.get('lastPrice'),
                            "volume": put.get('totalTradedVolume'),
                            "oi": put.get('openInterest'),
                            "iv": put.get('impliedVolatility'),
                            "delta": put.get('delta')
                        } if put else {}
                    })
                if chain:
                    return chain
            except Exception as e:
                logger.error(f"NSE option chain fallback failed: {e}")
            raise Exception("Failed to fetch option chain from both Zerodha and NSE.")
        except Exception as e:
            logger.error(f"Error in fetch_option_chain: {e}")
            raise

    async def fetch_historical_ohlcv(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1day") -> List[Dict]:
        """
        Fetches historical OHLCV data for backtesting.
        This will use Zerodha Kite historical_data if configured, otherwise Yahoo Finance.
        """
        if self.kite:
            try:
                instrument_token = self.config_manager.get(f"data.{symbol.lower().replace(' ', '_')}_instrument_token", None)
                if not instrument_token:
                    logger.warning(f"Instrument token for {symbol} not configured for historical data. Falling back to Yahoo Finance.")
                    raise Exception("No instrument token")
                kite_interval_map = {
                    "1min": "minute", "5min": "5minute", "15min": "15minute",
                    "30min": "30minute", "60min": "60minute", "1day": "day"
                }
                kite_interval = kite_interval_map.get(timeframe)
                if not kite_interval:
                    logger.error(f"Unsupported timeframe: {timeframe} for Zerodha historical data. Falling back to Yahoo Finance.")
                    raise Exception("Unsupported timeframe")
                from_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                to_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                logger.info(f"Fetching historical OHLCV for {symbol} ({instrument_token}) from {from_date_dt.isoformat()} to {to_date_dt.isoformat()} at {kite_interval} interval.")
                raw_ohlcv_data = await asyncio.to_thread(
                    self.kite.historical_data,
                    instrument_token,
                    from_date_dt,
                    to_date_dt,
                    kite_interval,
                    continuous=False
                )
                parsed_ohlcv = []
                for entry in raw_ohlcv_data:
                    parsed_ohlcv.append({
                        "timestamp": entry.get("date").isoformat(),
                        "open": entry.get("open"),
                        "high": entry.get("high"),
                        "low": entry.get("low"),
                        "close": entry.get("close"),
                        "volume": entry.get("volume")
                    })
                logger.info(f"Fetched {len(parsed_ohlcv)} historical OHLCV data points for {symbol} from Zerodha.")
                return parsed_ohlcv
            except Exception as e:
                logger.error(f"Error fetching historical OHLCV for {symbol} from Zerodha: {e}. Trying Yahoo Finance fallback.")
        # Fallback to Yahoo Finance
        try:
            yahoo_symbol = self._map_to_yahoo_symbol(symbol)
            if not yahoo_symbol:
                raise Exception("No Yahoo symbol mapping")
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(start=start_date, end=end_date, interval="1d" if timeframe == "1day" else "5m")
            if data.empty:
                raise Exception("No data from Yahoo Finance")
            formatted_data = []
            for index, row in data.iterrows():
                formatted_data.append({
                    'timestamp': index.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            logger.info(f"Fetched {len(formatted_data)} historical OHLCV data points for {symbol} from Yahoo Finance.")
            return formatted_data
        except Exception as e:
            logger.error(f"Error fetching historical OHLCV for {symbol} from Yahoo Finance: {e}")
            raise Exception(f"Failed to fetch historical OHLCV for {symbol} from both Zerodha and Yahoo Finance.")

    async def fetch_portfolio(self):
        """Fetch portfolio from Zerodha Kite, fallback to MCP if needed."""
        # Try Zerodha Kite first
        if self.kite:
            try:
                return await asyncio.to_thread(self.kite.holdings)
            except Exception as e:
                logger.error(f"Error fetching from Zerodha: {e}")
                if "permission" not in str(e).lower():
                    raise
        # Fallback to MCP
        try:
            import httpx
            mcp_token = os.getenv("ZERODHA_MCP_TOKEN")
            mcp_url = os.getenv("ZERODHA_MCP_URL", "https://mcp.zerodha.com/api/v1/portfolio")
            if not mcp_token:
                logger.error("MCP token not configured. Set ZERODHA_MCP_TOKEN in .env.")
                raise Exception("MCP token not configured.")
            headers = {"Authorization": f"Bearer {mcp_token}"}
            async with httpx.AsyncClient() as client:
                resp = await client.get(mcp_url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                else:
                    logger.error(f"MCP portfolio fetch failed: {resp.text}")
                    raise Exception(f"MCP portfolio fetch failed: {resp.text}")
        except Exception as e:
            logger.error(f"Portfolio fetch failed from both Zerodha and MCP: {e}")
            raise

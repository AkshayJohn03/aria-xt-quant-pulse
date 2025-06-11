import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.config_manager import ConfigManager
import httpx
from kiteconnect import KiteConnect

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
        if self.zerodha_api_key and self.zerodha_access_token:
            try:
                self.kite = KiteConnect(api_key=self.zerodha_api_key)
                self.kite.set_access_token(self.zerodha_access_token)
                logger.info("KiteConnect client initialized with access token.")
            except Exception as e:
                logger.error(f"Failed to initialize KiteConnect with access token: {e}. Zerodha functionality may be limited.")
        else:
            logger.warning("Zerodha API key or access token not found. Zerodha market data will not be fetched.")
            
        logger.info("DataFetcher initialized.")

    async def fetch_market_data(self) -> Dict[str, Any]:
        """
        Fetches real-time market data for NIFTY and SENSEX from Zerodha.
        Also includes a placeholder for AI Sentiment.
        """
        # Initialize with more realistic mock values as fallback
        nifty_data = {"value": 22000.00, "change": 50.00, "percentChange": 0.23}
        sensex_data = {"value": 73000.00, "change": -100.00, "percentChange": -0.14}
        market_status = "CLOSED"
        last_update = datetime.now().isoformat()

        # Dummy AI sentiment for now, will come from ModelInterface later
        ai_sentiment = {"direction": "NEUTRAL", "confidence": 50}

        if self.kite:
            try:
                nifty_instrument_token = self.config_manager.get("data.nifty_instrument_token", None)
                sensex_instrument_token = self.config_manager.get("data.sensex_instrument_token", None)
                
                if not nifty_instrument_token or not sensex_instrument_token:
                    logger.warning("NIFTY or SENSEX instrument tokens not configured in config.json. Cannot fetch live market data from Zerodha. Returning mock data.")
                    return {
                        "nifty": nifty_data,
                        "sensex": sensex_data,
                        "marketStatus": market_status,
                        "lastUpdate": last_update,
                        "aiSentiment": ai_sentiment
                    }
                
                instrument_list = [nifty_instrument_token, sensex_instrument_token]
                quotes = await asyncio.to_thread(self.kite.ltp, instrument_list)
                
                # --- DEBUG LOGGING ---
                logger.debug(f"Raw Zerodha LTP response for {instrument_list}: {quotes}")
                # --- END DEBUG LOGGING ---

                if nifty_instrument_token in quotes:
                    nifty_quote = quotes[nifty_instrument_token]
                    nifty_data["value"] = nifty_quote.get("last_price", nifty_data["value"])
                    nifty_data["change"] = nifty_quote.get("change", nifty_data["change"])
                    nifty_prev_close = nifty_quote.get("ohlc", {}).get("close")
                    if nifty_prev_close and nifty_prev_close != 0:
                        nifty_data["percentChange"] = (nifty_data["change"] / nifty_prev_close) * 100
                    else:
                        nifty_data["percentChange"] = nifty_quote.get("net_change", nifty_data["percentChange"])

                if sensex_instrument_token in quotes:
                    sensex_quote = quotes[sensex_instrument_token]
                    sensex_data["value"] = sensex_quote.get("last_price", sensex_data["value"])
                    sensex_data["change"] = sensex_quote.get("change", sensex_data["change"])
                    sensex_prev_close = sensex_quote.get("ohlc", {}).get("close")
                    if sensex_prev_close and sensex_prev_close != 0:
                        sensex_data["percentChange"] = (sensex_data["change"] / sensex_prev_close) * 100
                    else:
                        sensex_data["percentChange"] = sensex_quote.get("net_change", sensex_data["percentChange"])

                if nifty_data["value"] != 0 and sensex_data["value"] != 0:
                    market_status = "OPEN"
                last_update = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Error fetching real market data from Zerodha: {e}. Falling back to mock data.")
                pass 
        else:
            logger.warning("KiteConnect client not initialized. Cannot fetch live market data from Zerodha. Returning mock data.")

        return {
            "nifty": nifty_data,
            "sensex": sensex_data,
            "marketStatus": market_status,
            "lastUpdate": last_update,
            "aiSentiment": ai_sentiment
        }

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
        Fetches option chain data. This is still a placeholder.
        """
        logger.warning(f"fetch_option_chain for {symbol}: Not yet implemented for real data, returning mock data.")
        # TODO: Implement actual option chain fetching logic from Zerodha or other providers.
        return [
            {
                "strike": 22000,
                "expiry": "2025-06-27",
                "call": {"ltp": 150.0, "volume": 150000, "oi": 700000, "iv": 25.0, "delta": 0.55},
                "put": {"ltp": 80.0, "volume": 120000, "oi": 600000, "iv": 22.0, "delta": -0.45}
            },
            {
                "strike": 22100,
                "expiry": "2025-06-27",
                "call": {"ltp": 100.0, "volume": 100000, "oi": 500000, "iv": 23.0, "delta": 0.40},
                "put": {"ltp": 100.0, "volume": 110000, "oi": 550000, "iv": 24.0, "delta": -0.60}
            }
        ]

    async def fetch_historical_ohlcv(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1day") -> List[Dict]:
        """
        Fetches historical OHLCV data for backtesting.
        This will use Zerodha Kite historical_data if configured, otherwise mock.
        """
        if self.kite:
            try:
                # Use a more robust way to get instrument token for historical data
                # For example, by mapping from `config.json` based on symbol or fetching dynamically
                instrument_token = self.config_manager.get(f"data.{symbol.lower().replace(' ', '_')}_instrument_token", None)

                if not instrument_token:
                    logger.warning(f"Instrument token for {symbol} not configured for historical data. Returning mock data.")
                    num_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days + 1
                    return self._generate_mock_ohlcv(symbol, timeframe, num_days)
                
                kite_interval_map = {
                    "1min": "minute", "5min": "5minute", "15min": "15minute",
                    "30min": "30minute", "60min": "60minute", "1day": "day"
                }
                kite_interval = kite_interval_map.get(timeframe)

                if not kite_interval:
                    logger.error(f"Unsupported timeframe: {timeframe} for Zerodha historical data. Returning mock data.")
                    num_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days + 1
                    return self._generate_mock_ohlcv(symbol, timeframe, num_days)

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
                logger.error(f"Error fetching historical OHLCV for {symbol} from Zerodha: {e}. Returning mock data.")
                num_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days + 1
                return self._generate_mock_ohlcv(symbol, timeframe, num_days)
        else:
            logger.warning("KiteConnect client not initialized for historical data. Returning mock data.")
            num_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days + 1
            return self._generate_mock_ohlcv(symbol, timeframe, num_days)

    async def test_zerodha_connection(self) -> bool:
        """Tests connection to Zerodha API by fetching user profile."""
        if not self.kite:
            logger.warning("Zerodha Kite client not initialized for testing.")
            return False
        try:
            user_profile = await asyncio.to_thread(self.kite.profile)
            logger.info(f"Zerodha connection test: Successfully fetched user profile for {user_profile.get('user_name')}")
            return True
        except Exception as e:
            logger.error(f"Zerodha connection test failed: {e}")
            return False

    async def test_twelve_data_connection(self) -> bool:
        """Tests connection to Twelve Data API."""
        if not self.twelve_data_api_key:
            logger.warning("Twelve Data API key not configured for testing.")
            return False
        try:
            url = f"{self.twelve_data_base_url}/quote?symbol=AAPL&apikey={self.twelve_data_api_key}"
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                if data and data.get("status") == "ok":
                    logger.info("Twelve Data connection test: SUCCESS")
                    return True
                else:
                    logger.error(f"Twelve Data connection test failed with status: {data.get('status', 'N/A')}")
                    return False
        except httpx.RequestError as e:
            logger.error(f"Twelve Data connection test request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during Twelve Data connection test: {e}")
            return False

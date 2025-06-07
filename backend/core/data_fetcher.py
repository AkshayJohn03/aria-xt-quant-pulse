# D:\aria\aria-xt-quant-pulse\backend\core\data_fetcher.py

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataFetcher:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_keys = config.get("api_keys", {})
        self.data_providers = config.get("data_providers", {})
        logging.info("DataFetcher initialized.")

    def _make_api_call(self, url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None) -> Optional[Dict]:
        """Helper to make API calls with basic error handling."""
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error occurred: {e} - Response: {e.response.text}")
            return None
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection error occurred: {e}")
            return None
        except requests.exceptions.Timeout as e:
            logging.error(f"Timeout error occurred: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"An unexpected request error occurred: {e}")
            return None
        except ValueError as e: # For json.JSONDecodeError
            logging.error(f"Failed to decode JSON response: {e} - Response text: {response.text}")
            return None

    def fetch_nifty_banknifty_status(self) -> Optional[Dict]:
        """
        Fetches the current status (value, change, etc.) for Nifty and BankNifty.
        This is a placeholder. You'd integrate with actual API here (e.g., historical data provider for latest tick).
        For demonstration, we'll return mocked data.
        """
        logging.info("Fetching Nifty/BankNifty status...")

        # --- IMPORTANT: Replace this with actual API calls ---
        # Example using a mock data (replace with actual API integration)
        # You would typically get this from a real-time data feed or a reliable API
        # For a production system, you'd integrate with services like:
        # Zerodha KiteConnect, Fyers API, Upstox API, or external market data APIs (e.g., Alpha Vantage, Finnhub, Polygon.io)
        # Ensure your API keys are configured and handled securely.

        # Mocked data for demonstration
        current_time = datetime.now()
        mock_nifty_value = 22500.00 + (current_time.minute % 10 - 5) * 2.5
        mock_banknifty_value = 48000.00 + (current_time.minute % 10 - 5) * 5.0

        # Simulate a small random change
        import random
        nifty_change = round(random.uniform(-50, 50), 2)
        banknifty_change = round(random.uniform(-100, 100), 2)

        nifty_data = {
            "current_value": mock_nifty_value,
            "change": nifty_change,
            "change_percent": (nifty_change / (mock_nifty_value - nifty_change)) * 100 if (mock_nifty_value - nifty_change) != 0 else 0,
            "open": 22450.00,
            "high": 22550.00,
            "low": 22400.00,
            "close": 22500.00 - nifty_change # Simple inverse to ensure change makes sense
        }

        banknifty_data = {
            "current_value": mock_banknifty_value,
            "change": banknifty_change,
            "change_percent": (banknifty_change / (mock_banknifty_value - banknifty_change)) * 100 if (mock_banknifty_value - banknifty_change) != 0 else 0,
            "open": 47900.00,
            "high": 48100.00,
            "low": 47850.00,
            "close": 48000.00 - banknifty_change # Simple inverse to ensure change makes sense
        }

        return {
            "nifty": nifty_data,
            "banknifty": banknifty_data,
            # "finnifty": {...} # Add Finnifty data if needed
            "last_updated": current_time.isoformat()
        }

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetches historical data for a given symbol.
        (Placeholder for actual API integration, e.g., using Alpha Vantage, Yahoo Finance via pandas_datareader)
        """
        logging.info(f"Fetching historical data for {symbol} from {start_date} to {end_date} at {interval} interval.")
        # Example: Alpha Vantage API integration
        alpha_vantage_key = self.api_keys.get("ALPHA_VANTAGE")
        if not alpha_vantage_key:
            logging.warning("Alpha Vantage API key not found in config.")
            return None

        # This is a simplified example. Real-world integration involves more parameters.
        # E.g., function=TIME_SERIES_DAILY_ADJUSTED, outputsize=full, datatype=json
        # url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={alpha_vantage_key}"
        # response_data = self._make_api_call(url)

        # if response_data and "Time Series (Daily)" in response_data:
        #     df = pd.DataFrame.from_dict(response_data["Time Series (Daily)"], orient='index')
        #     df.index = pd.to_datetime(df.index)
        #     df = df.astype(float)
        #     return df
        # else:
        #     logging.warning(f"Could not fetch historical data for {symbol}.")
        #     return None

        # Mock Data for historical
        dates = pd.date_range(start=start_date, end=end_date, freq=interval)
        if dates.empty:
            return pd.DataFrame() # Return empty if date range is invalid

        data = {
            'Open': [random.uniform(100, 105) for _ in dates],
            'High': [random.uniform(105, 110) for _ in dates],
            'Low': [random.uniform(95, 100) for _ in dates],
            'Close': [random.uniform(98, 108) for _ in dates],
            'Volume': [random.randint(100000, 500000) for _ in dates]
        }
        df = pd.DataFrame(data, index=dates)
        return df


    # Add more fetching methods as needed, e.g.:
    # - fetch_options_chain(symbol, expiry_date)
    # - fetch_futures_data(symbol)
    # - fetch_fno_list()
    # - fetch_margin_details()
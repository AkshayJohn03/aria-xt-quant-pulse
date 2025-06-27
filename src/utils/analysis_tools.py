import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import ta # Import pandas_ta explicitly here for clarity in functions that use it

# Ensure root directory is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import necessary components from your inference setup
# Note: config and device are imported to ensure consistent parameters
from src.inference.inference_utils import load_inference_artifacts, fetch_and_preprocess_data, make_prediction, config, device
from src.models.lstm_model import LSTMModel # Needed here for type hinting and clarity of model structure

# Suppress warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Function 1: Verify Processed Data Labeling ---

def verify_aggressive_labeling(
    processed_data_dir: str = config['local_processed_data_dir'], # <--- UPDATED: Uses the correct path from config
    label_column: str = config['label_to_use'],
    sample_file_name: str = "NIFTY_50/NIFTY_50_ohlcv_indicators.joblib", # <--- UPDATED DEFAULT FILE NAME
    num_rows: int = 10
):
    """
    Loads a sample of preprocessed data and displays the 'Close' price,
    the aggressive label, and the implied next period's close change
    to help verify the labeling logic.

    Args:
        processed_data_dir (str): Directory containing the preprocessed .joblib files.
        label_column (str): The name of the label column (e.g., 'label_aggressive').
        sample_file_name (str): The specific .joblib file to load for verification.
                                 Example: 'your_market_data_symbol_1min_processed.joblib'
        num_rows (int): Number of rows to display from the loaded DataFrame.
    """
    print("\n--- Verifying Aggressive Labeling ---")
    
    file_path = os.path.join(processed_data_dir, sample_file_name) # Uses the correct config path now

    if not os.path.exists(file_path):
        print(f"Error: Sample file not found at {file_path}.")
        print(f"Please ensure '{sample_file_name}' exists in '{processed_data_dir}' and update 'sample_file_name' in the call.")
        return

    try:
        df = joblib.load(file_path)
        print(f"Loaded {file_path} for verification.")

        if 'Close' not in df.columns or label_column not in df.columns:
            print(f"Error: 'Close' or '{label_column}' column not found in the loaded data.")
            return

        df['Next_Close'] = df['Close'].shift(-1)
        df['Price_Change_Next_Percent'] = ((df['Next_Close'] - df['Close']) / df['Close']) * 100

        print(f"\nDisplaying {num_rows} rows from the tail to verify labeling:")
        print(df[['Close', 'Next_Close', 'Price_Change_Next_Percent', label_column]].tail(num_rows).to_string())
        print("\n--- Interpretation Guide ---")
        print(f"Compare 'Price_Change_Next_Percent' with '{label_column}'.")
        print(f"If '{label_column}' is 1, it should correspond to a significant positive 'Price_Change_Next_Percent' (your 'aggressive upward move' definition).")
        print(f"If '{label_column}' is 0, it should correspond to a non-significant or negative 'Price_Change_Next_Percent'.")
        print("Note: The exact threshold for 'significant' is defined in your data generation script.")

    except Exception as e:
        print(f"An error occurred during data verification: {e}")

# --- Function 2: Fetch Data with Fallback (Twelvedata -> Yahoo Finance) ---

def fetch_data_with_fallback(symbol: str, interval: str, twelvedata_api_key: str = config['twelvedata_api_key']):
    """
    Fetches market data using Twelvedata first, falls back to Yahoo Finance if needed.
    
    Args:
        symbol (str): The financial symbol to fetch data for
        interval (str): The data interval (e.g., '1min', '5min', '1h', '1d')
        twelvedata_api_key (str): Twelvedata API key
        
    Returns:
        pandas.DataFrame: Market data with OHLCV columns, or None if fetching fails.
    """
    # Try Twelvedata first
    try:
        import twelvedata
        td = twelvedata.TDClient(apikey=twelvedata_api_key)
        # Fetch enough data for indicators and lookback window
        # For minute data, 200 bars is usually enough. For daily, need much more.
        fetch_count = max(config['lookback_window'] + 50, 200) 
        if interval == '1d': # For daily data, fetch more historical data
            fetch_count = 365 # Fetch a year's worth of daily data for robust indicators

        # Map interval format for Twelvedata
        interval_map = {
            '1min': '1min', '5min': '5min', '15min': '15min', '1h': '1h', '1d': '1day'
        }
        td_interval = interval_map.get(interval, interval)
        
        ts = td.time_series(
            symbol=symbol,
            interval=td_interval,
            outputsize=fetch_count,
            timezone="exchange"
        )
        df = ts.as_pandas()
        
        if not df.empty:
            print(f"Successfully fetched data from Twelvedata for {symbol}")
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 
                               'close': 'Close', 'volume': 'Volume'}, inplace=True)
            return df
            
    except Exception as e:
        print(f"Twelvedata failed for {symbol}: {e}")
        print("Falling back to Yahoo Finance...")
    
    # Fallback to Yahoo Finance
    try:
        yf_interval_map = {
            '1min': '1m', '5min': '5m', '15min': '15m', '1h': '1h', '1d': '1d'
        }
        yf_interval = yf_interval_map.get(interval, interval)
        
        # Handle different symbol formats for Yahoo Finance
        yf_symbol = symbol
        if symbol.upper() == 'NIFTY' or symbol.upper() == 'NIFTY/INDEX':
            yf_symbol = '^NSEI' # NIFTY 50 index
        elif symbol.upper() == 'SENSEX':
            yf_symbol = '^BSESN' # SENSEX index
        elif symbol.upper() == 'BANKNIFTY':
            yf_symbol = '^NSEBANK' # BANK NIFTY index
        
        # Determine period for Yahoo Finance based on interval
        yf_period = "5d" # Default for intraday
        if interval == '1d':
            yf_period = "1y" # Fetch 1 year of daily data for robust indicator calculation
        elif interval == '1h':
            yf_period = "30d" # Fetch 30 days of hourly data

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=yf_period, interval=yf_interval) # Fetch enough history
        
        if not df.empty:
            print(f"Successfully fetched data from Yahoo Finance for {symbol} (using {yf_symbol})")
            df.rename(columns={
                'Open': 'Open', 'High': 'High', 'Low': 'Low',
                'Close': 'Close', 'Volume': 'Volume'
            }, inplace=True)
            return df
            
    except Exception as e:
        print(f"Yahoo Finance also failed for {symbol}: {e}")
        return None
        
    return None

if __name__ == "__main__":
    # For verification:
    verify_aggressive_labeling(sample_file_name='NIFTY_50/NIFTY_50_ohlcv_indicators.joblib', num_rows=10)
import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
import logging
from collections import deque
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

# Ensure the project root is in the Python path for imports
# Correcting the path to go up two directories from src/analysis to the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import core model and inference utilities
from src.models.lstm_model import AriaXaTModel
from src.inference.inference_utils import load_inference_artifacts, fetch_and_preprocess_data, make_prediction, config, device

# --- Logging Setup ---
# Configure a robust logging system for detailed insights and debugging
logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('backtest_log.log', mode='a')
                    ])
backtest_logger = logging.getLogger('BacktestRunner')
strategy_logger = logging.getLogger('TradingStrategy')
backtest_logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logs during development
strategy_logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logs during development


# --- MyLstmStrategy Class ---
class MyLstmStrategy(Strategy):
    """
    Trading strategy based on the Aria-XsT LSTM classification model.
    It uses the model's 'AGGRESSIVE UP (BUY!)' signal to enter long positions
    and 'NON-AGGRESSIVE (HOLD)' to close positions.
    """
    # Define any strategy parameters here (can be optimized later)
    confidence_threshold = 0.60  # Minimum confidence for a BUY signal
    stop_loss_pct = 0.005      # 0.5% stop-loss from entry price
    take_profit_pct = 0.01     # 1.0% take-profit from entry price

    def init(self):
        """
        Initializes the strategy, loading the trained model and scaler.
        This method is called once at the beginning of the backtest.
        """
        strategy_logger.info("Initializing MyLstmStrategy...")
        
        try:
            import torch
            import os
            model_path = os.path.join(config['local_models_save_dir'], 'best_aria_xat_model.pth')
            scaler_path = os.path.join(config['local_models_save_dir'], 'scaler_xaat_minmaxscaler.pkl')
            metadata_path = os.path.join(config['local_models_save_dir'], 'chunk_metadata_xaat.pkl')
            metadata = joblib.load(metadata_path)
            correct_input_features = metadata['num_features']
            self.lookback_window = metadata['lookback_window']
            self.features = metadata['features']

            self.model = AriaXaTModel(
                input_features=correct_input_features,
                hidden_size=config['hidden_dim'],
                num_layers=config['num_layers'],
                output_classes=3,
                dropout_rate=config.get('dropout_rate', 0.2)
            ).to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            self.scaler = joblib.load(scaler_path)
            strategy_logger.info(f"Model and scaler loaded successfully on {device}. Model expects {correct_input_features} input features.")
        except Exception as e:
            strategy_logger.error(f"Failed to load model or scaler: {e}")
            raise RuntimeError(f"Strategy initialization failed: {e}")

        # Custom logging for strategy-specific details
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def next(self):
        """
        Defines the trading logic for each new bar (time step) in the backtest.
        This method is called for every incoming historical data point.
        """
        # Calculate required initial bars for the strategy to start making predictions without error
        max_indicator_period = 26 # Based on EMA_Slow and MACD's slow period
        required_historical_bars = self.lookback_window + (2 * max_indicator_period) 
        
        if len(self.data.df) < required_historical_bars:
            strategy_logger.debug(f"Insufficient data for lookback_window and indicators. Current data points: {len(self.data.df)}. Required: {required_historical_bars}. Waiting for more data.")
            return # Wait for more data to accumulate

        # Create a temporary DataFrame for preprocessing from the historical slice
        # temp_df_raw will have OHLCV columns capitalized from backtesting.py's internal data.
        temp_df_raw = self.data.df.iloc[-required_historical_bars:].copy()

        strategy_logger.debug(f"Processing data slice ending at {self.data.index[-1]}")
        strategy_logger.debug(f"Slice head:\n{temp_df_raw.head().to_string()}")
        strategy_logger.debug(f"Slice tail:\n{temp_df_raw.tail().to_string()}")
        strategy_logger.debug(f"DEBUG: Data slice for preprocessing (temp_df_raw) shape: {temp_df_raw.shape}")
        strategy_logger.debug(f"DEBUG: Data slice for preprocessing (temp_df_raw) columns: {temp_df_raw.columns.tolist()}")

        # Ensure columns in temp_df_raw match the casing expected by fetch_and_preprocess_data (i.e., lowercase)
        # Create a mapping from current column names (likely capitalized or mixed) to the expected lowercase names from self.features
        rename_map_to_expected_features = {col: col.lower() for col in temp_df_raw.columns if col.lower() in self.features}
        temp_df_raw.rename(columns=rename_map_to_expected_features, inplace=True)
        
        # Verify that the feature columns (lowercase) are actually present and have no NaNs
        # It's crucial that temp_df_raw[self.features] doesn't result in an empty DataFrame after subsetting and dropping NaNs
        missing_features = [f for f in self.features if f not in temp_df_raw.columns]
        if missing_features:
            strategy_logger.error(f"Critical Error: Missing features in temp_df_raw after renaming for preprocessing: {missing_features}")
            return # Cannot proceed without all expected features

        strategy_logger.debug(f"DEBUG: Nulls in temp_df_raw relevant features BEFORE passing to fetch_and_preprocess_data:\n{temp_df_raw[self.features].isnull().sum().to_string()}")
        strategy_logger.debug(f"DEBUG: First few rows of temp_df_raw[self.features] BEFORE passing to fetch_and_preprocess_data:\n{temp_df_raw[self.features].head().to_string()}")

        # Preprocess the data slice using the shared utility function
        try:
            input_sequence, current_close_price = fetch_and_preprocess_data(
                symbol=config['target_symbol'], 
                interval=config['target_interval'], 
                lookback_window=self.lookback_window,
                features_list=self.features, # This is the crucial list of lowercase features
                scaler=self.scaler,
                td_api_key=config['twelvedata_api_key'], 
                df_input=temp_df_raw # Pass the historical slice with all columns (now with correct casing)
            )
            if input_sequence is None:
                strategy_logger.warning(f"Preprocessing returned None for data ending at {self.data.index[-1]}. Skipping prediction.")
                return

        except Exception as e:
            strategy_logger.error(f"Error during preprocessing at {self.data.index[-1]}: {e}")
            return # Skip if preprocessing fails

        # Make prediction
        with torch.no_grad():
            class_logits, reg_output = self.model(input_sequence.to(device))
            class_probs = torch.softmax(class_logits, dim=1)
            predicted_class = class_probs.argmax(dim=1).item()
            confidence = class_probs.max(dim=1).values.item()
            predicted_movement = reg_output.item()
        strategy_logger.info(f"At {self.data.index[-1]}, Close: {current_close_price:.2f}, Model predicts: {predicted_class} (Confidence: {confidence:.2%}), Predicted Move: {predicted_movement:.4f}")
        strategy_logger.debug(f"Predicted class: {predicted_class}, Confidence: {confidence}, Predicted movement: {predicted_movement}")

        # --- Trading Logic (Ideology: Aggressive Entry, Aggressive Exit) ---
        if self.position: # If currently in a position, manage exits
            # Take-Profit
            if self.data.Close[-1] >= self.trades[-1].entry_price * (1 + self.take_profit_pct):
                self.position.close()
                strategy_logger.info(f"TAKE PROFIT hit at {self.data.index[-1]}. Closed position for profit at {self.data.Close[-1]:.2f}.")
                self.winning_trades += 1
                return
            # Stop-Loss
            if self.data.Close[-1] <= self.trades[-1].entry_price * (1 - self.stop_loss_pct):
                self.position.close()
                strategy_logger.info(f"STOP LOSS hit at {self.data.index[-1]}. Closed position at {self.data.Close[-1]:.2f}.")
                self.losing_trades += 1
                return
            # Model Exit Signal (HOLD)
            if predicted_class == 1: # 1 = HOLD
                self.position.close()
                strategy_logger.info(f"HOLD signal at {self.data.index[-1]}. Closed position at {self.data.Close[-1]:.2f}.")
                return
        # If not in a position, and model gives BUY CALL or BUY PUT with sufficient confidence
        if not self.position and confidence >= self.confidence_threshold:
            if predicted_class == 2: # 2 = BUY CALL
                self.buy()
                self.trade_count += 1
                strategy_logger.info(f"BUY CALL signal at {self.data.index[-1]} (Conf: {confidence:.2%}). Opened position at {self.data.Close[-1]:.2f}.")
            elif predicted_class == 0: # 0 = BUY PUT
                self.sell()
                self.trade_count += 1
                strategy_logger.info(f"BUY PUT signal at {self.data.index[-1]} (Conf: {confidence:.2%}). Opened short position at {self.data.Close[-1]:.2f}.")
        else:
            strategy_logger.debug(f"No clear trading signal or conditions not met at {self.data.index[-1]}.")

# --- Main Execution for Backtest ---
if __name__ == "__main__":
    backtest_logger.info("-" * 75)
    backtest_logger.info(" Aria-XsT Historical Backtesting Framework")
    backtest_logger.info("-" * 75)
    backtest_logger.info("Starting backtest...")
    
    # 1. Load your large historical processed data file
    processed_data_file = os.path.join(
        config['local_processed_data_dir'],
        'All_Indices_XaT_enriched_ohlcv_indicators.joblib'
    )
    
    if not os.path.exists(processed_data_file):
        backtest_logger.error(f"Error: Processed data file not found at {processed_data_file}.")
        backtest_logger.error("Please ensure your processed NIFTY_50 data is in that location.")
        sys.exit(1)

    try:
        full_historical_df = joblib.load(processed_data_file)
        backtest_logger.info(f"Loaded historical data for backtesting: {processed_data_file}")
        backtest_logger.debug(f"Initial columns after loading: {full_historical_df.columns.tolist()}")
    except Exception as e:
        backtest_logger.error(f"Failed to load historical data from {processed_data_file}: {e}")
        sys.exit(1)
    
    # 2. Prepare data for backtesting.py
    # backtesting.py expects specific column names (Open, High, Low, Close, Volume)
    # and a DatetimeIndex.

    # --- CRITICAL FIX: Forcefully assign a new DatetimeIndex without using to_datetime(tz=...) ---
    try:
        full_historical_df.reset_index(drop=True, inplace=True)
        start_dt = pd.Timestamp('2024-01-01 09:15:00')
        new_index = [start_dt + pd.Timedelta(minutes=i) for i in range(len(full_historical_df))]
        full_historical_df.index = pd.DatetimeIndex(new_index).tz_localize('Asia/Kolkata')
        backtest_logger.warning("Forcefully reset and replaced DataFrame index with a new DatetimeIndex (Asia/Kolkata). Please ensure your processed data has a proper DatetimeIndex in `create_preprocessed_npy_data.py` for accuracy.")
    except Exception as e:
        backtest_logger.error(f"Could not forcefully reset and replace DataFrame index: {e}")
        sys.exit(1)

    # --- Standardize all columns to lowercase after loading ---
    # This ensures consistency with the `self.features` list from metadata.
    # We create a map from current column names to their lowercase versions.
    # We handle cases where some might already be lowercase, some capitalized.
    rename_to_lowercase_map = {col: col.lower() for col in full_historical_df.columns}
    full_historical_df.rename(columns=rename_to_lowercase_map, inplace=True)
    backtest_logger.debug(f"Columns after standardizing to lowercase: {full_historical_df.columns.tolist()}")

    # Define the OHLCV columns backtesting.py expects to be CAPITALIZED
    bt_required_ohlcv_cols_capitalized = ['Open', 'High', 'Low', 'Close', 'Volume']
    bt_required_ohlcv_cols_lowercase = [c.lower() for c in bt_required_ohlcv_cols_capitalized]
    
    # Verify that the *lowercase* OHLCV columns exist after standardization
    for col_lc in bt_required_ohlcv_cols_lowercase:
        if col_lc not in full_historical_df.columns:
            backtest_logger.error(f"Missing expected lowercase OHLCV column '{col_lc}' after standardization. Check your processed data source.")
            sys.exit(1)

    # --- Filter for NIFTY Market Trading Hours (9:15 AM to 3:30 PM IST) ---
    market_hours_df = full_historical_df.between_time('09:15', '15:30')
    backtest_logger.info(f"Filtered data to {len(market_hours_df)} bars within NIFTY trading hours (09:15-15:30 IST).")

    # --- Slice data correctly for the backtest (from filtered data) ---
    max_indicator_period = 26 
    try:
        metadata_path_main = os.path.join(config['local_models_save_dir'], 'chunk_metadata_xaat.pkl')
        metadata_main = joblib.load(metadata_path_main)
        actual_lookback_window = metadata_main['lookback_window']
    except Exception as e:
        backtest_logger.error(f"Failed to load lookback_window from metadata for warmup calculation: {e}. Using default 60.")
        actual_lookback_window = 60 # Fallback default
    
    required_strategy_warmup_bars = actual_lookback_window + (2 * max_indicator_period) 
    
    num_bars_for_test = 5000 
    
    if len(market_hours_df) < num_bars_for_test:
        backtest_logger.warning(f"Requested {num_bars_for_test} bars for backtest, but only {len(market_hours_df)} available after filtering. Using all available filtered data.")
        data_for_backtest = market_hours_df.copy()
    else:
        data_for_backtest = market_hours_df.iloc[-num_bars_for_test:].copy()
        
    if len(data_for_backtest) < required_strategy_warmup_bars:
        backtest_logger.error(f"The selected backtest data slice ({len(data_for_backtest)} bars) is too short for strategy warmup ({required_strategy_warmup_bars} bars). Please select a larger data range or ensure your filtered data is sufficient.")
        sys.exit(1)

    # Convert the OHLCV columns to capitalized names for backtesting.py ONLY on data_for_backtest
    # This must be done *after* slicing and *before* passing to Backtest
    # and ONLY for the OHLCV columns. Other columns should remain lowercase.
    rename_map_for_bt_ohlcv = {col_lc: col_cap for col_lc, col_cap in zip(bt_required_ohlcv_cols_lowercase, bt_required_ohlcv_cols_capitalized)}
    data_for_backtest.rename(columns=rename_map_for_bt_ohlcv, inplace=True)
    
    # Ensure all required OHLCV columns (capitalized) are present and numerical
    for col in bt_required_ohlcv_cols_capitalized:
        if col not in data_for_backtest.columns:
            backtest_logger.error(f"Missing required OHLCV column for backtesting.py: {col} after renaming. Check initial data source or rename logic.")
            sys.exit(1)
        data_for_backtest[col] = pd.to_numeric(data_for_backtest[col], errors='coerce')
    
    # Crucial: Drop NaNs AFTER converting to numeric, and only on the features that matter for processing
    # The 'nan counts' you showed indicated 0 NaNs initially. The problem must be introduced later,
    # likely by `pd.to_numeric(errors='coerce')` for columns that aren't purely numeric, or by `fetch_and_preprocess_data` itself.
    # The `self.features` list defines the actual features used by the model.
    try:
        metadata_for_features = joblib.load(os.path.join(config['local_models_save_dir'], 'chunk_metadata_xaat.pkl'))
        all_model_features_lowercase = metadata_for_features['features']
    except Exception as e:
        backtest_logger.error(f"Could not load features list from metadata for NaN check: {e}. Cannot proceed reliably.")
        sys.exit(1)

    # Check for NaNs only in the features the model *actually* uses (which are now all lowercase in data_for_backtest)
    # The OHLCV cols are capitalized in `data_for_backtest` but `self.features` are lowercase.
    # We need to consider both cases for dropping NaNs at this stage.
    
    # First, log NaNs for the OHLCV columns (capitalized)
    ohlcv_nan_check_cols = [c for c in bt_required_ohlcv_cols_capitalized if c in data_for_backtest.columns]
    if not data_for_backtest[ohlcv_nan_check_cols].isnull().sum().sum() == 0:
        backtest_logger.warning(f"NaNs found in OHLCV columns after numeric conversion:\n{data_for_backtest[ohlcv_nan_check_cols].isnull().sum().to_string()}")

    # Then, log NaNs for the other model features (lowercase)
    other_model_features_lowercase = [f for f in all_model_features_lowercase if f not in bt_required_ohlcv_cols_lowercase and f in data_for_backtest.columns]
    if not data_for_backtest[other_model_features_lowercase].isnull().sum().sum() == 0:
         backtest_logger.warning(f"NaNs found in other model features (lowercase):\n{data_for_backtest[other_model_features_lowercase].isnull().sum().to_string()}")
    
    # Combine all relevant columns (OHLCV capitalized + other features lowercase) for the final dropna
    all_relevant_cols_for_dropna = ohlcv_nan_check_cols + other_model_features_lowercase

    backtest_logger.info(f"Rows before final dropna based on model features: {len(data_for_backtest)}")
    data_for_backtest.dropna(subset=all_relevant_cols_for_dropna, inplace=True)
    backtest_logger.info(f"Rows after final dropna: {len(data_for_backtest)}")

    if len(data_for_backtest) == 0:
        backtest_logger.error("No data remaining after dropping NaNs based on model features. Cannot run backtest.")
        sys.exit(1)

    # After all manipulations, ensure the index is sorted and unique
    data_for_backtest = data_for_backtest.sort_index().loc[~data_for_backtest.index.duplicated(keep='first')]

    backtest_logger.info(f"Backtesting on {len(data_for_backtest)} bars from {data_for_backtest.index.min()} to {data_for_backtest.index.max()}")
    
    # 3. Run the Backtest
    bt = Backtest(data_for_backtest, MyLstmStrategy,
                  cash=1_000_000,     # Starting capital (e.g., 10 Lakh INR)
                  commission=.001,    # 0.1% commission per trade (adjust based on your broker)
                  exclusive_orders=True # Only one order (buy/sell) per bar
                 )

    backtest_logger.info("Running backtest...")
    stats = bt.run()
    
    backtest_logger.info("\n--- Backtest Results ---")
    backtest_logger.info(stats)

    # 4. Plotting Results
    plot_filename = 'backtest_results.html'
    backtest_logger.info(f"Generating plot: {plot_filename}...")
    try:
        bt.plot(filename=plot_filename, open_browser=True) # Save and open plot in browser
        backtest_logger.info(f"Backtest plot generated and opened in browser: {plot_filename}")
    except Exception as e:
        backtest_logger.error(f"Error generating or opening plot: {e}")

    backtest_logger.info("-" * 75)
    backtest_logger.info("Backtesting complete.")
    backtest_logger.info("-" * 75)
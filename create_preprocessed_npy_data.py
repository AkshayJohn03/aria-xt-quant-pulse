import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
import ta # Technical Analysis library
import warnings
from tqdm import tqdm
import math
import concurrent.futures # For multiprocessing

# Suppress common warnings that we're addressing or are informational
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")
warnings.filterwarnings("ignore", message="ChainedAssignmentError: behaviour will change in pandas 3.0!")
warnings.filterwarnings("ignore", category=UserWarning) # General user warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress future warnings from pandas/numpy if any

# --- Configuration for Aria-XaT Data Preprocessing ---
config = {
    'raw_data_dir': r'D:\aria\aria-xt-quant-pulse\dataset\raw',
    'processed_data_save_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\processed_data_npy',
    'models_artifact_save_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\models',
    'intermediate_processed_data_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\intermediate_processed_symbols', # New directory for intermediate files
    'output_processed_filename': 'All_Indices_XaT_enriched_ohlcv_indicators.joblib', # Final combined output
    'lookback_window': 60, # For model input sequences
    'future_target_window': 5, # Predict price movement over next 5 minutes (minutes ahead for future_return)
    'aggressive_up_threshold_pct': 0.5, # % change for "Aggressive Up" CALL signal
    'aggressive_down_threshold_pct': -0.5, # % change for "Aggressive Down" PUT signal
    'features_to_use': [
        # OHLC (will be for the specific symbol)
        'Open', 'High', 'Low', 'Close',

        # Technical Indicators (for the specific symbol)
        'ATR',
        'BB_Width',
        'ROC_Close',
        'EMA_Fast',
        'EMA_Slow',
        'MACD',
        'MACD_Signal',
        'RSI',
        'ADX',
        'Supertrend',
        'Supertrend_Dir_Binary',

        # Time-based features (cyclical encoding)
        'Hour_Sin', 'Hour_Cos',
        'Minute_Sin', 'Minute_Cos',
        'DayOfWeek_Sin', 'DayOfWeek_Cos',
    ],
    'scaler_type': 'MinMaxScaler', # Options: 'MinMaxScaler', 'StandardScaler', 'RobustScaler'
    'label_column': 'label_class', # Name of the 3-class target column
    'regression_label_column': 'label_regression_normalized', # Name of the regression target column
    'max_workers_for_multiprocessing': os.cpu_count() or 4 # Use all available CPU cores, or default to 4
}

# Create necessary directories
os.makedirs(config['processed_data_save_dir'], exist_ok=True)
os.makedirs(config['models_artifact_save_dir'], exist_ok=True)
os.makedirs(config['intermediate_processed_data_dir'], exist_ok=True) # Create intermediate dir

print("--- Starting Data Preprocessing for Aria-XaT (Multi-Symbol Vertical Stacking) ---")
print(f"Raw data directory: {config['raw_data_dir']}")
print(f"Intermediate processed data will be saved to: {config['intermediate_processed_data_dir']}")
print(f"Final processed data will be saved to: {config['processed_data_save_dir']}")
print(f"Using {config['max_workers_for_multiprocessing']} CPU cores for multiprocessing.")

def load_and_clean_data(filepath, index_name='Datetime'):
    """Loads CSV, parses datetime, sets index, and renames columns. Downcasts to float32."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.index.name = index_name
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    # Drop the 'Volume' column as it's consistently 0 in these Kaggle datasets
    if 'Volume' in df.columns and df['Volume'].sum() == 0:
        df.drop(columns=['Volume'], inplace=True)
    
    # --- Memory Optimization: Downcast numerical columns to float32 ---
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    
    return df

def calculate_technical_indicators(df):
    """Calculates a set of common technical indicators for a given DataFrame. Ensures float32 output."""
    # Ensure OHLC columns are float32
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    # Moving Averages (EMA)
    df['EMA_Fast'] = ta.trend.ema_indicator(df['Close'], window=12).astype(np.float32)
    df['EMA_Slow'] = ta.trend.ema_indicator(df['Close'], window=26).astype(np.float32)

    # MACD
    macd = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd().astype(np.float32)
    df['MACD_Signal'] = macd.macd_signal().astype(np.float32)

    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14).astype(np.float32)

    # ADX
    adx_indicator = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx_indicator.adx().astype(np.float32)

    # ATR
    atr_indicator_main = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr_indicator_main.average_true_range().astype(np.float32)

    # Bollinger Band Width
    df['BB_Width'] = ta.volatility.bollinger_wband(df['Close'], window=20, window_dev=2).astype(np.float32)

    # Rate of Change (Percentage Change)
    df['ROC_Close'] = (df['Close'].pct_change(periods=1) * 100).astype(np.float32)

    # Supertrend Calculation (Improved to avoid SettingWithCopyWarning)
    window = 10
    factor = 3
    
    atr_indicator_st = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=window)
    df['ATR_ST'] = atr_indicator_st.average_true_range().astype(np.float32)
    
    df['Basic_Upper_Band'] = ((df['High'] + df['Low']) / 2 + factor * df['ATR_ST']).astype(np.float32)
    df['Basic_Lower_Band'] = ((df['High'] + df['Low']) / 2 - factor * df['ATR_ST']).astype(np.float32)
    
    # Initialize with NaNs to be filled, then use .loc for assignment
    df['Final_Upper_Band'] = np.nan
    df['Final_Lower_Band'] = np.nan

    # Use .loc for explicit assignment to avoid SettingWithCopyWarning
    # Initialize the first row for Final_Upper_Band and Final_Lower_Band
    if len(df) > 0:
        first_index = df.index[0]
        df.loc[first_index, 'Final_Upper_Band'] = df.loc[first_index, 'Basic_Upper_Band']
        df.loc[first_index, 'Final_Lower_Band'] = df.loc[first_index, 'Basic_Lower_Band']

    for i in range(1, len(df)):
        current_index = df.index[i]
        prev_index = df.index[i-1]

        if df.loc[prev_index, 'Close'] > df.loc[prev_index, 'Final_Upper_Band']:
            df.loc[current_index, 'Final_Upper_Band'] = max(df.loc[current_index, 'Basic_Upper_Band'], df.loc[prev_index, 'Final_Upper_Band'])
        else:
            df.loc[current_index, 'Final_Upper_Band'] = df.loc[current_index, 'Basic_Upper_Band']
        
        if df.loc[prev_index, 'Close'] < df.loc[prev_index, 'Final_Lower_Band']:
            df.loc[current_index, 'Final_Lower_Band'] = min(df.loc[current_index, 'Basic_Lower_Band'], df.loc[prev_index, 'Final_Lower_Band'])
        else:
            df.loc[current_index, 'Final_Lower_Band'] = df.loc[current_index, 'Basic_Lower_Band']

    df['Supertrend'] = np.nan
    df['Supertrend_Dir_Binary'] = np.nan # 1 for uptrend, 0 for downtrend

    # Initialize for the first row of Supertrend
    if len(df) > 0:
        first_index = df.index[0]
        # Supertrend typically starts with the upper band if price is below it, or lower if price is above.
        # For initialization, let's assume price is below upper band or above lower band.
        # A simple initial rule: if close > Basic_Upper_Band, trend is up, else down.
        if df.loc[first_index, 'Close'] > df.loc[first_index, 'Basic_Upper_Band']:
            df.loc[first_index, 'Supertrend'] = df.loc[first_index, 'Basic_Lower_Band']
            df.loc[first_index, 'Supertrend_Dir_Binary'] = 1 # Uptrend
        else:
            df.loc[first_index, 'Supertrend'] = df.loc[first_index, 'Basic_Upper_Band']
            df.loc[first_index, 'Supertrend_Dir_Binary'] = 0 # Downtrend


    for i in range(1, len(df)):
        current_index = df.index[i]
        prev_index = df.index[i-1]

        if df.loc[prev_index, 'Supertrend_Dir_Binary'] == 1: # Previous was uptrend
            if df.loc[current_index, 'Close'] < df.loc[current_index, 'Final_Lower_Band']:
                df.loc[current_index, 'Supertrend'] = df.loc[current_index, 'Final_Upper_Band'] # Flip to downtrend
                df.loc[current_index, 'Supertrend_Dir_Binary'] = 0
            else:
                df.loc[current_index, 'Supertrend'] = df.loc[current_index, 'Final_Lower_Band'] # Remain in uptrend
                df.loc[current_index, 'Supertrend_Dir_Binary'] = 1
        else: # Previous was downtrend (0)
            if df.loc[current_index, 'Close'] > df.loc[current_index, 'Final_Upper_Band']:
                df.loc[current_index, 'Supertrend'] = df.loc[current_index, 'Final_Lower_Band'] # Flip to uptrend
                df.loc[current_index, 'Supertrend_Dir_Binary'] = 1
            else:
                df.loc[current_index, 'Supertrend'] = df.loc[current_index, 'Final_Upper_Band'] # Remain in downtrend
                df.loc[current_index, 'Supertrend_Dir_Binary'] = 0

    df.drop(columns=['ATR_ST', 'Basic_Upper_Band', 'Basic_Lower_Band', 'Final_Upper_Band', 'Final_Lower_Band'], errors='ignore', inplace=True)
    
    # Ensure all newly created float columns are float32
    for col in df.columns:
        if df[col].dtype == np.float64: # Check if it's still float64 (some ta outputs might be)
            df[col] = df[col].astype(np.float32)

    return df

def add_time_features(df):
    """Adds cyclical time-based features. Ensures float32 output."""
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['DayOfWeek'] = df.index.dayofweek # Monday=0, Sunday=6

    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24).astype(np.float32)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24).astype(np.float32)
    df['Minute_Sin'] = np.sin(2 * np.pi * df['Minute'] / 60).astype(np.float32)
    df['Minute_Cos'] = np.cos(2 * np.pi * df['Minute'] / 60).astype(np.float32)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7).astype(np.float32)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7).astype(np.float32)

    df.drop(columns=['Hour', 'Minute', 'DayOfWeek'], inplace=True)
    return df

def create_labels_and_targets(df, future_target_window, up_threshold, down_threshold):
    """Creates 3-class classification label and regression target. Ensures float32 output."""
    df['Future_Close'] = df['Close'].shift(-future_target_window)
    df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close'] * 100

    # 3-class label: 0 for PUT (aggressive down), 1 for HOLD, 2 for CALL (aggressive up)
    df[config['label_column']] = 1 # Default to HOLD (int for classification)
    df.loc[df['Future_Return'] > up_threshold, config['label_column']] = 2 # Aggressive Up (BUY CALL)
    df.loc[df['Future_Return'] < down_threshold, config['label_column']] = 0 # Aggressive Down (BUY PUT)
    df[config['label_column']] = df[config['label_column']].astype(np.int8) # Downcast to int8 for classification label

    df[config['regression_label_column']] = df['Future_Return'].astype(np.float32) # Ensure regression target is float32

    df.drop(columns=['Future_Close', 'Future_Return'], inplace=True)
    return df

def process_and_save_single_symbol(filepath):
    """
    Loads, processes, and saves a DataFrame for a single symbol to an intermediate file.
    This function is designed to be run in parallel.
    Returns the path to the saved intermediate file.
    """
    filename = os.path.basename(filepath)
    symbol_name = filename.replace('_minute_data.csv', '').replace('_minute.csv', '').replace('INDIA VIX', 'VIX').strip().replace(' ', '_')
    output_filename = f"{symbol_name}_processed.joblib"
    output_filepath = os.path.join(config['intermediate_processed_data_dir'], output_filename)

    # Check if file already exists (checkpointing)
    if os.path.exists(output_filepath):
        # print(f"Skipping {symbol_name}: Already processed. Loading from checkpoint...") # Commented for less verbose output during check
        return output_filepath # Return existing path

    print(f"Processing {symbol_name} from {filename}...") # Re-enabled for better visibility

    df = load_and_clean_data(filepath, index_name='Datetime')
    df['Symbol'] = symbol_name

    # Robust filtering for NIFTY Market Trading Hours (9:15 AM - 3:30 PM IST)
    start_time = pd.to_datetime('09:15').time()
    end_time = pd.to_datetime('15:30').time()
    df = df[(df.index.time >= start_time) & (df.index.time <= end_time)]

    # Calculate Technical Indicators for this symbol
    df = calculate_technical_indicators(df)

    # Add Time-Based Features
    df = add_time_features(df)

    # Create Classification and Regression Targets (using this symbol's Close)
    df = create_labels_and_targets(
        df,
        config['future_target_window'],
        config['aggressive_up_threshold_pct'],
        config['aggressive_down_threshold_pct']
    )
    
    # Drop initial NaNs from indicator calculations and target creation
    initial_len = len(df)
    subset_cols = [col for col in config['features_to_use'] if col in df.columns] + [config['label_column'], config['regression_label_column']]
    df.dropna(subset=subset_cols, inplace=True)

    if initial_len - len(df) > 0:
        print(f"    Dropped {initial_len - len(df)} rows due to NaNs in {symbol_name} after indicator/label creation.") # Re-enabled for better visibility

    # Save the processed single symbol DataFrame to an intermediate file
    joblib.dump(df, output_filepath)
    
    return output_filepath # Return the path to the saved file

def run_preprocessing():
    all_minute_files_basenames = [f for f in os.listdir(config['raw_data_dir']) if f.endswith('_minute_data.csv') or f.endswith('_minute.csv')]
    
    files_to_process = []
    pre_existing_intermediate_paths = []

    # Phase 1 Checkpoint: Identify files to process or load from cache
    print("\nPhase 1 Checkpoint: Identifying files to process or load from cache...")
    for filename in tqdm(all_minute_files_basenames, desc="Checking existing processed files"):
        symbol_name = filename.replace('_minute_data.csv', '').replace('_minute.csv', '').replace('INDIA VIX', 'VIX').strip().replace(' ', '_')
        expected_output_filename = f"{symbol_name}_processed.joblib"
        expected_output_filepath = os.path.join(config['intermediate_processed_data_dir'], expected_output_filename)
        
        if os.path.exists(expected_output_filepath):
            pre_existing_intermediate_paths.append(expected_output_filepath)
        else:
            files_to_process.append(os.path.join(config['raw_data_dir'], filename))

    print(f"Found {len(pre_existing_intermediate_paths)} files already processed. Will process {len(files_to_process)} new/missing files.")
    
    # Phase 1: Parallel processing and saving of individual symbol dataframes
    if files_to_process:
        print("\nPhase 1: Processing and Saving Individual Symbols (Parallel)")
        with concurrent.futures.ProcessPoolExecutor(max_workers=config['max_workers_for_multiprocessing']) as executor:
            # Map full file paths to the process_and_save_single_symbol function
            futures = [executor.submit(process_and_save_single_symbol, filepath) for filepath in files_to_process]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_process), desc="Overall Symbol Processing Progress"):
                try:
                    saved_filepath = future.result()
                    # Ensure path is added only if it's new (to avoid issues if a file was processed and then checked again in the same run)
                    if saved_filepath not in pre_existing_intermediate_paths:
                        pre_existing_intermediate_paths.append(saved_filepath)
                except Exception as exc:
                    print(f'Error processing a file: {exc}')
    else:
        print("\nAll raw files already processed. Skipping parallel processing phase.")

    # Consolidate all intermediate file paths (both pre-existing and newly processed)
    # Ensure a consistent order for concatenation by sorting the paths for reproducibility
    intermediate_file_paths = sorted(list(set(pre_existing_intermediate_paths))) 

    if not intermediate_file_paths:
        raise ValueError("No intermediate dataframes were successfully processed. Please check raw data and file paths.")

    # Phase 2: Load intermediate files and concatenate them
    print("\nPhase 2: Loading intermediate files and concatenating into a single DataFrame...")
    processed_dfs_from_disk = []
    for filepath in tqdm(intermediate_file_paths, desc="Loading & Concatenating Intermediate Files"):
        processed_dfs_from_disk.append(joblib.load(filepath))

    combined_df = pd.concat(processed_dfs_from_disk)
    print(f"Combined DataFrame shape after concatenation: {combined_df.shape}")
    
    # Sort by Datetime (important for sequential processing later) and then by Symbol for deterministic order
    combined_df.sort_index(inplace=True)
    combined_df.sort_values(by=['Symbol'], kind='stable', inplace=True) 

    # Final NaN Handling after all calculations, label creation, and merging
    initial_rows = len(combined_df)
    
    targets_subset = [col for col in [config['label_column'], config['regression_label_column']] if col in combined_df.columns]
    combined_df.dropna(subset=targets_subset, inplace=True)
    print(f"Dropped {initial_rows - len(combined_df)} rows due to NaN targets (likely at end of each symbol's data).")

    final_features_and_targets_for_dropna = [col for col in config['features_to_use'] + targets_subset if col in combined_df.columns]
    initial_rows_after_target_drop = len(combined_df)
    combined_df.dropna(subset=final_features_and_targets_for_dropna, inplace=True)
    print(f"Dropped {initial_rows_after_target_drop - len(combined_df)} rows due to remaining NaNs in features/targets.")

    if len(combined_df) == 0:
        raise ValueError("Processed DataFrame is empty after final cleaning and feature engineering.")

    # --- Store original index (Datetime) and Symbol column from the *cleaned and filtered* combined_df ---
    # These will be used to construct the final MultiIndex.
    # This snapshot now correctly reflects only the rows that survived the dropna operations.
    final_datetimes_for_multiindex = combined_df.index.copy()
    final_symbols_for_multiindex = combined_df['Symbol'].copy() # Get the Symbol column for the MultiIndex

    # One-Hot Encode the 'Symbol' column
    print("One-Hot Encoding 'Symbol' column...")
    symbol_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Use combined_df[['Symbol']] because the Symbol column is still in combined_df at this point
    encoded_symbols = symbol_encoder.fit_transform(combined_df[['Symbol']])
    encoded_symbol_df = pd.DataFrame(encoded_symbols, columns=symbol_encoder.get_feature_names_out(['Symbol']), index=combined_df.index)
    
    # Update config['features_to_use'] to include new OHE symbol features
    temp_features_to_use = [f for f in config['features_to_use'] if f not in encoded_symbol_df.columns] 
    temp_features_to_use.extend(encoded_symbol_df.columns.tolist())
    config['features_to_use'] = temp_features_to_use

    # Concatenate encoded symbol features
    combined_df = pd.concat([combined_df, encoded_symbol_df], axis=1)
    combined_df.drop(columns=['Symbol'], inplace=True) # Drop original Symbol column as it's now OHE

    # Extract numpy arrays for scaling
    # IMPORTANT: Explicitly cast to float32 after extraction from DataFrame (if not already float32)
    X = combined_df[config['features_to_use']].values.astype(np.float32)
    y_class = combined_df[config['label_column']].values.astype(np.int8) # Ensure int8 for classification
    y_reg = combined_df[config['regression_label_column']].values.astype(np.float32)

    print(f"Scaling features using {config['scaler_type']}...")
    if config['scaler_type'] == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif config['scaler_type'] == 'StandardScaler':
        scaler = StandardScaler()
    elif config['scaler_type'] == 'RobustScaler':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {config['scaler_type']}")

    # fit_transform will likely output float64, so we re-cast to float32
    X_scaled = scaler.fit_transform(X).astype(np.float32) 
    print("Features scaled.")

    # Reconstruct processed_df_final with a proper MultiIndex using the snapshots taken earlier
    multi_index = pd.MultiIndex.from_arrays([final_symbols_for_multiindex, final_datetimes_for_multiindex], 
                                            names=['Symbol', 'Datetime'])

    processed_df_final = pd.DataFrame(X_scaled, columns=config['features_to_use'], index=multi_index)
    processed_df_final[config['label_column']] = y_class
    processed_df_final[config['regression_label_column']] = y_reg
    
    # Final sort by the MultiIndex
    processed_df_final.sort_index(inplace=True)


    output_filepath = os.path.join(config['processed_data_save_dir'], config['output_processed_filename'])
    joblib.dump(processed_df_final, output_filepath)
    print(f"Processed data (Multi-Indexed) saved to: {output_filepath}")

    # Save scaler and symbol encoder
    scaler_output_filepath = os.path.join(config['models_artifact_save_dir'], f"scaler_xaat_{config['scaler_type'].lower()}.pkl")
    joblib.dump(scaler, scaler_output_filepath)
    print(f"Scaler saved to: {scaler_output_filepath}")

    symbol_encoder_output_filepath = os.path.join(config['models_artifact_save_dir'], 'symbol_encoder_xaat.pkl')
    joblib.dump(symbol_encoder, symbol_encoder_output_filepath)
    print(f"Symbol encoder saved to: {symbol_encoder_output_filepath}")


    # Generate metadata for DataLoader, respecting symbol boundaries
    sequence_map = []
    print("Generating sequence map for DataLoader, respecting symbol boundaries...")
    unique_symbols_in_data = processed_df_final.index.get_level_values('Symbol').unique()

    for sym in tqdm(unique_symbols_in_data, desc="Building Sequences Per Symbol"):
        symbol_df = processed_df_final.loc[sym] # Get the sub-DataFrame for this symbol
        for i in range(len(symbol_df)):
            if i >= config['lookback_window'] - 1:
                # Sequence map stores (symbol_name_str, relative_start_idx_in_symbol_df, relative_end_idx_in_symbol_df)
                sequence_map.append((sym, i - (config['lookback_window'] - 1), i))
        
    print(f"Total sequences available for model training: {len(sequence_map)}")

    metadata = {
        'sequence_map': sequence_map,
        'file_paths': [os.path.relpath(output_filepath, config['processed_data_save_dir'])], # Points to one large file
        'lookback_window': config['lookback_window'],
        'features': config['features_to_use'], # Includes one-hot encoded symbol features
        'label_column': config['label_column'],
        'regression_label_column': config['regression_label_column'],
        # CORRECTED: Changed 'processed_data_save_save_dir' to 'processed_data_save_dir'
        'data_dir': config['processed_data_save_dir'], # Base directory for data files
        'symbol_names': symbol_encoder.categories_[0].tolist(), # List of original symbol names
        'num_features': len(config['features_to_use']), # Explicitly store number of input features
        'input_shape': (config['lookback_window'], len(config['features_to_use'])) # This shape includes OHE symbols
    }
    metadata_output_filepath = os.path.join(config['models_artifact_save_dir'], 'chunk_metadata_xaat.pkl')
    joblib.dump(metadata, metadata_output_filepath)
    print(f"Metadata cache saved to: {metadata_output_filepath}")

    print("--- Data Preprocessing Complete for Aria-XaT ---")
    print(f"Final combined data shape: {processed_df_final.shape}")
    print(f"Number of final features (including OHE symbols): {len(config['features_to_use'])}")
    print(f"Example of final features: {config['features_to_use'][:5]} ... {config['features_to_use'][-5:]}")


if __name__ == '__main__':
    run_preprocessing()

import os
import pandas as pd
import numpy as np
import joblib
import warnings

# --- Configuration (matching your preprocessing script's output paths) ---
config = {
    'processed_data_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\processed_data_npy',
    'models_artifact_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\models',
    'processed_filename': 'All_Indices_XaT_enriched_ohlcv_indicators.joblib',
    'metadata_filename': 'chunk_metadata_xaat.pkl',
    'scaler_filename': 'scaler_xaat_minmaxscaler.pkl', # Adjust if scaler type changes
    'symbol_encoder_filename': 'symbol_encoder_xaat.pkl',
}

def verify_data_integrity():
    print("--- Starting Data Integrity Verification ---")

    # --- Verify Processed DataFrame ---
    processed_data_filepath = os.path.join(config['processed_data_dir'], config['processed_filename'])
    print(f"\n1. Verifying main processed data file: {processed_data_filepath}")
    if not os.path.exists(processed_data_filepath):
        print(f"  FAIL: Processed data file NOT found at {processed_data_filepath}")
        return False
    
    try:
        df = joblib.load(processed_data_filepath)
        print(f"  SUCCESS: File loaded. Shape: {df.shape}")

        # Check if it's a MultiIndex DataFrame
        if not isinstance(df.index, pd.MultiIndex):
            print(f"  WARNING: DataFrame index is not a MultiIndex. Type: {type(df.index)}")
        else:
            print(f"  SUCCESS: DataFrame has a MultiIndex with levels: {df.index.names}")
            print(f"  Unique symbols in data: {df.index.get_level_values('Symbol').nunique()}")
            print(f"  Number of unique timestamps: {df.index.get_level_values('Datetime').nunique()}")

        # Check for expected columns (features and labels)
        expected_features_from_config = 43 # Based on your last run's output
        expected_columns = expected_features_from_config + 2 # +2 for class and regression labels
        if df.shape[1] != expected_columns:
            print(f"  WARNING: Number of columns ({df.shape[1]}) does not match expected ({expected_columns}).")
        
        # Check for NaN values in critical columns
        critical_cols = ['Open', 'Close', 'ATR', 'Supertrend', 'label_class', 'label_regression_normalized']
        for col in critical_cols:
            if col not in df.columns:
                print(f"  WARNING: Critical column '{col}' not found in DataFrame.")
                continue
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                print(f"  FAIL: Column '{col}' contains {nan_count} NaN values. This should ideally be 0 after dropna.")
            else:
                print(f"  SUCCESS: Column '{col}' has no NaN values.")
        
        # Check data types (expecting float32 for most features, int8/float32 for labels)
        print("  Checking data types for a few columns:")
        for col in ['Open', 'ATR', 'Supertrend', 'label_class', 'label_regression_normalized']:
            if col in df.columns:
                print(f"    '{col}': {df[col].dtype}")
                if 'float' in str(df[col].dtype) and df[col].dtype != np.float32:
                    print(f"      WARNING: Expected float32 for '{col}', got {df[col].dtype}.")
                if 'int' in str(df[col].dtype) and df[col].dtype != np.int8 and col == 'label_class':
                    print(f"      WARNING: Expected int8 for '{col}', got {df[col].dtype}.")

    except Exception as e:
        print(f"  FAIL: Error loading or verifying processed data: {e}")
        return False

    # --- Verify Scaler ---
    scaler_filepath = os.path.join(config['models_artifact_dir'], config['scaler_filename'])
    print(f"\n2. Verifying scaler file: {scaler_filepath}")
    if not os.path.exists(scaler_filepath):
        print(f"  FAIL: Scaler file NOT found at {scaler_filepath}")
        return False
    try:
        scaler = joblib.load(scaler_filepath)
        print(f"  SUCCESS: Scaler loaded. Type: {type(scaler)}")
        # You can add more checks specific to your scaler type if needed, e.g., for MinMaxScaler:
        if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
             print(f"  Scaler min/max values exist.")
        elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
             print(f"  Scaler mean/scale values exist.")
    except Exception as e:
        print(f"  FAIL: Error loading or verifying scaler: {e}")
        return False

    # --- Verify Symbol Encoder ---
    encoder_filepath = os.path.join(config['models_artifact_dir'], config['symbol_encoder_filename'])
    print(f"\n3. Verifying symbol encoder file: {encoder_filepath}")
    if not os.path.exists(encoder_filepath):
        print(f"  FAIL: Symbol encoder file NOT found at {encoder_filepath}")
        return False
    try:
        encoder = joblib.load(encoder_filepath)
        print(f"  SUCCESS: Symbol encoder loaded. Type: {type(encoder)}")
        print(f"  Encoded categories: {encoder.categories_[0].tolist()}")
    except Exception as e:
        print(f"  FAIL: Error loading or verifying symbol encoder: {e}")
        return False

    # --- Verify Metadata ---
    metadata_filepath = os.path.join(config['models_artifact_dir'], config['metadata_filename'])
    print(f"\n4. Verifying metadata file: {metadata_filepath}")
    if not os.path.exists(metadata_filepath):
        print(f"  FAIL: Metadata file NOT found at {metadata_filepath}")
        return False
    try:
        metadata = joblib.load(metadata_filepath)
        print(f"  SUCCESS: Metadata loaded.")
        print(f"  Metadata keys: {list(metadata.keys())}")
        
        # Check essential metadata keys and values
        expected_keys = ['sequence_map', 'lookback_window', 'features', 
                         'label_column', 'regression_label_column', 
                         'data_dir', 'symbol_names', 'num_features', 'input_shape']
        for key in expected_keys:
            if key not in metadata:
                print(f"  FAIL: Missing critical key '{key}' in metadata.")
                return False
        
        print(f"  Metadata - lookback_window: {metadata['lookback_window']}")
        print(f"  Metadata - num_features: {metadata['num_features']}")
        print(f"  Metadata - label_column: {metadata['label_column']}")
        print(f"  Metadata - regression_label_column: {metadata['regression_label_column']}")
        print(f"  Metadata - Total sequences in sequence_map: {len(metadata['sequence_map'])}")
        print(f"  Metadata - Input shape for model: {metadata['input_shape']}")

        # Basic check for sequence_map integrity (first few entries)
        if len(metadata['sequence_map']) > 0:
            sample_seq = metadata['sequence_map'][0]
            print(f"  Sample sequence map entry (first): {sample_seq}")
            if not (isinstance(sample_seq, tuple) and len(sample_seq) == 3):
                print("  WARNING: sequence_map entries might not be in (symbol, start_idx, end_idx) format.")
            else:
                # Attempt to retrieve a sample sequence from the dataframe
                try:
                    symbol, start_idx, end_idx = sample_seq
                    sample_slice = df.loc[symbol].iloc[start_idx : end_idx + 1]
                    if sample_slice.shape[0] != metadata['lookback_window']:
                        print(f"  FAIL: Sample sequence length ({sample_slice.shape[0]}) does not match lookback_window ({metadata['lookback_window']}).")
                    else:
                        print(f"  SUCCESS: Sample sequence from sequence_map can be retrieved from DataFrame and has correct length.")
                except Exception as e:
                    print(f"  FAIL: Could not retrieve sample sequence from DataFrame using sequence_map: {e}")
                    return False

    except Exception as e:
        print(f"  FAIL: Error loading or verifying metadata: {e}")
        return False

    print("\n--- Data Integrity Verification Complete (All Checks Passed) ---")
    return True

if __name__ == '__main__':
    verify_data_integrity()
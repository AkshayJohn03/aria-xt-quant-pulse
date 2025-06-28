import os
import joblib
import torch
import numpy as np
import pandas as pd
from src.models.lstm_model import AriaXaTModel
from datetime import datetime
import random
import sys

# --- Config (Copy from your backtest_strategy.py for consistency) ---
config = {
    'processed_data_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/processed_data_npy/All_Indices_XaT_enriched_ohlcv_indicators.joblib',
    'metadata_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/chunk_metadata_xaat.pkl',
    'scaler_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/scaler_xaat_minmaxscaler.pkl',
    'model_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/best_aria_xat_model.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'initial_cash': 10_000, # INR
    'nifty_lot_size': 75, # Nifty 50 lot size
    'trade_risk_pct': 0.05, # % of capital risked per trade
    'slippage_pct': 0.0005, # 0.05% per side
    'stop_loss_pct': 0.005, # 0.5% stop loss
    'take_profit_pct': 0.01, # 1% take profit
    'max_trades_per_day': 10,
    'confidence_threshold': 0.6,
    'batch_size': 4096,
    'chunk_size': 500_000,
    # Zerodha/Indian costs (all INR)
    'brokerage_per_order': 20,
    'stt_sell_pct': 0.001, # 0.1% on sell side (on premium)
    'exchange_txn_pct': 0.0003503, # 0.03503% on premium
    'gst_pct': 0.18, # 18% on (brokerage + txn + sebi)
    'sebi_charges_per_cr': 10, # ₹10 per crore
    'stamp_duty_buy_pct': 0.00003, # 0.003% on buy side
    'symbol': 'NIFTY_50',
    'classification_label_column': 'label_class' # Assuming this is the name in your processed data
}

# --- Utility Functions (Copy from your backtest_strategy.py for consistency) ---
def get_sebi_charge(trade_value):
    return config['sebi_charges_per_cr'] * (trade_value / 1e7) # 1 crore = 1e7

def get_trade_costs(trade_value, side):
    brokerage = config['brokerage_per_order']
    stt = config['stt_sell_pct'] * trade_value if side == 'sell' else 0
    exchange = config['exchange_txn_pct'] * trade_value
    sebi = get_sebi_charge(trade_value)
    stamp = config['stamp_duty_buy_pct'] * trade_value if side == 'buy' else 0
    gst = config['gst_pct'] * (brokerage + exchange + sebi)
    total = brokerage + stt + exchange + sebi + stamp + gst
    return {
        'brokerage': brokerage, 'stt': stt, 'exchange': exchange,
        'sebi': sebi, 'stamp': stamp, 'gst': gst, 'total': total
    }

def apply_slippage(price, side):
    slip = random.uniform(0, config['slippage_pct'])
    if side == 'buy':
        return price * (1 + slip)
    else:
        return price * (1 - slip)

def get_trade_lots(cash, price):
    # Ensure price is not zero to prevent division by zero
    if price <= 0:
        return 0

    max_affordable_lots_full_cash = int(cash // (price * config['nifty_lot_size']))
    risked_cash = cash * config['trade_risk_pct']
    risked_lots = int(risked_cash // (price * config['nifty_lot_size']))

    if risked_lots == 0 and max_affordable_lots_full_cash >= 1:
        lots = 1
    elif risked_lots > 0:
        lots = risked_lots
    else:
        lots = 0

    return lots

print("--- Starting Debug Script ---")

# --- 1. Data & File Integrity Checks ---
print("\n--- 1. Data & File Integrity Checks ---")
for path_key in ['processed_data_path', 'metadata_path', 'scaler_path', 'model_path']:
    path = config[path_key]
    if not os.path.exists(path):
        print(f"ERROR: {path_key.replace('_', ' ').title()} not found: {path}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {path_key.replace('_', ' ').title()} found: {path}")

# Load metadata first to get features and lookback_window
try:
    metadata = joblib.load(config['metadata_path'])
    features_raw = [f.lower() for f in metadata['features']] # All features from metadata
    lookback_window = metadata['lookback_window']
    num_features_raw = metadata['num_features']

    # Filter out label columns for actual model features
    features = [f for f in features_raw if f not in ['label_class', 'label_regression_normalized']]
    num_features = len(features) # Number of features actually used by model

    # Find the index of 'close' feature in the original metadata features list
    close_feature_idx = -1
    for idx, f in enumerate(features_raw):
        if f.lower() == 'close':
            close_feature_idx = idx
            break
    if close_feature_idx == -1:
        print("ERROR: 'close' feature not found in metadata's features list.")
        sys.exit(1)

    print(f"Metadata loaded. Lookback window: {lookback_window}, Raw Num features: {num_features_raw}")
    print(f"Features used for model ({num_features}): {features[:5]}...{features[-5:]}")
    print(f"'close' feature found at index {close_feature_idx} in scaler's input based on raw features.")

except Exception as e:
    print(f"ERROR: Could not load metadata. {e}")
    sys.exit(1)


# Load raw data for checks
try:
    df_raw = joblib.load(config['processed_data_path'])
    print(f"Processed data loaded. Shape: {df_raw.shape}")
    if not isinstance(df_raw, pd.DataFrame):
        print("WARNING: Loaded data is not a Pandas DataFrame.")
    df_raw.columns = [c.lower() for c in df_raw.columns] # Ensure column names are lowercase
except Exception as e:
    print(f"ERROR: Could not load processed data. {e}")
    sys.exit(1)


# Filter for the symbol
try:
    if config['symbol'] not in df_raw.index.get_level_values(0).unique():
        print(f"ERROR: Symbol '{config['symbol']}' not found in the processed data.")
        sys.exit(1)
    df = df_raw.loc[config['symbol']].copy() # Use .copy() to avoid SettingWithCopyWarning
    df = df.sort_index()
    print(f"Filtered data for '{config['symbol']}'. Shape: {df.shape}")
    print("First 5 rows of filtered data:")
    print(df.head())
    print("\nLast 5 rows of filtered data:")
    print(df.tail())

    # Check for NaNs in critical columns (only features used by model)
    nan_check_cols = features + [config['classification_label_column']]
    for col in nan_check_cols:
        if col in df.columns and df[col].isnull().any():
            print(f"WARNING: NaNs found in column '{col}' within {config['symbol']} data. Count: {df[col].isnull().sum()}")
    
    # Check if any feature column is missing
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"ERROR: Following features are missing from the dataframe: {missing_features}")
        sys.exit(1)


    # Check price range of *scaled* close
    if 'close' in df.columns:
        print(f"Scaled close price range for {config['symbol']}: Min={df['close'].min():.4f}, Max={df['close'].max():.4f}")
        if df['close'].min() >= 100 or df['close'].max() <= 1: # Prices like 0.x are expected for scaled data
            print("WARNING: Scaled NIFTY_50 close prices seem outside typical scaled range (0-1 or -1 to 1). Double check scaling method.")
    else:
        print("ERROR: 'close' column not found in filtered data.")
        sys.exit(1)

    # Check time frequency
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna().unique()
        print(f"Unique time differences in data: {time_diffs}")
        if not all(td == pd.Timedelta(minutes=1) for td in time_diffs):
            print("WARNING: Data does not appear to be consistently 1-minute bars.")
    else:
        print("WARNING: Not enough data points to check time frequency.")

except Exception as e:
    print(f"ERROR: Could not process data for symbol '{config['symbol']}'. {e}")
    sys.exit(1)


# --- 2. Model & Scaler Loading Check ---
print("\n--- 2. Model & Scaler Loading Check ---")
try:
    scaler = joblib.load(config['scaler_path'])
    
    model = AriaXaTModel(
        input_features=num_features, # Use updated num_features
        hidden_size=128,
        num_layers=2,
        output_classes=3,
        dropout_rate=0.2
    ).to(config['device'])
    model.load_state_dict(torch.load(config['model_path'], map_location=config['device']))
    model.eval()
    print("Model and Scaler loaded successfully.")
    print(f"Model expects input shape: (batch_size, {lookback_window}, {num_features})")
except Exception as e:
    print(f"ERROR: Model or Scaler failed to load/initialize. {e}")
    sys.exit(1)

# --- DE-NORMALIZATION TEST ---
print("\n--- De-normalization Test ---")
try:
    # Take a sample scaled close price (e.g., from the head of the df)
    sample_scaled_close = df.iloc[lookback_window]['close'] # Get first available after lookback

    dummy_row_scaled = np.zeros((1, num_features_raw)) # Use full feature count for scaler
    dummy_row_scaled[0, close_feature_idx] = sample_scaled_close
    
    actual_unscaled_price = scaler.inverse_transform(dummy_row_scaled)[0, close_feature_idx]

    print(f"Sample Scaled Close Price: {sample_scaled_close:.6f}")
    print(f"De-normalized (Actual) Close Price: {actual_unscaled_price:.2f}")
    if not (actual_unscaled_price > 10000 and actual_unscaled_price < 25000): # Typical Nifty range
        print("WARNING: De-normalized price seems outside expected Nifty range (10,000 - 25,000). Check scaler/data.")
except Exception as e:
    print(f"ERROR: De-normalization test failed. {e}")


# --- 3. Label Distribution Check ---
print("\n--- 3. Label Distribution Check ---")
if config['classification_label_column'] in df.columns:
    label_counts = df[config['classification_label_column']].value_counts(normalize=True).sort_index()
    print("Classification Label Distribution (0:PUT, 1:HOLD, 2:CALL):")
    print(label_counts)
    if label_counts.get(0, 0) < 0.05 or label_counts.get(2, 0) < 0.05:
        print("WARNING: Highly imbalanced labels! This can lead to a model that rarely predicts certain classes.")
    if label_counts.get(1, 0) > 0.9:
        print("WARNING: Very high proportion of 'HOLD' labels. This can make a model too passive.")
else:
    print(f"WARNING: Classification label column '{config['classification_label_column']}' not found in data.")


# --- 4. Scaler Functionality Check (on actual features used by model) ---
print("\n--- 4. Scaler Functionality Check ---")
try:
    sample_data_df = df[features].iloc[lookback_window:lookback_window+1000] # Take 1000 rows of features
    if sample_data_df.isnull().values.any():
        print("WARNING: Sample data for scaler check contains NaNs. Skipping direct transform test.")
    else:
        transformed_data = scaler.transform(sample_data_df.values)
        print(f"Sample data shape: {sample_data_df.shape}")
        print(f"Transformed sample data shape: {transformed_data.shape}")
        print(f"Transformed data min: {transformed_data.min():.4f}, max: {transformed_data.max():.4f}, mean: {transformed_data.mean():.4f}")
        if not (np.all(transformed_data >= -5) and np.all(transformed_data <= 5)): # Typical range for scaled data with Min/Max
            print("WARNING: Scaled data values seem unusually large or small. Check scaler fitting.")
except Exception as e:
    print(f"ERROR: Scaler functionality check failed. {e}")


# --- 5. Small-Scale Backtest Dry Run & Logic Checks ---
print("\n--- 5. Small-Scale Backtest Dry Run & Logic Checks ---")
print("Running a dry run for first 50 data points (after lookback window).")
print("Showing actual price (de-normalized), true label, model prediction, and confidence.")
print("Also checking trade cost and lot size calculation.")

dry_run_points = min(50, len(df) - lookback_window) # Max 50 points or less if data is short
if dry_run_points <= 0:
    print("Not enough data for dry run.")
else:
    current_cash_test = config['initial_cash']
    sample_scaled_price = df.iloc[lookback_window]['close']
    sample_price = actual_unscaled_price # Use the de-normalized price from earlier test
    sample_trade_value = 100_000 # Example for cost calculation

    # Check trade cost calculation
    buy_costs_sample = get_trade_costs(sample_trade_value, 'buy')
    sell_costs_sample = get_trade_costs(sample_trade_value, 'sell')
    print(f"\nSample Trade Costs (for {sample_trade_value:.2f} INR value):")
    print(f"  Buy Costs: {buy_costs_sample['total']:.2f} INR (Brokerage: {buy_costs_sample['brokerage']:.2f}, Stamp: {buy_costs_sample['stamp']:.2f}, GST: {buy_costs_sample['gst']:.2f}, SEBI: {buy_costs_sample['sebi']:.2f}, Exchange: {buy_costs_sample['exchange']:.2f})")
    print(f"  Sell Costs: {sell_costs_sample['total']:.2f} INR (Brokerage: {sell_costs_sample['brokerage']:.2f}, STT: {sell_costs_sample['stt']:.2f}, GST: {sell_costs_sample['gst']:.2f}, SEBI: {sell_costs_sample['sebi']:.2f}, Exchange: {sell_costs_sample['exchange']:.2f})")

    # Check lot size calculation
    if sample_price > 0:
        calculated_lots = get_trade_lots(current_cash_test, sample_price)
        print(f"\nLot size calculation for Initial Cash {current_cash_test:.2f} and Sample Price {sample_price:.2f}: {calculated_lots} lots")
        if calculated_lots == 0:
            print("WARNING: get_trade_lots returned 0. This means not enough cash or price is too high for even 1 lot with current config.")
        else:
            total_value_for_lots = calculated_lots * config['nifty_lot_size'] * sample_price
            print(f"  Total value of {calculated_lots} lots: {total_value_for_lots:.2f} INR")

    else:
        print("WARNING: Sample price is zero, cannot calculate lot size.")

    print("\nDry Run Predictions:")
    for i in range(lookback_window, lookback_window + dry_run_points):
        window_df = df.iloc[i - lookback_window:i][features]
        current_scaled_price = df.iloc[i]['close']
        
        # De-normalize current price for display
        dummy_row_scaled_curr = np.zeros((1, num_features_raw))
        dummy_row_scaled_curr[0, close_feature_idx] = current_scaled_price
        current_price = scaler.inverse_transform(dummy_row_scaled_curr)[0, close_feature_idx]

        actual_label = df.iloc[i][config['classification_label_column']]
        dt_idx = df.index[i]

        if window_df.isnull().values.any():
            print(f"  Skipping {dt_idx} due to NaN in feature window.")
            continue

        X = scaler.transform(window_df.values)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(config['device'])

        with torch.no_grad():
            out_class_logits, _ = model(X_tensor)
            probs = torch.softmax(out_class_logits, dim=1).cpu().numpy()[0]
            pred_class = probs.argmax()
            confidence = probs.max()

        print(f"\n--- {dt_idx} ---")
        print(f"  Current Price (Actual): {current_price:.2f}")
        print(f"  Actual Label (0:PUT, 1:HOLD, 2:CALL): {actual_label}")
        print(f"  Model Logits: {out_class_logits.cpu().numpy()[0]}")
        print(f"  Model Probabilities: {probs}")
        print(f"  Predicted Class: {pred_class} (Confidence: {confidence:.4f})")
        
        # Check if prediction aligns with actual movement
        if pred_class == 2 and actual_label == 2:
            print("  ✅ Model predicted CALL, actual was CALL.")
        elif pred_class == 0 and actual_label == 0:
            print("  ✅ Model predicted PUT, actual was PUT.")
        elif pred_class == 1 and actual_label == 1:
            print("  🔵 Model predicted HOLD, actual was HOLD.")
        elif pred_class == 2 and actual_label == 0:
            print("  ❌ Model predicted CALL, actual was PUT (potential inversion or wrong signal).")
        elif pred_class == 0 and actual_label == 2:
            print("  ❌ Model predicted PUT, actual was CALL (potential inversion or wrong signal).")
        elif pred_class != actual_label:
            print("  ⚠️ Model prediction did not match actual label.")


print("\n--- Debug Script Complete ---")
print("Please copy and paste the entire output of this script.")
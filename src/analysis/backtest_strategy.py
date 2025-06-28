import os
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.models.lstm_model import AriaXaTModel
from numba import njit # For Numba optimization
from datetime import datetime

# --- Config ---
config = {
    'processed_data_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/processed_data_npy/All_Indices_XaT_enriched_ohlcv_indicators.joblib',
    'metadata_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/chunk_metadata_xaat.pkl',
    'scaler_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/scaler_xaat_minmaxscaler.pkl',
    'model_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/best_aria_xat_model.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'initial_cash': 10_000, # INR
    'nifty_lot_size': 75, # Nifty 50 lot size
    'trade_risk_pct': 0.05, # % of capital risked per trade (e.g., 5%) - Adjusted for more realistic risk
    'slippage_pct': 0.0005, # 0.05% per side (consider typical Nifty slippage)
    'stop_loss_pct': 0.005, # 0.5% stop loss
    'take_profit_pct': 0.01, # 1% take profit
    'max_trades_per_day': 10, # Max trades per day
    'confidence_threshold': 0.6,
    'batch_size': 4096, # For model inference
    # Zerodha/Indian costs (all INR) - Pass these to numba functions
    'brokerage_per_order': 20.0, # Must be float for numba
    'stt_sell_pct': 0.001, # 0.1% on sell side (on premium)
    'exchange_txn_pct': 0.0003503, # 0.03503% on premium
    'gst_pct': 0.18, # 18% on (brokerage + txn + sebi)
    'sebi_charges_per_cr': 10.0, # ₹10 per crore (must be float)
    'stamp_duty_buy_pct': 0.00003, # 0.003% on buy side
    'symbol': 'NIFTY_50',
    'classification_label_column': 'label_class' # Name of your true label column
}

# --- Numba-compatible Utility Functions ---
@njit
def _get_sebi_charge_numba(trade_value, sebi_charges_per_cr):
    return sebi_charges_per_cr * (trade_value / 1e7)

@njit
def _get_trade_costs_numba(trade_value, side, brokerage_per_order, stt_sell_pct, exchange_txn_pct, sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct):
    # side: 0 for buy, 1 for sell
    brokerage = brokerage_per_order
    stt = stt_sell_pct * trade_value if side == 1 else 0.0
    exchange = exchange_txn_pct * trade_value
    sebi = _get_sebi_charge_numba(trade_value, sebi_charges_per_cr)
    stamp = stamp_duty_buy_pct * trade_value if side == 0 else 0.0
    gst = gst_pct * (brokerage + exchange + sebi)
    total = brokerage + stt + exchange + sebi + stamp + gst
    return total

@njit
def _apply_slippage_numba(price, side, slippage_pct):
    slip = slippage_pct * np.random.random()  # Use numpy's random, which is Numba-compatible
    if side == 0:  # buy
        return price * (1 + slip)
    else:  # sell
        return price * (1 - slip)

@njit
def _get_trade_lots_numba(cash, price, nifty_lot_size, trade_risk_pct):
    if price <= 0 or cash <= 0:
        return 0

    # Max lots based on 100% of current_cash if price is positive
    max_affordable_lots_full_cash = int(cash // (price * nifty_lot_size))

    # Lots based on trade_risk_pct of current_cash
    risked_cash = cash * trade_risk_pct
    risked_lots = int(risked_cash // (price * nifty_lot_size))

    # Ensure we buy at least 1 lot, and cap by max affordable and risked amount
    # If risked_lots is 0 but max_affordable_lots_full_cash is > 0, it means risk_pct is too small for 1 lot.
    # In this case, we allow 1 lot if affordable.
    if risked_lots == 0 and max_affordable_lots_full_cash >= 1:
        lots = 1
    elif risked_lots > 0:
        lots = risked_lots
    else: # Cannot afford even 1 lot based on initial_cash or risked_cash
        lots = 0

    return lots


# --- Numba Main Backtest Loop ---
@njit
def _run_backtest_numba(
    unscaled_prices, pred_classes, confidences, actual_labels, timestamps_numeric, # seconds_since_epoch for holding
    date_ordinals, # NEW: for daily trade reset
    initial_cash, nifty_lot_size, trade_risk_pct, slippage_pct,
    stop_loss_pct, take_profit_pct, max_trades_per_day, confidence_threshold,
    brokerage_per_order, stt_sell_pct, exchange_txn_pct, sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct
):
    current_cash = initial_cash
    current_position = 0 # In lots
    current_entry_price = 0.0
    current_entry_time_numeric = 0.0 # Storing timestamps as floats for Numba

    # Trade log (list of tuples for Numba)
    # (type_code, datetime_numeric, price, lots, side_code, pnl, cash_after, holding_min, reason_code)
    # type_code: 0=BUY, 1=SELL, 2=EXIT, 3=FINAL_EXIT
    # side_code: 0=LONG, 1=SHORT
    # reason_code: 0=model_exit, 1=stop_loss, 2=take_profit, 3=final_exit
    trade_log_list = []
    equity_curve = []
    drawdown_curve = []

    max_equity = initial_cash
    trades_today = 0
    last_trade_date_ordinal = -1 # Initialize with an invalid date to ensure first day reset

    N = len(unscaled_prices)

    # Defensive: Numba-safe check for empty input
    if N == 0:
        empty_trade_log = np.empty((0, 9), dtype=np.float64)
        empty_equity = np.empty((0,), dtype=np.float64)
        empty_drawdown = np.empty((0,), dtype=np.float64)
        return empty_trade_log, empty_equity, empty_drawdown

    for i in range(N):
        # Defensive: check index bounds (Numba-safe, no print)
        if not (0 <= i < len(unscaled_prices)):
            continue
        # Check for bad price values (Numba-safe, no print)
        price = unscaled_prices[i]
        if np.isnan(price) or np.isinf(price) or price <= 0:
            continue
        pred_class = pred_classes[i]
        confidence = confidences[i]
        current_timestamp = timestamps_numeric[i] # seconds since epoch
        current_date_ordinal = date_ordinals[i] # integer date ordinal
        
        # Daily trade count reset logic
        if last_trade_date_ordinal != current_date_ordinal:
            trades_today = 0
            last_trade_date_ordinal = current_date_ordinal

        # --- Exit logic (stop-loss, take-profit, model signal) ---
        if current_position != 0:
            # Calculate P&L if we were to exit now for potential stop/profit checks
            move_percentage = (price - current_entry_price) / current_entry_price
            
            stop_hit = False
            tp_hit = False
            
            if current_position > 0: # Long position
                if move_percentage <= -stop_loss_pct:
                    stop_hit = True
                elif move_percentage >= take_profit_pct:
                    tp_hit = True
            elif current_position < 0: # Short position
                if move_percentage >= stop_loss_pct: # Price moved up for short is loss
                    stop_hit = True
                elif move_percentage <= -take_profit_pct: # Price moved down for short is profit
                    tp_hit = True

            # Model exit signal
            model_exit = False
            if confidence >= confidence_threshold: # Only consider high confidence model exits
                if pred_class == 1: # Model predicts HOLD
                    model_exit = True
                elif current_position > 0 and pred_class == 0: # Long and model predicts SELL (PUT)
                    model_exit = True
                elif current_position < 0 and pred_class == 2: # Short and model predicts BUY (CALL)
                    model_exit = True

            if stop_hit or tp_hit or model_exit:
                exit_side_code = 1 if current_position > 0 else 0 # 1=sell (for long exit), 0=buy (for short exit)
                exit_price = _apply_slippage_numba(price, exit_side_code, slippage_pct)
                
                trade_value = abs(current_position) * nifty_lot_size * exit_price
                costs = _get_trade_costs_numba(trade_value, exit_side_code,
                                               brokerage_per_order, stt_sell_pct, exchange_txn_pct,
                                               sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct)
                
                # Calculate PnL correctly for both long and short exits
                pnl = (exit_price - current_entry_price) * current_position * nifty_lot_size - costs
                
                # Update cash for closing position
                if current_position > 0: # Exiting a long position (selling)
                    current_cash += (abs(current_position) * nifty_lot_size * exit_price)
                else: # Exiting a short position (buying back)
                    current_cash -= (abs(current_position) * nifty_lot_size * exit_price) # This is a buy transaction
                current_cash -= costs # Deduct costs

                holding_period = (current_timestamp - current_entry_time_numeric) if current_entry_time_numeric > 0 else 0.0

                reason_code = 0 # Default model_exit
                if stop_hit:
                    reason_code = 1
                elif tp_hit:
                    reason_code = 2

                trade_log_list.append((
                    np.int64(2),
                    np.float64(current_timestamp),
                    np.float64(exit_price),
                    np.int64(abs(current_position)),
                    np.int64(0 if current_position > 0 else 1),
                    np.float64(pnl),
                    np.float64(current_cash),
                    np.float64(holding_period),
                    np.int64(reason_code)
                ))
                current_position = 0
                current_entry_price = 0.0
                current_entry_time_numeric = 0.0
                trades_today += 1

        # --- Entry logic ---
        if current_position == 0 and confidence >= confidence_threshold and trades_today < max_trades_per_day:
            lots = _get_trade_lots_numba(current_cash, price, nifty_lot_size, trade_risk_pct)
            
            if lots > 0: # Can afford to trade at least one lot
                if pred_class == 2: # BUY CALL (LONG)
                    entry_side_code = 0 # 0=buy
                    entry_price = _apply_slippage_numba(price, entry_side_code, slippage_pct)
                    trade_value = lots * nifty_lot_size * entry_price
                    costs = _get_trade_costs_numba(trade_value, entry_side_code,
                                                   brokerage_per_order, stt_sell_pct, exchange_txn_pct,
                                                   sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct)
                    
                    total_cost = trade_value + costs
                    
                    if current_cash >= total_cost:
                        current_cash -= total_cost
                        current_position = lots
                        current_entry_price = entry_price
                        current_entry_time_numeric = current_timestamp
                        trade_log_list.append((
                            np.int64(0),
                            np.float64(current_timestamp),
                            np.float64(entry_price),
                            np.int64(lots),
                            np.int64(0),
                            np.float64(0.0),
                            np.float64(current_cash),
                            np.float64(0.0),
                            np.int64(-1)
                        ))
                        trades_today += 1
                elif pred_class == 0: # BUY PUT (SHORT)
                    entry_side_code = 1 # 1=sell (for short entry)
                    entry_price = _apply_slippage_numba(price, entry_side_code, slippage_pct)
                    trade_value = lots * nifty_lot_size * entry_price
                    costs = _get_trade_costs_numba(trade_value, entry_side_code,
                                                   brokerage_per_order, stt_sell_pct, exchange_txn_pct,
                                                   sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct)
                    
                    # For short, initial capital is just for costs, margin assumed elsewhere.
                    # This assumes you have enough margin beyond the initial cash to cover the notional value.
                    # In a simplified backtest, we just deduct costs from current cash.
                    total_cost_for_short_entry = costs 
                    
                    if current_cash >= total_cost_for_short_entry:
                        current_cash -= total_cost_for_short_entry
                        current_position = -lots # Negative for short position
                        current_entry_price = entry_price
                        current_entry_time_numeric = current_timestamp
                        trade_log_list.append((
                            np.int64(1),
                            np.float64(current_timestamp),
                            np.float64(entry_price),
                            np.int64(lots),
                            np.int64(1),
                            np.float64(0.0),
                            np.float64(current_cash),
                            np.float64(0.0),
                            np.int64(-1)
                        ))
                        trades_today += 1

        # --- Equity curve and drawdown ---
        # Calculate current equity: cash + (unrealized P&L if position open)
        current_equity = current_cash
        if current_position != 0:
            current_equity += (price - current_entry_price) * current_position * nifty_lot_size
        equity_curve.append(current_equity)
        
        # Max equity and drawdown calculation
        if len(equity_curve) > 0:
            if current_equity > max_equity:
                max_equity = current_equity
            drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0.0
            drawdown_curve.append(drawdown)
        else:
            drawdown_curve.append(0.0) # No drawdown yet

    # Final Exit if position open at the end of backtest period
    if current_position != 0:
        price = unscaled_prices[N-1] # Last price
        current_timestamp = timestamps_numeric[N-1] # Last timestamp
        
        exit_side_code = 1 if current_position > 0 else 0 # 1=sell (for long exit), 0=buy (for short exit)
        exit_price = _apply_slippage_numba(price, exit_side_code, slippage_pct)
        trade_value = abs(current_position) * nifty_lot_size * exit_price
        costs = _get_trade_costs_numba(trade_value, exit_side_code,
                                       brokerage_per_order, stt_sell_pct, exchange_txn_pct,
                                       sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct)
        
        pnl = (exit_price - current_entry_price) * current_position * nifty_lot_size - costs
        
        # Update cash for closing position
        if current_position > 0: # Exiting a long position (selling)
            current_cash += (abs(current_position) * nifty_lot_size * exit_price)
        else: # Exiting a short position (buying back)
            current_cash -= (abs(current_position) * nifty_lot_size * exit_price)
        current_cash -= costs # Deduct costs

        holding_period = (current_timestamp - current_entry_time_numeric) if current_entry_time_numeric > 0 else 0.0

        trade_log_list.append((
            np.int64(3),
            np.float64(current_timestamp),
            np.float64(exit_price),
            np.int64(abs(current_position)),
            np.int64(0 if current_position > 0 else 1),
            np.float64(pnl),
            np.float64(current_cash),
            np.float64(holding_period),
            np.int64(3)
        ))
    
    return np.array(trade_log_list), np.array(equity_curve), np.array(drawdown_curve)


# --- Load Artifacts ---
print('Loading metadata...')
metadata = joblib.load(config['metadata_path'])
print('Metadata loaded.')
print('Loading scaler...')
scaler = joblib.load(config['scaler_path'])
print('Scaler loaded.')

# Determine features and their index for de-normalization - MOVE THIS UP
all_features_from_metadata = [f.lower() for f in metadata['features']]
# Filter out labels as features
features = [f for f in all_features_from_metadata if f not in ['label_class', 'label_regression_normalized']]
num_features = len(features) # Update num_features after filtering

# Find the index of 'close' feature in the original metadata features list
close_feature_idx = -1
for idx, f in enumerate(all_features_from_metadata):
    if f.lower() == 'close':
        close_feature_idx = idx
        break
if close_feature_idx == -1:
    raise ValueError(" 'close' feature not found in metadata's features list.")

print(f"Features used for model ({num_features}): {features[:5]}...{features[-5:]}")
print(f"'close' feature found at index {close_feature_idx} in scaler's input.")

print('Initializing model...')
model = AriaXaTModel(
    input_features=num_features, # Use updated num_features
    hidden_size=128,
    num_layers=2,
    output_classes=3,
    dropout_rate=0.2
).to(config['device'])
print('Loading model weights...')
model.load_state_dict(torch.load(config['model_path'], map_location=config['device']))
model.eval()
print('Model loaded and set to eval mode.')

print('Loading processed data...')
df_raw = joblib.load(config['processed_data_path'])
df_raw.columns = [c.lower() for c in df_raw.columns] # Ensure column names are lowercase
print('Processed data loaded.')

# --- Filter for NIFTY_50 only & Prepare for Backtest ---
print(f"Filtering data for symbol: {config['symbol']}...")
if config['symbol'] not in df_raw.index.get_level_values(0).unique():
    raise ValueError(f"Symbol '{config['symbol']}' not found in the processed data.")
df = df_raw.loc[config['symbol']].copy() # Use .copy() to avoid SettingWithCopyWarning
df = df.sort_index() # Ensure time order

# NEW: Drop rows with NaNs in relevant columns *after* symbol filter but *before* lookback slicing
# This ensures a clean DataFrame for the entire relevant period
nan_check_cols = features + [config['classification_label_column']]
initial_rows_after_symbol_filter = len(df)
df.dropna(subset=nan_check_cols, inplace=True)
dropped_rows_after_nan_filter = initial_rows_after_symbol_filter - len(df)
if dropped_rows_after_nan_filter > 0:
    print(f"WARNING: Dropped {dropped_rows_after_nan_filter} rows due to NaNs in features or label columns for {config['symbol']}.")

print("NaN counts after dropna:")
print(df[nan_check_cols].isnull().sum())

print(f"Data for {config['symbol']} loaded. Shape: {df.shape}")

# Important: Only proceed with data that can form a full lookback window
# This means the first 'lookback_window' rows cannot have predictions based on them.
# The first valid prediction aligns with the data point at index 'lookback_window'
lookback_window = metadata['lookback_window'] # Ensure this is fetched from metadata
df_backtest = df.iloc[lookback_window:].copy()
if df_backtest.empty:
    raise ValueError(f"Not enough data for backtesting after applying lookback window of {lookback_window}. Filtered df has {len(df)} rows.")

timestamps = df_backtest.index # These are datetime objects for the portion used in backtest

# Convert timestamps to Numba-compatible formats
# For holding period: seconds since epoch
seconds_since_epoch = np.array([dt.timestamp() for dt in timestamps], dtype=np.float64)

# For daily trade count reset: date ordinal (integer representing date)
date_ordinals_np = np.array([dt.toordinal() for dt in timestamps], dtype=np.int64) # Use new name and explicit dtype


# --- DE-NORMALIZE CLOSE PRICE ---
print("De-normalizing 'close' prices...")
unscaled_close_prices = np.zeros(len(df_backtest), dtype=np.float64) # Ensure float64 for Numba

for i, scaled_price in enumerate(df_backtest['close'].values):
    dummy_row_scaled = np.zeros((1, len(all_features_from_metadata)), dtype=np.float64) # Use full feature count for scaler
    dummy_row_scaled[0, close_feature_idx] = scaled_price
    
    # Ensure scaler.inverse_transform receives float64
    unscaled_value = scaler.inverse_transform(dummy_row_scaled)[0, close_feature_idx]
    unscaled_close_prices[i] = unscaled_value

# --- Pre-compute all predictions for efficiency ---
print("Pre-computing model predictions for entire dataset...")
all_preds = []
all_confs = []
all_actual_labels = []

total_batches = (len(df_backtest) + config['batch_size'] - 1) // config['batch_size']

with torch.no_grad():
    for i in tqdm(range(0, len(df_backtest), config['batch_size']), total=total_batches, desc="Inferencing"):
        batch_end = min(i + config['batch_size'], len(df_backtest))
        batch_features_df = df_backtest.iloc[i:batch_end][features]
        batch_labels = df_backtest.iloc[i:batch_end][config['classification_label_column']].values.astype(np.int64)

        nan_mask = batch_features_df.isnull().any(axis=1)
        if nan_mask.any():
            print(f"WARNING: NaNs found in features for batch {i}-{batch_end}. Skipping only those rows.")
            for j, is_nan in enumerate(nan_mask):
                if is_nan:
                    all_preds.append(1)  # HOLD
                    all_confs.append(0.0)
                    all_actual_labels.append(batch_labels[j])
                else:
                    X_row_scaled = scaler.transform(batch_features_df.iloc[[j]].values.astype(np.float64))
                    X_tensor_row = torch.tensor(X_row_scaled, dtype=torch.float32).to(config['device'])
                    out_class_logits, _ = model(X_tensor_row.unsqueeze(1))
                    probs = torch.softmax(out_class_logits, dim=1).cpu().numpy()[0]
                    pred_class = probs.argmax()
                    confidence = probs.max()
                    all_preds.append(pred_class)
                    all_confs.append(confidence)
                    all_actual_labels.append(batch_labels[j])
            continue  # Done with this batch
        # If no NaNs, process the whole batch as before
        X_batch_scaled = scaler.transform(batch_features_df.values.astype(np.float64))
        X_tensor_batch = torch.tensor(X_batch_scaled, dtype=torch.float32).to(config['device'])
        out_class_logits, _ = model(X_tensor_batch.unsqueeze(1))
        probs = torch.softmax(out_class_logits, dim=1).cpu().numpy()
        pred_class_batch = probs.argmax(axis=1)
        confidence_batch = probs.max(axis=1)
        all_preds.extend(pred_class_batch.tolist())
        all_confs.extend(confidence_batch.tolist())
        all_actual_labels.extend(batch_labels.tolist())

# Convert lists to numpy arrays for Numba
pred_classes_np = np.array(all_preds, dtype=np.int64)
confidences_np = np.array(all_confs, dtype=np.float64)
actual_labels_np = np.array(all_actual_labels, dtype=np.int64)


# FINAL LENGTH CHECK FOR ALL ARRAYS PASSED TO NUMBA
# All these arrays MUST have the same length as df_backtest
expected_len = len(df_backtest)

print("Array lengths before Numba call:")
print(f"  unscaled_close_prices: {len(unscaled_close_prices)}")
print(f"  pred_classes_np: {len(pred_classes_np)}")
print(f"  confidences_np: {len(confidences_np)}")
print(f"  actual_labels_np: {len(actual_labels_np)}")
print(f"  seconds_since_epoch: {len(seconds_since_epoch)}")
print(f"  date_ordinals_np: {len(date_ordinals_np)}")
print(f"  df_backtest: {len(df_backtest)}")

lengths = [
    len(unscaled_close_prices),
    len(pred_classes_np),
    len(confidences_np),
    len(actual_labels_np),
    len(seconds_since_epoch),
    len(date_ordinals_np),
    len(df_backtest)
]
if len(set(lengths)) != 1:
    print("ERROR: Array length mismatch detected! Aborting before Numba call.")
    import sys; sys.exit(1)

print('Starting NIFTY_50 backtest with realistic costs and risk management...')

# Before the Numba call, add a deep data sanity check:
print("==== DEEP DATA SANITY CHECK BEFORE NUMBA ====")
arrays = {
    "unscaled_close_prices": unscaled_close_prices,
    "pred_classes_np": pred_classes_np,
    "confidences_np": confidences_np,
    "actual_labels_np": actual_labels_np,
    "seconds_since_epoch": seconds_since_epoch,
    "date_ordinals_np": date_ordinals_np,
}
for name, arr in arrays.items():
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  min={np.min(arr)}, max={np.max(arr)}, any NaN={np.isnan(arr).any()}, any inf={np.isinf(arr).any()}")
    if arr.ndim != 1:
        print(f"  ERROR: {name} is not 1D!")
    if len(arr) != len(df_backtest):
        print(f"  ERROR: {name} length {len(arr)} != df_backtest length {len(df_backtest)}")
print("==== END SANITY CHECK ====")

# --- Run Numba Backtest ---
trade_log_raw, equity_curve_np, drawdown_curve_np = _run_backtest_numba(
    unscaled_close_prices, pred_classes_np, confidences_np, actual_labels_np,
    seconds_since_epoch, # Pass seconds_since_epoch for holding period
    date_ordinals_np, # Pass date_ordinals_np for daily reset
    config['initial_cash'], config['nifty_lot_size'], config['trade_risk_pct'], config['slippage_pct'],
    config['stop_loss_pct'], config['take_profit_pct'], config['max_trades_per_day'], config['confidence_threshold'],
    config['brokerage_per_order'], config['stt_sell_pct'], config['exchange_txn_pct'],
    config['sebi_charges_per_cr'], config['stamp_duty_buy_pct'], config['gst_pct']
)

# --- Post-process Numba output ---
trade_log = []
type_map = {0: 'BUY', 1: 'SELL', 2: 'EXIT', 3: 'FINAL_EXIT'}
side_map = {0: 'LONG', 1: 'SHORT'} # Represents position type
reason_map = {0: 'model_exit', 1: 'stop_loss', 2: 'take_profit', 3: 'final_exit'}

for trade_tuple in trade_log_raw:
    trade_type, dt_numeric, price, lots, pos_side_code, pnl, cash_after, holding_sec, reason_code = trade_tuple
    
    trade_log.append({
        'type': type_map.get(trade_type, 'UNKNOWN'),
        'datetime': datetime.fromtimestamp(dt_numeric) if dt_numeric > 0 else None,
        'price': price,
        'lots': lots,
        'side': side_map.get(pos_side_code, 'UNKNOWN'), # Use pos_side_code for the 'side' column
        'pnl': pnl,
        'cash_after': cash_after,
        'holding_minutes': holding_sec / 60.0, # Convert seconds to minutes for holding period
        'reason': reason_map.get(reason_code, 'N/A')
    })

trade_log_df = pd.DataFrame(trade_log)

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = f'backtest_trades_NIFTY50_{timestamp_str}.csv'
trade_log_df.to_csv(log_path, index=False)
print(f"Trade log saved to {log_path}")

# --- Performance Metrics ---
def calc_metrics(trades_df, equity_curve_array, initial_cash):
    final_cash = equity_curve_array[-1] if len(equity_curve_array) > 0 else initial_cash
    total_return = (final_cash - initial_cash) / initial_cash * 100 if initial_cash > 0 else 0

    # Ensure returns are calculated from the equity curve directly for Sharpe
    equity_series = pd.Series(equity_curve_array)
    returns = equity_series.pct_change().dropna()
    
    # Adjust for annualization. Assuming 1-minute bars.
    # Trading minutes per year for Indian market (approx 6.5 hours * 60 minutes * 252 trading days)
    annualization_factor_mins = np.sqrt(252 * 6.5 * 60)
    sharpe = returns.mean() / returns.std() * annualization_factor_mins if returns.std() > 0 else 0

    max_dd = np.max(drawdown_curve_np) * 100 # Drawdown curve is already a ratio

    # Filter for closed trades (exit or final_exit)
    closed_trades = trades_df[trades_df['type'].isin(['EXIT', 'FINAL_EXIT'])]
    wins = closed_trades[closed_trades['pnl'] > 0]
    losses = closed_trades[closed_trades['pnl'] < 0]
    
    total_closed_trades = len(wins) + len(losses)
    win_rate = len(wins) / total_closed_trades * 100 if total_closed_trades > 0 else 0
    
    avg_win = wins['pnl'].mean() if not wins.empty else 0.0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0.0
    
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else np.inf
    if losses.empty: profit_factor = np.inf # If no losses, profit factor is infinite

    avg_hold = closed_trades['holding_minutes'].mean() if 'holding_minutes' in closed_trades and not closed_trades.empty else 0.0
    med_hold = closed_trades['holding_minutes'].median() if 'holding_minutes' in closed_trades and not closed_trades.empty else 0.0

    return {
        'Total Return (%)': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_dd,
        'Win Rate (%)': win_rate,
        'Avg Win (INR)': avg_win,
        'Avg Loss (INR)': avg_loss,
        'Profit Factor': profit_factor,
        'Avg Hold (min)': avg_hold,
        'Med Hold (min)': med_hold,
        'Final Cash (INR)': final_cash,
        'Total Trades': total_closed_trades
    }

metrics = calc_metrics(trade_log_df, equity_curve_np, config['initial_cash'])
metrics_path = f'backtest_metrics_NIFTY50_{timestamp_str}.csv'
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"Metrics saved to {metrics_path}")
print(metrics)

# --- Plots ---
# Create a valid date range for plotting X-axis for equity curve
plot_dates = pd.to_datetime(seconds_since_epoch, unit='s')[:len(equity_curve_np)]

plt.figure(figsize=(14, 7))
plt.plot(plot_dates, equity_curve_np)
plt.title('Equity Curve (NIFTY50)')
plt.xlabel('Date')
plt.ylabel('Equity (INR)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'equity_curve_NIFTY50_{timestamp_str}.png')
plt.close()

plt.figure(figsize=(14, 5))
plt.plot(plot_dates, drawdown_curve_np * 100) # Convert to percentage for plotting
plt.title('Drawdown Curve (NIFTY50)')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'drawdown_NIFTY50_{timestamp_str}.png')
plt.close()

if 'pnl' in trade_log_df and not trade_log_df['pnl'].empty:
    plt.figure(figsize=(10, 6))
    trade_log_df['pnl'].hist(bins=50)
    plt.title('Trade P&L Distribution (NIFTY50)')
    plt.xlabel('P&L (INR)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'trade_pnl_hist_NIFTY50_{timestamp_str}.png')
    plt.close()

print('Backtest complete. All outputs saved.')

# --- NOTE for future: For realistic options backtesting, you must use actual option chain data (strike, expiry, premium) ---
# This script uses the underlying index/futures price. For options, you'd need to fetch OTM/ITM/ATM premiums and trade those.
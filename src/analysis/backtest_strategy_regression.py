import os
os.environ['NUMBA_CAPTURED_ERRORS'] = 'stderr'
os.environ['NUMBA_FULL_TRACEBACKS'] = '1'
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.lstm_model import AriaXaTModel
from numba import njit, types
from numba.typed import List
from datetime import datetime

# --- Config ---
config = {
    'processed_data_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/processed_data_npy/All_Indices_XaT_enriched_ohlcv_indicators.joblib',
    'metadata_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/chunk_metadata_xaat.pkl',
    'scaler_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/scaler_xaat_minmaxscaler.pkl',
    'model_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/best_aria_xat_model.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'initial_cash': 1_000_000, # INR
    'nifty_lot_size': 75,
    'trade_risk_pct': 0.05,
    'slippage_pct': 0.0005,
    'stop_loss_pct': 0.005,
    'take_profit_pct': 0.01,
    'max_trades_per_day': 10,
    'confidence_threshold': 0.5, # For classification output
    'regression_threshold': 0.001, # Minimum predicted price change to take action (0.1%)
    'batch_size': 4096,
    'brokerage_per_order': 20.0,
    'stt_sell_pct': 0.001,
    'exchange_txn_pct': 0.0003503,
    'gst_pct': 0.18,
    'sebi_charges_per_cr': 10.0,
    'stamp_duty_buy_pct': 0.00003,
    'symbol': 'NIFTY_50',
    'classification_label_column': 'label_class',
    'regression_label_column': 'label_regression_normalized'
}

# --- Numba Utility Functions ---
@njit
def _get_sebi_charge_numba(trade_value, sebi_charges_per_cr):
    return sebi_charges_per_cr * (trade_value / 1e7)

@njit
def _get_trade_costs_numba(trade_value, side, brokerage_per_order, stt_sell_pct, exchange_txn_pct, sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct):
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
    slip = slippage_pct * np.random.random()
    if side == 0:  # buy
        return price * (1 + slip)
    else:  # sell
        return price * (1 - slip)

@njit
def _get_trade_lots_numba(cash, price, nifty_lot_size, trade_risk_pct):
    if price <= 0 or cash <= 0:
        return 0
    max_affordable_lots_full_cash = int(cash // (price * nifty_lot_size))
    risked_cash = cash * trade_risk_pct
    risked_lots = int(risked_cash // (price * nifty_lot_size))
    if risked_lots == 0 and max_affordable_lots_full_cash >= 1:
        lots = 1
    elif risked_lots > 0:
        lots = risked_lots
    else:
        lots = 0
    return lots

# --- Regression-based Backtest Loop ---
@njit
def _run_regression_backtest_numba(
    unscaled_prices, pred_price_changes, pred_classes, confidences, actual_labels, timestamps_numeric,
    date_ordinals, initial_cash, nifty_lot_size, trade_risk_pct, slippage_pct,
    stop_loss_pct, take_profit_pct, max_trades_per_day, confidence_threshold, regression_threshold,
    brokerage_per_order, stt_sell_pct, exchange_txn_pct, sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct
):
    current_cash = initial_cash
    current_position = 0
    current_entry_price = 0.0
    current_entry_time_numeric = 0.0

    trade_log_list = []
    equity_curve = List.empty_list(types.float64)
    drawdown_curve = List.empty_list(types.float64)

    max_equity = initial_cash
    trades_today = 0
    last_trade_date_ordinal = -1

    N = len(unscaled_prices)

    if N == 0:
        return trade_log_list, equity_curve, drawdown_curve

    for i in range(N):
        if not (0 <= i < len(unscaled_prices)):
            continue
        price = unscaled_prices[i]
        if np.isnan(price) or np.isinf(price) or price <= 0:
            continue
        
        pred_price_change = pred_price_changes[i]
        pred_class = pred_classes[i]
        confidence = confidences[i]
        current_timestamp = timestamps_numeric[i]
        current_date_ordinal = date_ordinals[i]

        if last_trade_date_ordinal != current_date_ordinal:
            trades_today = 0
            last_trade_date_ordinal = current_date_ordinal

        # --- Exit logic ---
        if current_position != 0:
            move_percentage = (price - current_entry_price) / current_entry_price
            
            stop_hit = False
            tp_hit = False
            
            if current_position > 0:  # Long position
                if move_percentage <= -stop_loss_pct:
                    stop_hit = True
                elif move_percentage >= take_profit_pct:
                    tp_hit = True
            elif current_position < 0:  # Short position
                if move_percentage >= stop_loss_pct:
                    stop_hit = True
                elif move_percentage <= -take_profit_pct:
                    tp_hit = True

            # Regression-based exit: Exit if predicted price change is opposite to position
            regression_exit = False
            if abs(pred_price_change) >= regression_threshold:
                if current_position > 0 and pred_price_change < -regression_threshold:  # Long but predict price down
                    regression_exit = True
                elif current_position < 0 and pred_price_change > regression_threshold:  # Short but predict price up
                    regression_exit = True

            # Classification-based exit
            model_exit = False
            if confidence >= confidence_threshold:
                if pred_class == 1:  # HOLD
                    model_exit = True
                elif current_position > 0 and pred_class == 0:  # Long and predict SELL
                    model_exit = True
                elif current_position < 0 and pred_class == 2:  # Short and predict BUY
                    model_exit = True

            if stop_hit or tp_hit or model_exit or regression_exit:
                exit_side_code = 1 if current_position > 0 else 0
                exit_price = _apply_slippage_numba(price, exit_side_code, slippage_pct)
                
                trade_value = abs(current_position) * nifty_lot_size * exit_price
                costs = _get_trade_costs_numba(trade_value, exit_side_code,
                                               brokerage_per_order, stt_sell_pct, exchange_txn_pct,
                                               sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct)
                
                pnl = (exit_price - current_entry_price) * current_position * nifty_lot_size - costs
                
                if current_position > 0:
                    current_cash += (abs(current_position) * nifty_lot_size * exit_price)
                else:
                    current_cash -= (abs(current_position) * nifty_lot_size * exit_price)
                current_cash -= costs

                holding_period = (current_timestamp - current_entry_time_numeric) if current_entry_time_numeric > 0 else 0.0

                reason_code = 0  # model_exit
                if stop_hit:
                    reason_code = 1
                elif tp_hit:
                    reason_code = 2
                elif regression_exit:
                    reason_code = 7  # regression_exit

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

        # --- Entry logic based on regression predictions ---
        if current_position == 0 and trades_today < max_trades_per_day:
            should_take_signal = False
            entry_direction = 0  # 0 = no entry, 1 = long, -1 = short
            
            # Check regression signal first
            if pred_price_change > 0:  # Predict up
                entry_direction = 1
            elif pred_price_change < 0:  # Predict down  
                entry_direction = -1
            
            # If no regression signal, check classification signal
            elif confidence >= confidence_threshold:
                if pred_class == 2:  # BUY
                    entry_direction = 1
                elif pred_class == 0:  # SELL
                    entry_direction = -1
            
            if entry_direction != 0:
                lots = _get_trade_lots_numba(current_cash, price, nifty_lot_size, trade_risk_pct)
                
                if lots > 0:
                    if entry_direction == 1:  # LONG
                        entry_side_code = 0
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
                    elif entry_direction == -1:  # SHORT
                        entry_side_code = 1
                        entry_price = _apply_slippage_numba(price, entry_side_code, slippage_pct)
                        trade_value = lots * nifty_lot_size * entry_price
                        costs = _get_trade_costs_numba(trade_value, entry_side_code,
                                                       brokerage_per_order, stt_sell_pct, exchange_txn_pct,
                                                       sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct)
                        
                        total_cost_for_short_entry = costs
                        
                        if current_cash >= total_cost_for_short_entry:
                            current_cash -= total_cost_for_short_entry
                            current_position = -lots
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
        current_equity = current_cash
        if current_position != 0:
            current_equity += (price - current_entry_price) * current_position * nifty_lot_size
        equity_curve.append(current_equity)
        
        if len(equity_curve) > 0:
            if current_equity > max_equity:
                max_equity = current_equity
            drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0.0
            drawdown_curve.append(drawdown)
        else:
            drawdown_curve.append(0.0)

    # Final exit
    if current_position != 0:
        price = unscaled_prices[N-1]
        current_timestamp = timestamps_numeric[N-1]
        
        exit_side_code = 1 if current_position > 0 else 0
        exit_price = _apply_slippage_numba(price, exit_side_code, slippage_pct)
        trade_value = abs(current_position) * nifty_lot_size * exit_price
        costs = _get_trade_costs_numba(trade_value, exit_side_code,
                                       brokerage_per_order, stt_sell_pct, exchange_txn_pct,
                                       sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct)
        
        pnl = (exit_price - current_entry_price) * current_position * nifty_lot_size - costs
        
        if current_position > 0:
            current_cash += (abs(current_position) * nifty_lot_size * exit_price)
        else:
            current_cash -= (abs(current_position) * nifty_lot_size * exit_price)
        current_cash -= costs

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
    
    return trade_log_list, equity_curve, drawdown_curve

# --- Main Execution ---
if __name__ == "__main__":
    print("=== REGRESSION-BASED BACKTEST ===")
    print("Testing model's regression output for price prediction trading")
    
    # Load artifacts
    print('Loading metadata...')
    metadata = joblib.load(config['metadata_path'])
    print('Metadata loaded.')
    print('Loading scaler...')
    scaler = joblib.load(config['scaler_path'])
    print('Scaler loaded.')
    
    # Determine features
    all_features_from_metadata = [f.lower() for f in metadata['features']]
    features = [f for f in all_features_from_metadata if f not in ['label_class', 'label_regression_normalized']]
    num_features = len(features)
    
    # Find close feature index
    close_feature_idx = -1
    for idx, f in enumerate(all_features_from_metadata):
        if f.lower() == 'close':
            close_feature_idx = idx
            break
    if close_feature_idx == -1:
        raise ValueError("'close' feature not found in metadata's features list.")
    
    print(f"Features used for model ({num_features}): {features[:5]}...{features[-5:]}")
    print(f"'close' feature found at index {close_feature_idx} in scaler's input.")
    
    # Load model
    print('Initializing model...')
    model = AriaXaTModel(
        input_features=num_features,
        hidden_size=128,
        num_layers=2,
        output_classes=3,
        dropout_rate=0.2
    ).to(config['device'])
    print('Loading model weights...')
    model.load_state_dict(torch.load(config['model_path'], map_location=config['device']))
    model.eval()
    print('Model loaded and set to eval mode.')
    
    # Configuration summary
    print('\n=== REGRESSION BACKTEST CONFIGURATION ===')
    print(f"Initial Capital: ₹{config['initial_cash']:,.0f}")
    print(f"Regression Threshold: {config['regression_threshold']*100:.2f}%")
    print(f"Confidence Threshold: {config['confidence_threshold']}")
    print(f"Trade Risk: {config['trade_risk_pct']*100:.1f}%")
    print(f"Stop Loss: {config['stop_loss_pct']*100:.1f}%")
    print(f"Take Profit: {config['take_profit_pct']*100:.1f}%")
    print(f"Max Trades/Day: {config['max_trades_per_day']}")
    print("=== END CONFIGURATION ===\n")
    
    # Load data
    print('Loading processed data...')
    df_raw = joblib.load(config['processed_data_path'])
    df_raw.columns = [c.lower() for c in df_raw.columns]
    print('Processed data loaded.')
    
    # Filter for NIFTY_50
    print(f"Filtering data for symbol: {config['symbol']}...")
    if config['symbol'] not in df_raw.index.get_level_values(0).unique():
        raise ValueError(f"Symbol '{config['symbol']}' not found in the processed data.")
    df = df_raw.loc[config['symbol']].copy()
    df = df.sort_index()
    
    # Drop NaNs
    nan_check_cols = features + [config['classification_label_column'], config['regression_label_column']]
    initial_rows = len(df)
    df.dropna(subset=nan_check_cols, inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"WARNING: Dropped {dropped_rows} rows due to NaNs.")
    
    print(f"Data for {config['symbol']} loaded. Shape: {df.shape}")
    
    # Apply lookback window
    lookback_window = metadata['lookback_window']
    df_backtest = df.iloc[lookback_window:].copy()
    if df_backtest.empty:
        raise ValueError(f"Not enough data for backtesting after applying lookback window of {lookback_window}.")
    
    timestamps = df_backtest.index
    seconds_since_epoch = np.array([dt.timestamp() for dt in timestamps], dtype=np.float64)
    date_ordinals_np = np.array([dt.toordinal() for dt in timestamps], dtype=np.int64)
    
    # De-normalize close prices
    print("De-normalizing 'close' prices...")
    unscaled_close_prices = np.zeros(len(df_backtest), dtype=np.float64)
    
    for i, scaled_price in enumerate(df_backtest['close'].values):
        dummy_row_scaled = np.zeros((1, len(all_features_from_metadata)), dtype=np.float64)
        dummy_row_scaled[0, close_feature_idx] = scaled_price
        unscaled_value = scaler.inverse_transform(dummy_row_scaled)[0, close_feature_idx]
        unscaled_close_prices[i] = unscaled_value
    
    # Pre-compute predictions
    print("Pre-computing model predictions (classification + regression)...")
    all_preds = []
    all_confs = []
    all_regression_preds = []
    all_actual_labels = []
    
    total_batches = (len(df_backtest) + config['batch_size'] - 1) // config['batch_size']
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df_backtest), config['batch_size']), total=total_batches, desc="Inferencing"):
            batch_end = min(i + config['batch_size'], len(df_backtest))
            batch_features_df = df_backtest.iloc[i:batch_end][features]
            batch_labels = df_backtest.iloc[i:batch_end][config['classification_label_column']].values.astype(np.int64)
            
            # Handle NaNs
            nan_mask = batch_features_df.isnull().any(axis=1)
            if nan_mask.any():
                for j, is_nan in enumerate(nan_mask):
                    if is_nan:
                        all_preds.append(1)  # HOLD
                        all_confs.append(0.0)
                        all_regression_preds.append(0.0)
                        all_actual_labels.append(batch_labels[j])
                    else:
                        X_row_scaled = scaler.transform(batch_features_df.iloc[[j]].values.astype(np.float64))
                        X_tensor_row = torch.tensor(X_row_scaled, dtype=torch.float32).to(config['device'])
                        out_class_logits, out_regression = model(X_tensor_row.unsqueeze(1))
                        probs = torch.softmax(out_class_logits, dim=1).cpu().numpy()[0]
                        pred_class = probs.argmax()
                        confidence = probs.max()
                        regression_pred = out_regression.cpu().numpy()[0]
                        all_preds.append(pred_class)
                        all_confs.append(confidence)
                        all_regression_preds.append(regression_pred)
                        all_actual_labels.append(batch_labels[j])
                continue
            
            # Process batch
            X_batch_scaled = scaler.transform(batch_features_df.values.astype(np.float64))
            X_tensor_batch = torch.tensor(X_batch_scaled, dtype=torch.float32).to(config['device'])
            out_class_logits, out_regression = model(X_tensor_batch.unsqueeze(1))
            probs = torch.softmax(out_class_logits, dim=1).cpu().numpy()
            pred_class_batch = probs.argmax(axis=1)
            confidence_batch = probs.max(axis=1)
            regression_pred_batch = out_regression.cpu().numpy()
            
            all_preds.extend(pred_class_batch.tolist())
            all_confs.extend(confidence_batch.tolist())
            all_regression_preds.extend(regression_pred_batch.tolist())
            all_actual_labels.extend(batch_labels.tolist())
    
    # Convert to numpy arrays
    pred_classes_np = np.array(all_preds, dtype=np.int64)
    confidences_np = np.array(all_confs, dtype=np.float64)
    regression_preds_np = np.array(all_regression_preds, dtype=np.float64)
    actual_labels_np = np.array(all_actual_labels, dtype=np.int64)
    
    # Convert regression predictions to price changes (assuming they're normalized)
    # We need to de-normalize the regression predictions
    print("De-normalizing regression predictions...")
    pred_price_changes = np.zeros(len(regression_preds_np), dtype=np.float64)
    
    for i, reg_pred in enumerate(regression_preds_np):
        # Create dummy row with regression prediction
        dummy_row_scaled = np.zeros((1, len(all_features_from_metadata)), dtype=np.float64)
        # Find regression label index
        reg_label_idx = -1
        for idx, f in enumerate(all_features_from_metadata):
            if f == config['regression_label_column']:
                reg_label_idx = idx
                break
        
        if reg_label_idx != -1:
            dummy_row_scaled[0, reg_label_idx] = reg_pred
            de_normalized = scaler.inverse_transform(dummy_row_scaled)[0, reg_label_idx]
            pred_price_changes[i] = de_normalized
        else:
            # If we can't find the regression label, use the prediction as is
            pred_price_changes[i] = reg_pred
    
    # Length check
    expected_len = len(df_backtest)
    print("Array lengths before Numba call:")
    print(f"  unscaled_close_prices: {len(unscaled_close_prices)}")
    print(f"  pred_price_changes: {len(pred_price_changes)}")
    print(f"  pred_classes_np: {len(pred_classes_np)}")
    print(f"  confidences_np: {len(confidences_np)}")
    print(f"  actual_labels_np: {len(actual_labels_np)}")
    print(f"  seconds_since_epoch: {len(seconds_since_epoch)}")
    print(f"  date_ordinals_np: {len(date_ordinals_np)}")
    print(f"  df_backtest: {len(df_backtest)}")
    
    lengths = [len(unscaled_close_prices), len(pred_price_changes), len(pred_classes_np), 
               len(confidences_np), len(actual_labels_np), len(seconds_since_epoch), 
               len(date_ordinals_np), len(df_backtest)]
    if len(set(lengths)) != 1:
        print("ERROR: Array length mismatch detected! Aborting before Numba call.")
        sys.exit(1)
    
    print('Starting regression-based backtest...')
    
    # Run backtest
    trade_log_raw, equity_curve_np, drawdown_curve_np = _run_regression_backtest_numba(
        unscaled_close_prices, pred_price_changes, pred_classes_np, confidences_np, actual_labels_np,
        seconds_since_epoch, date_ordinals_np, config['initial_cash'], config['nifty_lot_size'], 
        config['trade_risk_pct'], config['slippage_pct'], config['stop_loss_pct'], config['take_profit_pct'], 
        config['max_trades_per_day'], config['confidence_threshold'], config['regression_threshold'],
        config['brokerage_per_order'], config['stt_sell_pct'], config['exchange_txn_pct'],
        config['sebi_charges_per_cr'], config['stamp_duty_buy_pct'], config['gst_pct']
    )
    
    # Convert to np.array
    trade_log_raw = np.array(trade_log_raw)
    equity_curve_np = np.array(equity_curve_np)
    drawdown_curve_np = np.array(drawdown_curve_np)
    
    # Diagnostic information
    print(f"\n=== REGRESSION DIAGNOSTIC INFORMATION ===")
    print(f"Total data points: {len(pred_classes_np)}")
    print(f"Regression predictions:")
    print(f"  Min predicted change: {np.min(pred_price_changes)*100:.3f}%")
    print(f"  Max predicted change: {np.max(pred_price_changes)*100:.3f}%")
    print(f"  Mean predicted change: {np.mean(pred_price_changes)*100:.3f}%")
    print(f"  Std predicted change: {np.std(pred_price_changes)*100:.3f}%")
    print(f"  Regression threshold: {config['regression_threshold']*100:.2f}%")
    print(f"  Signals above threshold: {np.sum(np.abs(pred_price_changes) >= config['regression_threshold'])} ({np.sum(np.abs(pred_price_changes) >= config['regression_threshold'])/len(pred_price_changes)*100:.1f}%)")
    
    print(f"\nClassification predictions:")
    print(f"  Class 0 (SELL/PUT): {np.sum(pred_classes_np == 0)} ({np.sum(pred_classes_np == 0)/len(pred_classes_np)*100:.1f}%)")
    print(f"  Class 1 (HOLD): {np.sum(pred_classes_np == 1)} ({np.sum(pred_classes_np == 1)/len(pred_classes_np)*100:.1f}%)")
    print(f"  Class 2 (BUY/CALL): {np.sum(pred_classes_np == 2)} ({np.sum(pred_classes_np == 2)/len(pred_classes_np)*100:.1f}%)")
    
    print(f"\nConfidence statistics:")
    print(f"  Min confidence: {np.min(confidences_np):.3f}")
    print(f"  Max confidence: {np.max(confidences_np):.3f}")
    print(f"  Mean confidence: {np.mean(confidences_np):.3f}")
    print(f"  Confidence threshold: {config['confidence_threshold']}")
    print(f"  Points above threshold: {np.sum(confidences_np >= config['confidence_threshold'])} ({np.sum(confidences_np >= config['confidence_threshold'])/len(confidences_np)*100:.1f}%)")
    
    print(f"\nNumber of trades made: {len(trade_log_raw)}")
    print("=== END DIAGNOSTIC ===\n")
    
    # Post-process results
    trade_log = []
    type_map = {0: 'BUY', 1: 'SELL', 2: 'EXIT', 3: 'FINAL_EXIT'}
    side_map = {0: 'LONG', 1: 'SHORT'}
    reason_map = {0: 'model_exit', 1: 'stop_loss', 2: 'take_profit', 3: 'final_exit', 7: 'regression_exit'}
    
    for trade_tuple in trade_log_raw:
        trade_type, dt_numeric, price, lots, pos_side_code, pnl, cash_after, holding_sec, reason_code = trade_tuple
        
        trade_log.append({
            'type': type_map.get(trade_type, 'UNKNOWN'),
            'datetime': datetime.fromtimestamp(dt_numeric) if dt_numeric > 0 else None,
            'price': price,
            'lots': lots,
            'side': side_map.get(pos_side_code, 'UNKNOWN'),
            'pnl': pnl,
            'cash_after': cash_after,
            'holding_minutes': holding_sec / 60.0,
            'reason': reason_map.get(reason_code, 'N/A')
        })
    
    trade_log_df = pd.DataFrame(trade_log)
    
    # Save results
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'regression_backtest_trades_NIFTY50_{timestamp_str}.csv'
    trade_log_df.to_csv(log_path, index=False)
    print(f"Trade log saved to {log_path}")
    
    # Performance metrics
    if trade_log_df.empty or 'type' not in trade_log_df.columns:
        print("WARNING: No trades were made. Skipping metrics and plots.")
        sys.exit(0)
    
    def calc_metrics(trades_df, equity_curve_array, initial_cash):
        final_cash = equity_curve_array[-1] if len(equity_curve_array) > 0 else initial_cash
        total_return = (final_cash - initial_cash) / initial_cash * 100 if initial_cash > 0 else 0
        
        equity_series = pd.Series(equity_curve_array)
        returns = equity_series.pct_change().dropna()
        annualization_factor_mins = np.sqrt(252 * 6.5 * 60)
        sharpe = returns.mean() / returns.std() * annualization_factor_mins if returns.std() > 0 else 0
        
        max_dd = np.max(drawdown_curve_np) * 100
        
        closed_trades = trades_df[trades_df['type'].isin(['EXIT', 'FINAL_EXIT'])]
        wins = closed_trades[closed_trades['pnl'] > 0]
        losses = closed_trades[closed_trades['pnl'] < 0]
        
        total_closed_trades = len(wins) + len(losses)
        win_rate = len(wins) / total_closed_trades * 100 if total_closed_trades > 0 else 0
        
        avg_win = wins['pnl'].mean() if not wins.empty else 0.0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0.0
        
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else np.inf
        if losses.empty: profit_factor = np.inf
        
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
    metrics_path = f'regression_backtest_metrics_NIFTY50_{timestamp_str}.csv'
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    print("Regression Backtest Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Plots
    if len(equity_curve_np) > 0:
        plot_dates = pd.to_datetime(seconds_since_epoch, unit='s')[:len(equity_curve_np)]
        
        plt.figure(figsize=(14, 7))
        plt.plot(plot_dates, equity_curve_np)
        plt.title('Regression-Based Equity Curve (NIFTY50)')
        plt.xlabel('Date')
        plt.ylabel('Equity (INR)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'regression_equity_curve_NIFTY50_{timestamp_str}.png')
        plt.close()
        
        plt.figure(figsize=(14, 5))
        plt.plot(plot_dates, drawdown_curve_np * 100)
        plt.title('Regression-Based Drawdown Curve (NIFTY50)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'regression_drawdown_NIFTY50_{timestamp_str}.png')
        plt.close()
    
    print('\nRegression-based backtest complete. All outputs saved.') 
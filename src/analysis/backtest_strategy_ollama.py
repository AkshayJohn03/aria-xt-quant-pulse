import os
os.environ['NUMBA_CAPTURED_ERRORS'] = 'stderr'
os.environ['NUMBA_FULL_TRACEBACKS'] = '1'
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
from numba import njit, types # For Numba optimization
from numba.typed import List # Correct import for List
from datetime import datetime
import requests
import json
import time
from typing import List as TypeList, Dict, Any

# --- Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
AVAILABLE_MODELS = [
    "qwen2.5-0.5b-q4:latest",
    "gemma3:1b",
    "qwen2.5:0.5b",
    "qwen3:0.6b",
    "HammerAI/openhermes-2.5-mistral:latest",
    "mistral:latest"
]

# --- Config ---
config = {
    'processed_data_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/processed_data_npy/All_Indices_XaT_enriched_ohlcv_indicators.joblib',
    'metadata_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/chunk_metadata_xaat.pkl',
    'scaler_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/scaler_xaat_minmaxscaler.pkl',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'initial_cash': 1_000_000, # INR - Increased for realistic Nifty trading
    'nifty_lot_size': 75, # Nifty 50 lot size
    'trade_risk_pct': 0.05, # % of capital risked per trade (e.g., 5%) - Adjusted for more realistic risk
    'slippage_pct': 0.0005, # 0.05% per side (consider typical Nifty slippage)
    'stop_loss_pct': 0.005, # 0.5% stop loss
    'take_profit_pct': 0.01, # 1% take profit
    
    # Advanced Strategy Parameters
    'max_trades_per_day': 10, # Max trades per day
    'confidence_threshold': 0.5, # Lowered from 0.6 to capture more signals
    'force_exit_on_opposite_signal': True, # Exit position if model predicts opposite direction
    'time_based_exit_hours': 4, # Exit position after N hours if no other exit signal
    'use_dynamic_position_sizing': True, # Use volatility-based position sizing
    'use_trailing_stop': True, # Enable trailing stop loss
    'trailing_stop_pct': 0.003, # 0.3% trailing stop
    'use_volatility_adjusted_stops': True, # Adjust stops based on ATR
    'atr_multiplier_sl': 2.0, # ATR multiplier for stop loss
    'atr_multiplier_tp': 4.0, # ATR multiplier for take profit
    'min_confidence_for_short': 0.7, # Higher confidence required for short trades
    'entry_only_mode': False, # If True, only take long positions (avoid shorting)
    
    'batch_size': 100, # Smaller batch size for Ollama API calls
    # Zerodha/Indian costs (all INR) - Pass these to numba functions
    'brokerage_per_order': 20.0, # Must be float for numba
    'stt_sell_pct': 0.001, # 0.1% on sell side (on premium)
    'exchange_txn_pct': 0.0003503, # 0.03503% on premium
    'gst_pct': 0.18, # 18% on (brokerage + txn + sebi)
    'sebi_charges_per_cr': 10.0, # ₹10 per crore (must be float)
    'stamp_duty_buy_pct': 0.00003, # 0.003% on buy side
    'symbol': 'NIFTY_50',
    'classification_label_column': 'label_class', # Name of your true label column
    
    # Ollama specific settings
    'ollama_timeout': 30, # Timeout for API calls
    'max_retries': 3, # Max retries for failed API calls
    'temperature': 0.1, # Low temperature for consistent predictions
    'top_p': 0.9, # Nucleus sampling parameter
    'max_tokens': 50, # Max tokens for response
}

# --- Ollama API Functions ---
def check_ollama_connection():
    """Check if Ollama is running and get available models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            print(f"Available Ollama models: {available_models}")
            return available_models
        else:
            print(f"Failed to connect to Ollama: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

def get_ollama_prediction(model_name: str, features_dict: Dict[str, float], prompt_template: str) -> tuple:
    """
    Get prediction from Ollama model
    
    Returns:
        tuple: (prediction_class, confidence_score)
        prediction_class: 0 (SELL/PUT), 1 (HOLD), 2 (BUY/CALL)
        confidence_score: float between 0 and 1
    """
    try:
        # Create feature string
        features_str = "\n".join([f"{k}: {v:.4f}" for k, v in features_dict.items()])
        
        # Create prompt
        prompt = prompt_template.format(features=features_str)
        
        # Prepare request
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config['temperature'],
                "top_p": config['top_p'],
                "num_predict": config['max_tokens']
            }
        }
        
        # Make API call
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=config['ollama_timeout']
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '').strip().lower()
            
            # Parse response
            prediction_class = 1  # Default to HOLD
            confidence_score = 0.5  # Default confidence
            
            # Extract prediction class
            if 'buy' in response_text or 'call' in response_text or 'long' in response_text:
                prediction_class = 2
            elif 'sell' in response_text or 'put' in response_text or 'short' in response_text:
                prediction_class = 0
            elif 'hold' in response_text or 'wait' in response_text or 'neutral' in response_text:
                prediction_class = 1
            
            # Extract confidence (look for numbers between 0 and 1)
            import re
            confidence_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', response_text)
            if confidence_match:
                confidence_score = float(confidence_match.group(1))
                confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp between 0 and 1
            
            return prediction_class, confidence_score
        else:
            print(f"API call failed: {response.status_code} - {response.text}")
            return 1, 0.5  # Default to HOLD with low confidence
            
    except Exception as e:
        print(f"Error getting prediction from {model_name}: {e}")
        return 1, 0.5  # Default to HOLD with low confidence

def create_prompt_template():
    """Create a prompt template for the trading model"""
    return """You are an expert quantitative trader analyzing NIFTY 50 market data. Based on the following technical indicators and market features, predict the next market direction.

Market Features:
{features}

Instructions:
1. Analyze the technical indicators above
2. Predict the market direction for the next period
3. Provide your prediction as one of: BUY, SELL, or HOLD
4. Provide a confidence score between 0.0 and 1.0

Response Format:
Prediction: [BUY/SELL/HOLD]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation]

Your analysis:"""

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

# --- Advanced Utility Functions ---
@njit
def _get_dynamic_position_size_numba(cash, price, nifty_lot_size, trade_risk_pct, atr_value, base_atr):
    """Dynamic position sizing based on volatility"""
    if price <= 0 or cash <= 0 or atr_value <= 0:
        return 0
    
    # Adjust risk based on volatility
    volatility_factor = base_atr / atr_value if atr_value > 0 else 1.0
    # Manual clip for scalar values (Numba-compatible)
    if volatility_factor < 0.5:
        volatility_factor = 0.5
    elif volatility_factor > 2.0:
        volatility_factor = 2.0
    
    # Adjust risked amount based on volatility
    adjusted_risk_pct = trade_risk_pct * volatility_factor
    risked_cash = cash * adjusted_risk_pct
    
    # Calculate lots
    max_affordable_lots = int(cash // (price * nifty_lot_size))
    risked_lots = int(risked_cash // (price * nifty_lot_size))
    
    if risked_lots == 0 and max_affordable_lots >= 1:
        lots = 1
    elif risked_lots > 0:
        lots = min(risked_lots, max_affordable_lots)
    else:
        lots = 0
    
    return lots

@njit
def _get_volatility_adjusted_stops_numba(price, atr_value, atr_multiplier_sl, atr_multiplier_tp):
    """Calculate volatility-adjusted stop loss and take profit"""
    if atr_value <= 0:
        return 0.005, 0.01  # Default values
    
    sl_pct = (atr_value * atr_multiplier_sl) / price
    tp_pct = (atr_value * atr_multiplier_tp) / price
    
    # Manual clip for scalar values (Numba-compatible)
    if sl_pct < 0.002:
        sl_pct = 0.002
    elif sl_pct > 0.02:
        sl_pct = 0.02
    
    if tp_pct < 0.005:
        tp_pct = 0.005
    elif tp_pct > 0.05:
        tp_pct = 0.05
    
    return sl_pct, tp_pct

@njit
def _update_trailing_stop_numba(current_price, entry_price, current_trailing_stop, trailing_stop_pct, position_side):
    """Update trailing stop based on current price movement"""
    if position_side == 0:  # Long position
        if current_price > entry_price:
            new_trailing_stop = current_price * (1 - trailing_stop_pct)
            return max(new_trailing_stop, current_trailing_stop)
    else:  # Short position
        if current_price < entry_price:
            new_trailing_stop = current_price * (1 + trailing_stop_pct)
            return min(new_trailing_stop, current_trailing_stop) if current_trailing_stop > 0 else new_trailing_stop
    
    return current_trailing_stop

# --- Numba Main Backtest Loop ---
@njit
def _run_backtest_numba(
    unscaled_prices, pred_classes, confidences, actual_labels, timestamps_numeric, # seconds_since_epoch for holding
    date_ordinals, # NEW: for daily trade reset
    atr_values, # ATR values for volatility-adjusted stops and position sizing
    initial_cash, nifty_lot_size, trade_risk_pct, slippage_pct,
    stop_loss_pct, take_profit_pct, max_trades_per_day, confidence_threshold,
    brokerage_per_order, stt_sell_pct, exchange_txn_pct, sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct,
    # Advanced parameters
    force_exit_on_opposite_signal, time_based_exit_hours, use_dynamic_position_sizing,
    use_trailing_stop, trailing_stop_pct, use_volatility_adjusted_stops,
    atr_multiplier_sl, atr_multiplier_tp, min_confidence_for_short, entry_only_mode
):
    current_cash = initial_cash
    current_position = 0 # In lots
    current_entry_price = 0.0
    current_entry_time_numeric = 0.0 # Storing timestamps as floats for Numba
    current_trailing_stop = 0.0 # For trailing stop functionality

    # Trade log (list of tuples for Numba)
    # (type_code, datetime_numeric, price, lots, side_code, pnl, cash_after, holding_min, reason_code)
    # type_code: 0=BUY, 1=SELL, 2=EXIT, 3=FINAL_EXIT
    # side_code: 0=LONG, 1=SHORT
    # reason_code: 0=model_exit, 1=stop_loss, 2=take_profit, 3=final_exit, 4=trailing_stop, 5=time_exit, 6=opposite_signal
    trade_log_list = []
    equity_curve = List.empty_list(types.float64)
    drawdown_curve = List.empty_list(types.float64)

    max_equity = initial_cash
    trades_today = 0
    last_trade_date_ordinal = -1 # Initialize with an invalid date to ensure first day reset
    
    # Calculate base ATR for dynamic position sizing
    base_atr = np.mean(atr_values) if len(atr_values) > 0 else 100.0

    N = len(unscaled_prices)

    # Defensive: Numba-safe check for empty input
    if N == 0:
        return trade_log_list, equity_curve, drawdown_curve

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
            # Get volatility-adjusted stops if enabled
            current_sl_pct = stop_loss_pct
            current_tp_pct = take_profit_pct
            if use_volatility_adjusted_stops and i < len(atr_values):
                current_sl_pct, current_tp_pct = _get_volatility_adjusted_stops_numba(
                    price, atr_values[i], atr_multiplier_sl, atr_multiplier_tp
                )
            
            # Update trailing stop if enabled
            if use_trailing_stop:
                current_trailing_stop = _update_trailing_stop_numba(
                    price, current_entry_price, current_trailing_stop, trailing_stop_pct, 
                    0 if current_position > 0 else 1
                )
            
            # Calculate P&L if we were to exit now for potential stop/profit checks
            move_percentage = (price - current_entry_price) / current_entry_price
            
            stop_hit = False
            tp_hit = False
            trailing_stop_hit = False
            time_exit = False
            opposite_signal_exit = False
            
            if current_position > 0: # Long position
                if move_percentage <= -current_sl_pct:
                    stop_hit = True
                elif move_percentage >= current_tp_pct:
                    tp_hit = True
                elif use_trailing_stop and current_trailing_stop > 0 and price <= current_trailing_stop:
                    trailing_stop_hit = True
            elif current_position < 0: # Short position
                if move_percentage >= current_sl_pct: # Price moved up for short is loss
                    stop_hit = True
                elif move_percentage <= -current_tp_pct: # Price moved down for short is profit
                    tp_hit = True
                elif use_trailing_stop and current_trailing_stop > 0 and price >= current_trailing_stop:
                    trailing_stop_hit = True
            
            # Time-based exit
            if time_based_exit_hours > 0:
                holding_hours = (current_timestamp - current_entry_time_numeric) / 3600.0
                if holding_hours >= time_based_exit_hours:
                    time_exit = True
            
            # Model exit signal (including opposite signal exit)
            model_exit = False
            if confidence >= confidence_threshold:
                if pred_class == 1: # Model predicts HOLD
                    model_exit = True
                elif current_position > 0 and pred_class == 0: # Long and model predicts SELL (PUT)
                    if force_exit_on_opposite_signal:
                        opposite_signal_exit = True
                    else:
                        model_exit = True
                elif current_position < 0 and pred_class == 2: # Short and model predicts BUY (CALL)
                    if force_exit_on_opposite_signal:
                        opposite_signal_exit = True
                    else:
                        model_exit = True

            if stop_hit or tp_hit or model_exit or trailing_stop_hit or time_exit or opposite_signal_exit:
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
                elif trailing_stop_hit:
                    reason_code = 4
                elif time_exit:
                    reason_code = 5
                elif opposite_signal_exit:
                    reason_code = 6

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
                current_trailing_stop = 0.0
                trades_today += 1

        # --- Entry logic ---
        if current_position == 0 and confidence >= confidence_threshold and trades_today < max_trades_per_day:
            # Check if we should take this signal
            should_take_signal = True
            
            # Higher confidence required for short trades
            if pred_class == 0 and confidence < min_confidence_for_short:
                should_take_signal = False
            
            # Entry-only mode: only take long positions
            if entry_only_mode and pred_class == 0:
                should_take_signal = False
            
            if should_take_signal:
                # Use dynamic position sizing if enabled
                if use_dynamic_position_sizing and i < len(atr_values):
                    lots = _get_dynamic_position_size_numba(
                        current_cash, price, nifty_lot_size, trade_risk_pct, 
                        atr_values[i], base_atr
                    )
                else:
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
                            current_trailing_stop = 0.0
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
                            current_trailing_stop = 0.0
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
    
    return trade_log_list, equity_curve, drawdown_curve

# --- Main Execution ---
if __name__ == "__main__":
    # Check Ollama connection
    print("Checking Ollama connection...")
    available_models = check_ollama_connection()
    
    if not available_models:
        print("No Ollama models available. Please ensure Ollama is running and models are installed.")
        sys.exit(1)
    
    # Filter to models we want to test
    models_to_test = [model for model in AVAILABLE_MODELS if model in available_models]
    
    if not models_to_test:
        print("None of the specified models are available. Available models:")
        for model in available_models:
            print(f"  - {model}")
        sys.exit(1)
    
    print(f"Testing models: {models_to_test}")
    
    # Load artifacts
    print('Loading metadata...')
    metadata = joblib.load(config['metadata_path'])
    print('Metadata loaded.')
    print('Loading scaler...')
    scaler = joblib.load(config['scaler_path'])
    print('Scaler loaded.')
    
    # Determine features and their index for de-normalization
    all_features_from_metadata = [f.lower() for f in metadata['features']]
    features = [f for f in all_features_from_metadata if f not in ['label_class', 'label_regression_normalized']]
    num_features = len(features)
    
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
    
    # Create prompt template
    prompt_template = create_prompt_template()
    
    print('Loading processed data...')
    df_raw = joblib.load(config['processed_data_path'])
    df_raw.columns = [c.lower() for c in df_raw.columns]
    print('Processed data loaded.')
    
    # Filter for NIFTY_50 only & Prepare for Backtest
    print(f"Filtering data for symbol: {config['symbol']}...")
    if config['symbol'] not in df_raw.index.get_level_values(0).unique():
        raise ValueError(f"Symbol '{config['symbol']}' not found in the processed data.")
    df = df_raw.loc[config['symbol']].copy()
    df = df.sort_index()
    
    # Drop rows with NaNs in relevant columns
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
    lookback_window = metadata['lookback_window']
    df_backtest = df.iloc[lookback_window:].copy()
    if df_backtest.empty:
        raise ValueError(f"Not enough data for backtesting after applying lookback window of {lookback_window}. Filtered df has {len(df)} rows.")
    
    timestamps = df_backtest.index
    
    # Convert timestamps to Numba-compatible formats
    seconds_since_epoch = np.array([dt.timestamp() for dt in timestamps], dtype=np.float64)
    date_ordinals_np = np.array([dt.toordinal() for dt in timestamps], dtype=np.int64)
    
    # DE-NORMALIZE CLOSE PRICE
    print("De-normalizing 'close' prices...")
    unscaled_close_prices = np.zeros(len(df_backtest), dtype=np.float64)
    
    for i, scaled_price in enumerate(df_backtest['close'].values):
        dummy_row_scaled = np.zeros((1, len(all_features_from_metadata)), dtype=np.float64)
        dummy_row_scaled[0, close_feature_idx] = scaled_price
        unscaled_value = scaler.inverse_transform(dummy_row_scaled)[0, close_feature_idx]
        unscaled_close_prices[i] = unscaled_value
    
    # Compute base_atr and atr_values for diagnostics and advanced feature analysis
    base_atr = np.mean(df_backtest['atr'].values) if 'atr' in df_backtest.columns else 100.0
    atr_values = df_backtest['atr'].values if 'atr' in df_backtest.columns else np.ones(len(df_backtest)) * 100.0
    
    # Test each model
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}")
        
        # Configuration summary for this model
        print('\n=== OLLAMA BACKTEST CONFIGURATION ===')
        print(f"Model: {model_name}")
        print(f"Initial Capital: ₹{config['initial_cash']:,.0f}")
        print(f"Confidence Threshold: {config['confidence_threshold']}")
        print(f"Min Confidence for Short: {config['min_confidence_for_short']}")
        print(f"Trade Risk: {config['trade_risk_pct']*100:.1f}%")
        print(f"Stop Loss: {config['stop_loss_pct']*100:.1f}%")
        print(f"Take Profit: {config['take_profit_pct']*100:.1f}%")
        print(f"Max Trades/Day: {config['max_trades_per_day']}")
        print(f"Dynamic Position Sizing: {'ENABLED' if config['use_dynamic_position_sizing'] else 'DISABLED'}")
        print(f"Volatility-Adjusted Stops: {'ENABLED' if config['use_volatility_adjusted_stops'] else 'DISABLED'}")
        print(f"Trailing Stop: {'ENABLED' if config['use_trailing_stop'] else 'DISABLED'}")
        print(f"Force Exit on Opposite Signal: {'ENABLED' if config['force_exit_on_opposite_signal'] else 'DISABLED'}")
        print(f"Time-Based Exit: {'ENABLED' if config['time_based_exit_hours'] > 0 else 'DISABLED'}")
        print(f"Entry-Only Mode: {'ENABLED' if config['entry_only_mode'] else 'DISABLED'}")
        print("=== END CONFIGURATION ===\n")
        
        # Pre-compute predictions for this model
        print(f"Getting predictions from {model_name}...")
        all_preds = []
        all_confs = []
        all_actual_labels = []
        
        # Use a smaller sample for testing (first 1000 data points)
        sample_size = min(1000, len(df_backtest))
        df_sample = df_backtest.iloc[:sample_size]
        
        for i in tqdm(range(len(df_sample)), desc=f"Getting predictions from {model_name}"):
            # Get features for this data point
            features_row = df_sample.iloc[i][features]
            features_dict = {feature: float(value) for feature, value in features_row.items()}
            
            # Get prediction from Ollama
            pred_class, confidence = get_ollama_prediction(model_name, features_dict, prompt_template)
            
            all_preds.append(pred_class)
            all_confs.append(confidence)
            all_actual_labels.append(df_sample.iloc[i][config['classification_label_column']])
            
            # Add small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        # Convert to numpy arrays
        pred_classes_np = np.array(all_preds, dtype=np.int64)
        confidences_np = np.array(all_confs, dtype=np.float64)
        actual_labels_np = np.array(all_actual_labels, dtype=np.int64)
        
        # Update arrays for the sample
        unscaled_close_prices_sample = unscaled_close_prices[:sample_size]
        seconds_since_epoch_sample = seconds_since_epoch[:sample_size]
        date_ordinals_np_sample = date_ordinals_np[:sample_size]
        atr_values_sample = atr_values[:sample_size]
        
        # Run backtest
        print(f"Running backtest for {model_name}...")
        trade_log_raw, equity_curve_np, drawdown_curve_np = _run_backtest_numba(
            unscaled_close_prices_sample, pred_classes_np, confidences_np, actual_labels_np,
            seconds_since_epoch_sample, date_ordinals_np_sample, atr_values_sample,
            config['initial_cash'], config['nifty_lot_size'], config['trade_risk_pct'], config['slippage_pct'],
            config['stop_loss_pct'], config['take_profit_pct'], config['max_trades_per_day'], config['confidence_threshold'],
            config['brokerage_per_order'], config['stt_sell_pct'], config['exchange_txn_pct'],
            config['sebi_charges_per_cr'], config['stamp_duty_buy_pct'], config['gst_pct'],
            config['force_exit_on_opposite_signal'], config['time_based_exit_hours'], config['use_dynamic_position_sizing'],
            config['use_trailing_stop'], config['trailing_stop_pct'], config['use_volatility_adjusted_stops'],
            config['atr_multiplier_sl'], config['atr_multiplier_tp'], config['min_confidence_for_short'], config['entry_only_mode']
        )
        
        # Convert to np.array outside the njit function
        trade_log_raw = np.array(trade_log_raw)
        equity_curve_np = np.array(equity_curve_np)
        drawdown_curve_np = np.array(drawdown_curve_np)
        
        # Diagnostic information
        print(f"\n=== DIAGNOSTIC INFORMATION FOR {model_name} ===")
        print(f"Total data points: {len(pred_classes_np)}")
        print(f"Prediction distribution:")
        print(f"  Class 0 (SELL/PUT): {np.sum(pred_classes_np == 0)} ({np.sum(pred_classes_np == 0)/len(pred_classes_np)*100:.1f}%)")
        print(f"  Class 1 (HOLD): {np.sum(pred_classes_np == 1)} ({np.sum(pred_classes_np == 1)/len(pred_classes_np)*100:.1f}%)")
        print(f"  Class 2 (BUY/CALL): {np.sum(pred_classes_np == 2)} ({np.sum(pred_classes_np == 2)/len(pred_classes_np)*100:.1f}%)")

        print(f"\nConfidence statistics:")
        print(f"  Min confidence: {np.min(confidences_np):.3f}")
        print(f"  Max confidence: {np.max(confidences_np):.3f}")
        print(f"  Mean confidence: {np.mean(confidences_np):.3f}")
        print(f"  Confidence threshold: {config['confidence_threshold']}")
        print(f"  Points above threshold: {np.sum(confidences_np >= config['confidence_threshold'])} ({np.sum(confidences_np >= config['confidence_threshold'])/len(confidences_np)*100:.1f}%)")

        # Check potential trade opportunities
        high_conf_mask = confidences_np >= config['confidence_threshold']
        high_conf_preds = pred_classes_np[high_conf_mask]
        print(f"\nHigh confidence predictions (>= {config['confidence_threshold']}):")
        print(f"  Class 0 (SELL/PUT): {np.sum(high_conf_preds == 0)}")
        print(f"  Class 1 (HOLD): {np.sum(high_conf_preds == 1)}")
        print(f"  Class 2 (BUY/CALL): {np.sum(high_conf_preds == 2)}")

        # Check if we can afford trades
        avg_price = np.mean(unscaled_close_prices_sample)
        min_price = np.min(unscaled_close_prices_sample)
        max_price = np.max(unscaled_close_prices_sample)
        lot_cost_avg = avg_price * config['nifty_lot_size']
        lot_cost_min = min_price * config['nifty_lot_size']
        lot_cost_max = max_price * config['nifty_lot_size']
        risk_amount = config['initial_cash'] * config['trade_risk_pct']

        print(f"\nTrading affordability:")
        print(f"  Initial cash: ₹{config['initial_cash']:,.0f}")
        print(f"  Risk per trade (5%): ₹{risk_amount:,.0f}")
        print(f"  Average lot cost: ₹{lot_cost_avg:,.0f}")
        print(f"  Min lot cost: ₹{lot_cost_min:,.0f}")
        print(f"  Max lot cost: ₹{lot_cost_max:,.0f}")
        print(f"  Can afford 1 lot at avg price: {'Yes' if risk_amount >= lot_cost_avg else 'No'}")
        print(f"  Can afford 1 lot at min price: {'Yes' if risk_amount >= lot_cost_min else 'No'}")

        print(f"\nNumber of trades made: {len(trade_log_raw)}")
        print("=== END DIAGNOSTIC ===\n")

        # Enhanced Trade Analysis
        if len(trade_log_raw) > 0:
            print("=== ADVANCED FEATURE ANALYSIS ===")
            
            # Define reason mapping for analysis
            reason_map_analysis = {0: 'model_exit', 1: 'stop_loss', 2: 'take_profit', 3: 'final_exit', 
                                  4: 'trailing_stop', 5: 'time_exit', 6: 'opposite_signal'}
            
            exit_reasons = [trade_tuple[8] for trade_tuple in trade_log_raw if trade_tuple[0] in [2, 3]]  # Exit trades only
            
            if exit_reasons:
                reason_counts = {}
                for reason in exit_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                print("Exit reasons breakdown:")
                for reason_code, count in reason_counts.items():
                    reason_name = reason_map_analysis.get(reason_code, f'Unknown_{reason_code}')
                    percentage = (count / len(exit_reasons)) * 100
                    print(f"  {reason_name}: {count} ({percentage:.1f}%)")
            
            # Analyze position sizing
            if config['use_dynamic_position_sizing']:
                print(f"\nDynamic position sizing: ENABLED")
                print(f"  Base ATR: {base_atr:.2f}")
                print(f"  ATR range: {np.min(atr_values_sample):.2f} - {np.max(atr_values_sample):.2f}")
            
            # Analyze volatility-adjusted stops
            if config['use_volatility_adjusted_stops']:
                print(f"\nVolatility-adjusted stops: ENABLED")
                print(f"  ATR multipliers - SL: {config['atr_multiplier_sl']}, TP: {config['atr_multiplier_tp']}")
            
            if config['use_trailing_stop']:
                print(f"\nTrailing stop: ENABLED ({config['trailing_stop_pct']*100:.1f}%)")
            
            if config['force_exit_on_opposite_signal']:
                print(f"\nForce exit on opposite signal: ENABLED")
            
            if config['time_based_exit_hours'] > 0:
                print(f"\nTime-based exit: ENABLED ({config['time_based_exit_hours']} hours)")
            
            if config['entry_only_mode']:
                print(f"\nEntry-only mode: ENABLED (no short positions)")
            
            print("=== END ADVANCED FEATURE ANALYSIS ===\n")

        # Post-process Numba output
        trade_log = []
        type_map = {0: 'BUY', 1: 'SELL', 2: 'EXIT', 3: 'FINAL_EXIT'}
        side_map = {0: 'LONG', 1: 'SHORT'} # Represents position type
        reason_map = {0: 'model_exit', 1: 'stop_loss', 2: 'take_profit', 3: 'final_exit', 
                      4: 'trailing_stop', 5: 'time_exit', 6: 'opposite_signal'}

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

        # Save results for this model
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name_clean = model_name.replace(':', '_').replace('/', '_')
        log_path = f'backtest_trades_{model_name_clean}_{timestamp_str}.csv'
        trade_log_df.to_csv(log_path, index=False)
        print(f"Trade log saved to {log_path}")

        # Performance Metrics
        def calc_metrics(trades_df, equity_curve_array, initial_cash):
            final_cash = equity_curve_array[-1] if len(equity_curve_array) > 0 else initial_cash
            total_return = (final_cash - initial_cash) / initial_cash * 100 if initial_cash > 0 else 0

            # Ensure returns are calculated from the equity curve directly for Sharpe
            equity_series = pd.Series(equity_curve_array)
            returns = equity_series.pct_change().dropna()
            
            # Adjust for annualization. Assuming 1-minute bars.
            annualization_factor_mins = np.sqrt(252 * 6.5 * 60)
            sharpe = returns.mean() / returns.std() * annualization_factor_mins if returns.std() > 0 else 0

            max_dd = np.max(drawdown_curve_np) * 100

            # Filter for closed trades (exit or final_exit)
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
                'Model': model_name,
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

        if trade_log_df.empty or 'type' not in trade_log_df.columns:
            print("WARNING: No trades were made or trade log is empty for this model.")
            metrics = {
                'Model': model_name,
                'Total Return (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Win Rate (%)': 0.0,
                'Avg Win (INR)': 0.0,
                'Avg Loss (INR)': 0.0,
                'Profit Factor': 0.0,
                'Avg Hold (min)': 0.0,
                'Med Hold (min)': 0.0,
                'Final Cash (INR)': config['initial_cash'],
                'Total Trades': 0
            }
        else:
            metrics = calc_metrics(trade_log_df, equity_curve_np, config['initial_cash'])

        results[model_name] = metrics
        
        metrics_path = f'backtest_metrics_{model_name_clean}_{timestamp_str}.csv'
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
        print(f"Results for {model_name}:")
        for key, value in metrics.items():
            if key != 'Model':
                print(f"  {key}: {value}")

        # Plots for this model
        if len(equity_curve_np) > 0:
            plot_dates = pd.to_datetime(seconds_since_epoch_sample, unit='s')[:len(equity_curve_np)]

            plt.figure(figsize=(14, 7))
            plt.plot(plot_dates, equity_curve_np)
            plt.title(f'Equity Curve ({model_name})')
            plt.xlabel('Date')
            plt.ylabel('Equity (INR)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'equity_curve_{model_name_clean}_{timestamp_str}.png')
            plt.close()

            plt.figure(figsize=(14, 5))
            plt.plot(plot_dates, drawdown_curve_np * 100)
            plt.title(f'Drawdown Curve ({model_name})')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'drawdown_{model_name_clean}_{timestamp_str}.png')
            plt.close()

    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL MODEL COMPARISON")
    print(f"{'='*80}")
    
    if results:
        comparison_df = pd.DataFrame(list(results.values()))
        comparison_df = comparison_df.sort_values('Total Return (%)', ascending=False)
        
        print("\nModel Performance Ranking (by Total Return):")
        print(comparison_df[['Model', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Total Trades']].to_string(index=False))
        
        # Save comparison
        comparison_path = f'model_comparison_{timestamp_str}.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComplete comparison saved to {comparison_path}")
        
        # Best model
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model['Model']}")
        print(f"   Total Return: {best_model['Total Return (%)']:.2f}%")
        print(f"   Sharpe Ratio: {best_model['Sharpe Ratio']:.3f}")
        print(f"   Max Drawdown: {best_model['Max Drawdown (%)']:.2f}%")
        print(f"   Win Rate: {best_model['Win Rate (%)']:.1f}%")
        print(f"   Total Trades: {best_model['Total Trades']}")
    
    print("\nBacktest complete. All outputs saved.")
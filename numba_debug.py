import numpy as np
from numba import njit
from datetime import datetime # Needed for datetime.fromtimestamp in post-processing, but not strictly in _run_backtest_numba
import os # For setting env vars, just in case
from numba import njit, types # Add types here
from numba.typed import List # Add List here

# Set Numba debug flags directly here for this test file
os.environ['NUMBA_CAPTURED_ERRORS'] = 'stderr'
os.environ['NUMBA_FULL_TRACEBACKS'] = '1'

# --- Copy ALL your @njit functions here ---
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
    slip = slippage_pct * np.random.random()
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
    if risked_lots == 0 and max_affordable_lots_full_cash >= 1:
        lots = 1
    elif risked_lots > 0:
        lots = risked_lots
    else:
        lots = 0

    return lots

@njit
def _run_backtest_numba(
    unscaled_prices, pred_classes, confidences, actual_labels, timestamps_numeric,
    date_ordinals,
    initial_cash, nifty_lot_size, trade_risk_pct, slippage_pct,
    stop_loss_pct, take_profit_pct, max_trades_per_day, confidence_threshold,
    brokerage_per_order, stt_sell_pct, exchange_txn_pct, sebi_charges_per_cr, stamp_duty_buy_pct, gst_pct
):
    current_cash = initial_cash
    current_position = 0
    current_entry_price = 0.0
    current_entry_time_numeric = 0.0

    # Use a regular Python list for trade_log_list
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
        pred_class = pred_classes[i]
        confidence = confidences[i]
        current_timestamp = timestamps_numeric[i]
        current_date_ordinal = date_ordinals[i]

        if last_trade_date_ordinal != current_date_ordinal:
            trades_today = 0
            last_trade_date_ordinal = current_date_ordinal

        if current_position != 0:
            move_percentage = (price - current_entry_price) / current_entry_price

            stop_hit = False
            tp_hit = False

            if current_position > 0:
                if move_percentage <= -stop_loss_pct:
                    stop_hit = True
                elif move_percentage >= take_profit_pct:
                    tp_hit = True
            elif current_position < 0:
                if move_percentage >= stop_loss_pct:
                    stop_hit = True
                elif move_percentage <= -take_profit_pct:
                    tp_hit = True

            model_exit = False
            if confidence >= confidence_threshold:
                if pred_class == 1:
                    model_exit = True
                elif current_position > 0 and pred_class == 0:
                    model_exit = True
                elif current_position < 0 and pred_class == 2:
                    model_exit = True

            if stop_hit or tp_hit or model_exit:
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

                reason_code = 0
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

        if current_position == 0 and confidence >= confidence_threshold and trades_today < max_trades_per_day:
            lots = _get_trade_lots_numba(current_cash, price, nifty_lot_size, trade_risk_pct)

            if lots > 0:
                if pred_class == 2: # BUY CALL (LONG)
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
                elif pred_class == 0: # BUY PUT (SHORT)
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

# --- Create very minimal dummy data to test Numba compilation ---
num_data_points = 100 # Keep it small for quick testing
dummy_prices = np.linspace(10000, 11000, num_data_points).astype(np.float64)
dummy_preds = np.random.randint(0, 3, num_data_points).astype(np.int64) # 0=SELL, 1=HOLD, 2=BUY
dummy_confs = np.random.rand(num_data_points).astype(np.float64) * 0.5 + 0.5 # 0.5 to 1.0 confidence
dummy_labels = np.random.randint(0, 3, num_data_points).astype(np.int64)

# Get current time for timestamps and date ordinals
current_time_epoch = datetime.now().timestamp()
dummy_timestamps_numeric = np.array([current_time_epoch + i*60 for i in range(num_data_points)], dtype=np.float64) # 1-minute intervals
dummy_date_ordinals = np.array([datetime.fromtimestamp(ts).toordinal() for ts in dummy_timestamps_numeric], dtype=np.int64)

# Dummy config values (make sure types match Numba expectations)
test_config = {
    'initial_cash': 10000.0,
    'nifty_lot_size': 75,
    'trade_risk_pct': 0.05,
    'slippage_pct': 0.0005,
    'stop_loss_pct': 0.005,
    'take_profit_pct': 0.01,
    'max_trades_per_day': 10,
    'confidence_threshold': 0.6,
    'brokerage_per_order': 20.0,
    'stt_sell_pct': 0.001,
    'exchange_txn_pct': 0.0003503,
    'sebi_charges_per_cr': 10.0,
    'stamp_duty_buy_pct': 0.00003,
    'gst_pct': 0.18,
}

print("Attempting to compile _run_backtest_numba...")
try:
    # Call the Numba function once with dummy data to trigger compilation
    trade_log_raw_test, equity_curve_np_test, drawdown_curve_np_test = _run_backtest_numba(
        dummy_prices, dummy_preds, dummy_confs, dummy_labels,
        dummy_timestamps_numeric, dummy_date_ordinals,
        test_config['initial_cash'], test_config['nifty_lot_size'], test_config['trade_risk_pct'], test_config['slippage_pct'],
        test_config['stop_loss_pct'], test_config['take_profit_pct'], test_config['max_trades_per_day'], test_config['confidence_threshold'],
        test_config['brokerage_per_order'], test_config['stt_sell_pct'], test_config['exchange_txn_pct'],
        test_config['sebi_charges_per_cr'], test_config['stamp_duty_buy_pct'], test_config['gst_pct']
    )
    # Convert to np.array outside the njit function
    trade_log_raw_test = np.array(trade_log_raw_test)
    equity_curve_np_test = np.array(equity_curve_np_test)
    drawdown_curve_np_test = np.array(drawdown_curve_np_test)
    print("Numba compilation SUCCESSFUL!")
    print(f"Test Trade Log Shape: {trade_log_raw_test.shape}")
    print(f"Test Equity Curve Length: {len(equity_curve_np_test)}")
except Exception as e:
    print(f"Numba compilation FAILED: {e}")
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strike: float
    option_type: str  # 'CE', 'PE'
    confidence_score: float
    stop_loss_price: float
    take_profit_price: float
    quantity: int = 1
    
@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    quantity: int
    signal_type: str
    strike: float
    option_type: str
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = 'OPEN'  # 'OPEN', 'CLOSED'
    exit_reason: Optional[str] = None  # 'SL', 'TP', 'SIGNAL', 'EOD'

class Backtester:
    def __init__(
        self,
        historical_ohlc_data: pd.DataFrame,
        historical_option_chain_data: Optional[pd.DataFrame] = None,
        initial_capital: float = 100000.0,
        max_trades_per_day: int = 3,
        risk_per_trade: float = 0.02  # 2% risk per trade
    ):
        """
        Initialize the backtester with historical data and parameters.
        
        Args:
            historical_ohlc_data (pd.DataFrame): Historical OHLC data
            historical_option_chain_data (pd.DataFrame, optional): Historical option chain data
            initial_capital (float): Initial trading capital
            max_trades_per_day (int): Maximum number of trades per day
            risk_per_trade (float): Maximum risk per trade as percentage of capital
        """
        self.ohlc_data = historical_ohlc_data.copy()
        self.option_chain_data = historical_option_chain_data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_trades_per_day = max_trades_per_day
        self.risk_per_trade = risk_per_trade
        
        self.trades: List[Trade] = []
        self.active_trades: List[Trade] = []
        self.daily_trade_count = {}
        
        # Validate and prepare data
        self._validate_data()
        self._prepare_data()
        
    def _validate_data(self) -> None:
        """Validate input data format and required columns."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in self.ohlc_data.columns for col in required_columns):
            raise ValueError(f"OHLC data must contain columns: {required_columns}")
        
        if self.option_chain_data is not None:
            required_option_columns = ['timestamp', 'strike', 'option_type', 'last_price']
            if not all(col in self.option_chain_data.columns for col in required_option_columns):
                raise ValueError(f"Option chain data must contain columns: {required_option_columns}")
    
    def _prepare_data(self) -> None:
        """Prepare and sort data for backtesting."""
        self.ohlc_data['timestamp'] = pd.to_datetime(self.ohlc_data['timestamp'])
        self.ohlc_data = self.ohlc_data.sort_values('timestamp')
        
        if self.option_chain_data is not None:
            self.option_chain_data['timestamp'] = pd.to_datetime(self.option_chain_data['timestamp'])
            self.option_chain_data = self.option_chain_data.sort_values('timestamp')
    
    def simulate_trades(self, signals: List[TradeSignal]) -> None:
        """
        Simulate trades based on input signals.
        
        Args:
            signals (List[TradeSignal]): List of trade signals to simulate
        """
        for current_time in self.ohlc_data['timestamp']:
            # Process active trades first
            self._process_active_trades(current_time)
            
            # Check for new signals
            current_signals = [s for s in signals if s.timestamp == current_time]
            
            # Process new signals if we haven't exceeded daily trade limit
            current_date = current_time.date()
            if current_date not in self.daily_trade_count:
                self.daily_trade_count[current_date] = 0
                
            if self.daily_trade_count[current_date] < self.max_trades_per_day:
                for signal in current_signals:
                    if self._can_take_trade(signal):
                        self._execute_trade(signal)
                        self.daily_trade_count[current_date] += 1
            
            # Close all trades at end of day
            if len(self.active_trades) > 0 and current_time.time().hour >= 15:  # 3 PM
                self._close_all_trades(current_time, reason='EOD')
    
    def _can_take_trade(self, signal: TradeSignal) -> bool:
        """Check if a new trade can be taken based on risk management rules."""
        risk_amount = self.current_capital * self.risk_per_trade
        potential_loss = abs(signal.stop_loss_price - signal.take_profit_price) * signal.quantity
        return potential_loss <= risk_amount
    
    def _process_active_trades(self, current_time: datetime) -> None:
        """Process active trades for stop loss and take profit conditions."""
        current_candle = self.ohlc_data[self.ohlc_data['timestamp'] == current_time].iloc[0]
        
        for trade in self.active_trades[:]:  # Copy list to avoid modification during iteration
            if trade.signal_type == 'BUY':
                # Check stop loss
                if current_candle['low'] <= trade.stop_loss:
                    self._close_trade(trade, current_time, trade.stop_loss, 'SL')
                # Check take profit
                elif current_candle['high'] >= trade.take_profit:
                    self._close_trade(trade, current_time, trade.take_profit, 'TP')
            else:  # SELL trade
                # Check stop loss
                if current_candle['high'] >= trade.stop_loss:
                    self._close_trade(trade, current_time, trade.stop_loss, 'SL')
                # Check take profit
                elif current_candle['low'] <= trade.take_profit:
                    self._close_trade(trade, current_time, trade.take_profit, 'TP')
    
    def _execute_trade(self, signal: TradeSignal) -> None:
        """Execute a new trade based on the signal."""
        current_price = self.ohlc_data[
            self.ohlc_data['timestamp'] == signal.timestamp
        ]['close'].iloc[0]
        
        trade = Trade(
            entry_time=signal.timestamp,
            entry_price=current_price,
            quantity=signal.quantity,
            signal_type=signal.signal_type,
            strike=signal.strike,
            option_type=signal.option_type,
            stop_loss=signal.stop_loss_price,
            take_profit=signal.take_profit_price
        )
        
        self.trades.append(trade)
        self.active_trades.append(trade)
        
        logger.info(f"Executed {signal.signal_type} trade at {signal.timestamp}")
    
    def _close_trade(
        self,
        trade: Trade,
        exit_time: datetime,
        exit_price: float,
        reason: str
    ) -> None:
        """Close a trade and calculate P&L."""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.status = 'CLOSED'
        trade.exit_reason = reason
        
        # Calculate P&L
        if trade.signal_type == 'BUY':
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # SELL trade
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
            
        self.current_capital += trade.pnl
        self.active_trades.remove(trade)
        
        logger.info(
            f"Closed trade: {trade.signal_type} {trade.option_type} "
            f"PnL: {trade.pnl:.2f} Reason: {reason}"
        )
    
    def _close_all_trades(self, current_time: datetime, reason: str = 'EOD') -> None:
        """Close all active trades."""
        current_price = self.ohlc_data[
            self.ohlc_data['timestamp'] == current_time
        ]['close'].iloc[0]
        
        for trade in self.active_trades[:]:
            self._close_trade(trade, current_time, current_price, reason)
    
    def generate_pnl_log(self) -> pd.DataFrame:
        """Generate a detailed P&L log of all trades."""
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'signal_type': trade.signal_type,
                'strike': trade.strike,
                'option_type': trade.option_type,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'exit_reason': trade.exit_reason
            })
        
        return pd.DataFrame(trade_data)
    
    def calculate_metrics(self) -> Dict[str, Union[float, int]]:
        """Calculate key performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'avg_profit_per_trade': 0.0,
                'avg_trade_duration': timedelta(0)
            }
        
        pnl_df = self.generate_pnl_log()
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(pnl_df[pnl_df['pnl'] > 0])
        total_pnl = pnl_df['pnl'].sum()
        
        # Win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate daily returns for Sharpe Ratio
        daily_pnl = pnl_df.groupby(pnl_df['exit_time'].dt.date)['pnl'].sum()
        daily_returns = daily_pnl / self.initial_capital
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 1 else 0
        
        # Max Drawdown
        cumulative_pnl = pnl_df['pnl'].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdowns = (cumulative_pnl - rolling_max) / self.initial_capital * 100
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0
        
        # Average trade metrics
        avg_profit = total_pnl / total_trades if total_trades > 0 else 0
        avg_duration = (
            (pnl_df['exit_time'] - pnl_df['entry_time']).mean()
            if not pnl_df.empty else timedelta(0)
        )
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_profit_per_trade': avg_profit,
            'avg_trade_duration': avg_duration
        } 
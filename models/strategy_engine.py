import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import requests
from utils.backtester import TradeSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    # AI Model Weights
    aria_lstm_weight: float = 0.3
    prophet_weight: float = 0.2
    xgboost_weight: float = 0.2
    qwen_weight: float = 0.3
    
    # Fallback Filter Parameters
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    ema_short_period: int = 9
    ema_long_period: int = 21
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    adx_period: int = 14
    adx_threshold: float = 25.0
    vwap_period: int = 14
    kalman_process_variance: float = 0.01
    kalman_measurement_variance: float = 0.1
    
    # Black-Scholes Parameters
    risk_free_rate: float = 0.05
    time_to_expiry_days: int = 30
    volatility_window: int = 20
    
    # Delta Thresholds
    min_delta: float = 0.3
    max_delta: float = 0.7
    
    # Signal Generation
    confidence_threshold: float = 0.7
    min_signal_agreement: float = 0.6
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    atr_period: int = 14

class StrategyEngine:
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize the strategy engine with configuration.
        
        Args:
            config (StrategyConfig, optional): Strategy configuration parameters
        """
        self.config = config or StrategyConfig()
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize and load all required models."""
        try:
            # Initialize Kalman Filter
            self.kalman_filter = self._create_kalman_filter()
            
            # Load other models (placeholder for now)
            # TODO: Implement model loading when models are ready
            logger.info("Model initialization complete")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _create_kalman_filter(self) -> Dict[str, float]:
        """Create a simple Kalman filter for price smoothing."""
        return {
            'process_variance': self.config.kalman_process_variance,
            'measurement_variance': self.config.kalman_measurement_variance,
            'estimate': 0.0,
            'estimation_error': 1.0
        }
    
    def _apply_kalman_filter(self, price: float) -> float:
        """Apply Kalman filter to smooth price data."""
        kf = self.kalman_filter
        
        # Prediction
        prediction = kf['estimate']
        prediction_error = kf['estimation_error'] + kf['process_variance']
        
        # Update
        kalman_gain = prediction_error / (prediction_error + kf['measurement_variance'])
        kf['estimate'] = prediction + kalman_gain * (price - prediction)
        kf['estimation_error'] = (1 - kalman_gain) * prediction_error
        
        return kf['estimate']
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators used by fallback filters."""
        df = df.copy()
        
        # SuperTrend
        atr = self._calculate_atr(df, self.config.supertrend_period)
        df['supertrend'], df['supertrend_direction'] = self._calculate_supertrend(
            df, atr, self.config.supertrend_multiplier
        )
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.config.rsi_period)
        
        # EMAs
        df['ema_short'] = df['close'].ewm(span=self.config.ema_short_period).mean()
        df['ema_long'] = df['close'].ewm(span=self.config.ema_long_period).mean()
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(
            df['close'],
            self.config.macd_fast_period,
            self.config.macd_slow_period,
            self.config.macd_signal_period
        )
        
        # ADX
        df['adx'] = self._calculate_adx(df, self.config.adx_period)
        
        # VWAP
        df['vwap'] = self._calculate_vwap(df)
        
        # Kalman Filter
        df['price_smooth'] = df['close'].apply(self._apply_kalman_filter)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_supertrend(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        multiplier: float
    ) -> tuple:
        """Calculate SuperTrend indicator."""
        hl2 = (df['high'] + df['low']) / 2
        
        # Calculate SuperTrend bands
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)
        
        # Initialize SuperTrend series
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(df)):
            if basic_upperband[i] < supertrend[i-1] or df['close'][i-1] > supertrend[i-1]:
                supertrend[i] = basic_upperband[i]
            else:
                supertrend[i] = basic_lowerband[i]
                
            if supertrend[i] > df['close'][i]:
                direction[i] = -1
            else:
                direction[i] = 1
        
        return supertrend, direction
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int,
        slow_period: int,
        signal_period: int
    ) -> tuple:
        """Calculate MACD, Signal line, and MACD histogram."""
        exp1 = prices.ewm(span=fast_period).mean()
        exp2 = prices.ewm(span=slow_period).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period).mean()
        hist = macd - signal
        
        return macd, signal, hist
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average Directional Index."""
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        
        plus_dm = plus_dm.where(
            (plus_dm > 0) & (plus_dm > minus_dm.abs()),
            0
        )
        minus_dm = minus_dm.abs().where(
            (minus_dm > 0) & (minus_dm > plus_dm),
            0
        )
        
        atr = self._calculate_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def _apply_fallback_filters(self, df: pd.DataFrame) -> pd.Series:
        """Apply all fallback technical filters and return signal strength."""
        signals = pd.Series(0.0, index=df.index)
        
        # SuperTrend signals
        signals += df['supertrend_direction']
        
        # RSI signals
        signals += np.where(df['rsi'] < self.config.rsi_oversold, 1, 0)
        signals += np.where(df['rsi'] > self.config.rsi_overbought, -1, 0)
        
        # EMA signals
        signals += np.where(df['ema_short'] > df['ema_long'], 1, -1)
        
        # MACD signals
        signals += np.where(df['macd_hist'] > 0, 1, -1)
        
        # ADX filter
        signals *= np.where(df['adx'] > self.config.adx_threshold, 1, 0.5)
        
        # VWAP signals
        signals += np.where(df['close'] > df['vwap'], 0.5, -0.5)
        
        # Normalize signals to [-1, 1] range
        signals = signals / 6  # Divide by number of indicators
        
        return signals
    
    def _calculate_stop_loss_take_profit(
        self,
        df: pd.DataFrame,
        current_price: float,
        signal_type: str
    ) -> tuple:
        """Calculate stop loss and take profit levels based on ATR."""
        atr = self._calculate_atr(df, self.config.atr_period).iloc[-1]
        
        if signal_type == 'BUY':
            stop_loss = current_price - (atr * self.config.stop_loss_atr_multiplier)
            take_profit = current_price + (atr * self.config.take_profit_atr_multiplier)
        else:  # SELL
            stop_loss = current_price + (atr * self.config.stop_loss_atr_multiplier)
            take_profit = current_price - (atr * self.config.take_profit_atr_multiplier)
        
        return stop_loss, take_profit
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        strategy_profile: str = 'hybrid',
        mode: str = 'safe'
    ) -> List[TradeSignal]:
        """
        Generate trading signals based on all available models and filters.
        
        Args:
            df (pd.DataFrame): OHLC data with required columns
            strategy_profile (str): 'ai-only', 'fallback-only', or 'hybrid'
            mode (str): 'safe' or 'aggressive' trading mode
            
        Returns:
            List[TradeSignal]: List of generated trade signals
        """
        signals = []
        
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Get fallback filter signals
            fallback_signals = self._apply_fallback_filters(df)
            
            # TODO: Get AI model predictions when implemented
            # For now, use only fallback signals
            for i in range(len(df)):
                current_row = df.iloc[i]
                signal_strength = fallback_signals.iloc[i]
                
                # Skip if signal strength is too weak
                if abs(signal_strength) < self.config.confidence_threshold:
                    continue
                
                # Determine signal type
                signal_type = 'BUY' if signal_strength > 0 else 'SELL'
                
                # Calculate stop loss and take profit
                stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                    df.iloc[:i+1],
                    current_row['close'],
                    signal_type
                )
                
                # Create trade signal
                signal = TradeSignal(
                    timestamp=current_row.name,
                    signal_type=signal_type,
                    strike=current_row['close'],  # Use actual strike price in live trading
                    option_type='CE' if signal_type == 'BUY' else 'PE',
                    confidence_score=abs(signal_strength),
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit
                )
                
                signals.append(signal)
            
            logger.info(f"Generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise 
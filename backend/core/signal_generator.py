# D:\aria\aria-xt-quant-pulse\backend\core\signal_generator.py

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignalGenerator:
    def __init__(self, config: Dict[str, Any], model_interface, risk_manager):
        self.config = config
        self.strategy_params = config.get("trading", {})
        self.model_interface = model_interface
        self.risk_manager = risk_manager
        
        # Advanced strategy parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.bb_squeeze_threshold = 0.02
        self.volume_spike_multiplier = 1.5
        self.min_confluence_score = 0.65
        self.trend_strength_threshold = 0.6
        
        # Advanced strategy specific parameters
        self.iv_percentile_threshold = 70  # For IV-based strategies
        self.delta_range = (0.3, 0.7)  # For option selection
        self.min_oi_threshold = 50000  # Minimum open interest
        self.min_volume_threshold = 10000  # Minimum volume
        self.theta_decay_threshold = -0.1  # Maximum acceptable theta decay
        
        # Strategy weights
        self.strategy_weights = {
            'technical': 0.3,
            'ai_prediction': 0.3,
            'market_context': 0.2,
            'option_metrics': 0.2
        }
        
        logging.info("SignalGenerator initialized with Advanced Multi-Strategy Framework.")

    async def generate_signals(self, market_data: Dict[str, Any], option_chain: List[Dict]) -> List[Dict[str, Any]]:
        """
        Main signal generation method using Advanced Multi-Strategy Framework
        """
        signals = []
        
        try:
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv_5min', [])
            if len(ohlcv_data) < 50:
                logging.warning("Insufficient OHLCV data for signal generation")
                return []
            
            # Convert to DataFrame for analysis
            df = self._prepare_dataframe(ohlcv_data)
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(df)
            
            # Get AI model predictions
            ai_predictions = await self._get_ai_predictions(df)
            
            # Analyze market context
            market_context = self._analyze_market_context(df, indicators)
            
            # Generate confluence score with advanced weighting
            confluence_analysis = self._calculate_confluence_score(
                indicators, ai_predictions, market_context
            )
            
            # Apply advanced filtering
            if self._validate_signal_conditions(confluence_analysis, market_context):
                signal = self._create_trading_signal(
                    confluence_analysis, market_data, option_chain
                )
                if signal:
                    signals.append(signal)
            
            logging.info(f"Generated {len(signals)} signals with confluence score: {confluence_analysis['score']:.3f}")
            
        except Exception as e:
            logging.error(f"Error in signal generation: {str(e)}")
        
        return signals

    def set_strategy_params(self, **kwargs):
        """Dynamically update strategy parameters (for live tuning or optimization)."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
                logging.info(f"Strategy parameter '{k}' updated to {v}")

    async def backtest_signals(self, historical_data: List[Dict], option_chain: List[Dict]) -> List[Dict[str, Any]]:
        """Simulate signal generation on historical data for validation and optimization."""
        results = []
        for i in range(50, len(historical_data)):
            window_data = historical_data[:i]
            market_data = {'ohlcv_5min': window_data}
            signals = await self.generate_signals(market_data, option_chain)
            if signals:
                results.extend(signals)
        logging.info(f"Backtest generated {len(results)} signals.")
        return results

    def _prepare_dataframe(self, ohlcv_data: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(ohlcv_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.set_index('timestamp', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df.sort_index()

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        indicators = {}
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss.replace(0, np.nan))
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            indicators['macd'] = ema12 - ema26
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            indicators['bb_upper'] = sma20 + (std20 * 2)
            indicators['bb_lower'] = sma20 - (std20 * 2)
            indicators['bb_middle'] = sma20
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr'] = true_range.rolling(window=14).mean()
            
            # EMAs
            indicators['ema9'] = df['close'].ewm(span=9).mean()
            indicators['ema21'] = df['close'].ewm(span=21).mean()
            indicators['ema50'] = df['close'].ewm(span=50).mean()
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean()
            indicators['volume_spike'] = df['volume'] / indicators['volume_sma']
            
            # VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            indicators['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
        return indicators

    async def _get_ai_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get predictions from all AI models"""
        predictions = {}
        
        try:
            # Aria LSTM - Trend prediction
            lstm_input = {
                'ohlcv': df.tail(60).to_dict('records'),
                'timeframe': '5min'
            }
            predictions['aria_lstm'] = await self.model_interface.predict_trend(lstm_input)
            
            # Prophet - Pattern recognition
            prophet_input = {
                'historical_data': df.tail(100)[['close']].to_dict('records'),
                'forecast_periods': 5
            }
            predictions['prophet'] = await self.model_interface.forecast_pattern(prophet_input)
            
            # XGBoost - Volatility prediction
            xgb_input = {
                'features': self._prepare_xgb_features(df.tail(30))
            }
            predictions['xgboost'] = await self.model_interface.predict_volatility(xgb_input)
            
            # FinBERT - Sentiment (if news available)
            predictions['finbert'] = await self.model_interface.analyze_sentiment({
                'symbol': 'NIFTY50',
                'timeframe': '1H'
            })
            
            # Gemini - Market context
            gemini_prompt = self._create_gemini_prompt(df.tail(10))
            predictions['gemini'] = await self.model_interface.query_gemini(gemini_prompt)
            
        except Exception as e:
            logging.error(f"Error getting AI predictions: {str(e)}")
            predictions = self._get_default_predictions()
        
        return predictions

    def _prepare_xgb_features(self, df: pd.DataFrame) -> List[float]:
        """Prepare features for XGBoost model"""
        features = []
        
        # Price features
        features.extend([
            df['close'].iloc[-1],
            df['close'].pct_change().iloc[-1],
            df['high'].iloc[-1] - df['low'].iloc[-1],  # Daily range
            (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]  # Intraday return
        ])
        
        # Volume features
        features.extend([
            df['volume'].iloc[-1],
            df['volume'].iloc[-1] / df['volume'].mean(),  # Volume ratio
        ])
        
        # Volatility features
        returns = df['close'].pct_change().dropna()
        features.extend([
            returns.std(),  # Historical volatility
            returns.skew(),  # Skewness
            returns.kurt()   # Kurtosis
        ])
        
        return features

    def _create_gemini_prompt(self, recent_df: pd.DataFrame) -> str:
        """Create prompt for Gemini analysis"""
        latest_price = recent_df['close'].iloc[-1]
        price_change = ((recent_df['close'].iloc[-1] / recent_df['close'].iloc[-2]) - 1) * 100
        
        return f"""
        Analyze the following NIFTY50 market data:
        Current Price: {latest_price:.2f}
        Recent Change: {price_change:.2f}%
        
        Recent price action: {recent_df['close'].tail(5).tolist()}
        
        Provide a brief analysis (max 100 words) on:
        1. Market sentiment (Bullish/Bearish/Neutral)
        2. Key support/resistance levels
        3. Probability of trend continuation
        
        Format: SENTIMENT|PROBABILITY|REASONING
        """

    def _analyze_market_context(self, df: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Analyze current market context"""
        latest_idx = -1
        context = {}
        
        # Trend analysis
        ema9_vs_21 = indicators['ema9'].iloc[latest_idx] > indicators['ema21'].iloc[latest_idx]
        ema21_vs_50 = indicators['ema21'].iloc[latest_idx] > indicators['ema50'].iloc[latest_idx]
        context['trend_bullish'] = ema9_vs_21 and ema21_vs_50
        context['trend_strength'] = abs(indicators['macd'].iloc[latest_idx]) / indicators['atr'].iloc[latest_idx]
        
        # Momentum analysis
        context['rsi_oversold'] = indicators['rsi'].iloc[latest_idx] < self.rsi_oversold
        context['rsi_overbought'] = indicators['rsi'].iloc[latest_idx] > self.rsi_overbought
        context['macd_bullish'] = indicators['macd'].iloc[latest_idx] > indicators['macd_signal'].iloc[latest_idx]
        
        # Volatility analysis
        context['bb_squeeze'] = indicators['bb_width'].iloc[latest_idx] < self.bb_squeeze_threshold
        context['high_volatility'] = indicators['atr'].iloc[latest_idx] > indicators['atr'].rolling(20).mean().iloc[latest_idx]
        
        # Volume analysis
        context['volume_spike'] = indicators['volume_spike'].iloc[latest_idx] > self.volume_spike_multiplier
        context['price_vs_vwap'] = df['close'].iloc[latest_idx] > indicators['vwap'].iloc[latest_idx]
        
        return context

    def _calculate_confluence_score(self, indicators: Dict, ai_predictions: Dict, market_context: Dict) -> Dict[str, Any]:
        """Calculate confluence score for signal strength"""
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Technical signals
        if market_context.get('trend_bullish', False):
            bullish_signals += 2
        else:
            bearish_signals += 2
        total_signals += 2
        
        if market_context.get('macd_bullish', False):
            bullish_signals += 1
        else:
            bearish_signals += 1
        total_signals += 1
        
        if market_context.get('rsi_oversold', False):
            bullish_signals += 1
        elif market_context.get('rsi_overbought', False):
            bearish_signals += 1
        total_signals += 1
        
        if market_context.get('price_vs_vwap', False):
            bullish_signals += 1
        else:
            bearish_signals += 1
        total_signals += 1
        
        # AI model signals
        aria_prediction = ai_predictions.get('aria_lstm', {})
        if aria_prediction.get('direction') == 'BULLISH':
            bullish_signals += 2
        elif aria_prediction.get('direction') == 'BEARISH':
            bearish_signals += 2
        total_signals += 2
        
        prophet_prediction = ai_predictions.get('prophet', {})
        if prophet_prediction.get('trend') == 'UP':
            bullish_signals += 1
        elif prophet_prediction.get('trend') == 'DOWN':
            bearish_signals += 1
        total_signals += 1
        
        # Calculate final scores
        bullish_score = bullish_signals / total_signals if total_signals > 0 else 0
        bearish_score = bearish_signals / total_signals if total_signals > 0 else 0
        
        direction = 'BULLISH' if bullish_score > bearish_score else 'BEARISH'
        confidence = max(bullish_score, bearish_score)
        
        return {
            'direction': direction,
            'score': confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'total_signals': total_signals,
            'market_context': market_context,
            'ai_predictions': ai_predictions
        }

    def _create_trading_signal(self, confluence: Dict, market_data: Dict, option_chain: List) -> Optional[Dict[str, Any]]:
        """Create final trading signal"""
        try:
            direction = confluence['direction']
            confidence = confluence['score']
            
            # Find suitable option
            suitable_option = self._find_optimal_option(option_chain, direction, confidence)
            if not suitable_option:
                return None
            
            # Calculate position sizing
            position_size = self._calculate_position_size(suitable_option, confidence)
            
            # Calculate stop loss and target
            stop_loss, target = self._calculate_sl_target(suitable_option, direction, confluence)
            
            signal = {
                'type': 'BUY' if direction == 'BULLISH' else 'SELL',
                'symbol': suitable_option['symbol'],
                'price': suitable_option['ltp'],
                'quantity': position_size,
                'stop_loss': stop_loss,
                'target': target,
                'confidence': confidence * 100,
                'reasoning': self._generate_reasoning(confluence),
                'strategy': 'Advanced Multi-Strategy',
                'timestamp': datetime.now().isoformat(),
                'option_type': suitable_option['type'],
                'strike': suitable_option['strike'],
                'expiry': suitable_option['expiry']
            }
            
            return signal
            
        except Exception as e:
            logging.error(f"Error creating trading signal: {str(e)}")
            return None

    def _find_optimal_option(self, option_chain: List, direction: str, confidence: float) -> Optional[Dict]:
        """Enhanced option selection with advanced metrics"""
        if not option_chain:
            return None
        
        is_call = direction == 'BULLISH'
        
        # Filter and score options
        suitable_options = []
        for option in option_chain:
            option_data = option.get('call' if is_call else 'put', {})
            
            # Advanced filtering
            if not self._validate_option_metrics(option_data):
                continue
            
            # Calculate option score
            score = self._calculate_option_score(option_data, direction, confidence)
            
            suitable_options.append({
                'symbol': f"NIFTY {option['strike']} {'CE' if is_call else 'PE'}",
                'strike': option['strike'],
                'ltp': option_data.get('ltp', 0),
                'iv': option_data.get('iv', 0),
                'volume': option_data.get('volume', 0),
                'oi': option_data.get('oi', 0),
                'delta': abs(option_data.get('delta', 0)),
                'theta': option_data.get('theta', 0),
                'vega': option_data.get('vega', 0),
                'type': 'CE' if is_call else 'PE',
                'expiry': option.get('expiry', ''),
                'score': score
            })
        
        if not suitable_options:
            return None
        
        # Select best option based on score
        return max(suitable_options, key=lambda x: x['score'])

    def _validate_option_metrics(self, option_data: Dict) -> bool:
        """Validate option metrics against thresholds"""
        try:
            # Check liquidity
            if option_data.get('volume', 0) < self.min_volume_threshold:
                return False
            if option_data.get('oi', 0) < self.min_oi_threshold:
                return False
            
            # Check IV
            iv = option_data.get('iv', 0)
            if not (10 <= iv <= 40):
                return False
            
            # Check theta decay
            if option_data.get('theta', 0) < self.theta_decay_threshold:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error in option validation: {e}")
            return False

    def _calculate_option_score(self, option_data: Dict, direction: str, confidence: float) -> float:
        """Calculate comprehensive option score"""
        try:
            score = 0.0
            
            # Delta score (0.3-0.7 range preferred)
            delta = abs(option_data.get('delta', 0))
            delta_score = 1 - abs(delta - 0.5) * 2
            score += delta_score * 0.3
            
            # Liquidity score
            volume_score = min(1, option_data.get('volume', 0) / (self.min_volume_threshold * 2))
            oi_score = min(1, option_data.get('oi', 0) / (self.min_oi_threshold * 2))
            score += (volume_score + oi_score) * 0.2
            
            # IV score (lower IV preferred)
            iv = option_data.get('iv', 0)
            iv_score = 1 - (iv - 10) / 30  # Normalize IV between 10-40
            score += iv_score * 0.2
            
            # Theta score (less negative theta preferred)
            theta = option_data.get('theta', 0)
            theta_score = 1 - abs(theta) / 0.5  # Normalize theta
            score += theta_score * 0.2
            
            # Vega score (lower vega preferred for shorter-term trades)
            vega = option_data.get('vega', 0)
            vega_score = 1 - min(1, vega / 0.5)
            score += vega_score * 0.1
            
            return score
            
        except Exception as e:
            logging.error(f"Error calculating option score: {e}")
            return 0.0

    def _calculate_position_size(self, option: Dict, confidence: float) -> int:
        """Enhanced position sizing with risk management"""
        try:
            capital = self.strategy_params.get('capital_allocation', 100000)
            max_risk_per_trade = self.strategy_params.get('max_risk_per_trade', 2.5) / 100
            
            # Adjust risk based on confidence and option metrics
            base_risk = max_risk_per_trade * (0.5 + confidence * 0.5)
            
            # Adjust for IV
            iv = option.get('iv', 20) / 100
            iv_adjustment = 1 - (iv - 0.2) * 2  # Reduce size for high IV
            adjusted_risk = base_risk * max(0.5, iv_adjustment)
            
            # Calculate max affordable quantity
            max_risk_amount = capital * adjusted_risk
            option_price = option['ltp']
            
            if option_price <= 0:
                return 0
            
            max_quantity = int(max_risk_amount / option_price)
            
            # Minimum viable quantity
            min_quantity = max(1, int(10000 / option_price))
            
            # Apply position limits
            return max(min_quantity, min(max_quantity, 1000))
            
        except Exception as e:
            logging.error(f"Error in position sizing: {e}")
            return 0

    def _calculate_sl_target(self, option: Dict, direction: str, confluence: Dict) -> Tuple[float, float]:
        """Enhanced stop loss and target calculation"""
        try:
            entry_price = option['ltp']
            confidence = confluence['score']
            
            # Dynamic stop loss based on IV, ATR, and confidence
            iv = option.get('iv', 20) / 100
            atr = confluence.get('market_context', {}).get('atr', 0)
            
            # Base stop loss percentage
            base_sl_pct = 0.2 + (iv * 0.5) - (confidence * 0.1)
            
            # Adjust for volatility
            if atr > 0:
                atr_adjustment = min(0.3, atr / entry_price)
                base_sl_pct = max(base_sl_pct, atr_adjustment)
            
            # Final stop loss percentage
            stop_loss_pct = max(0.15, min(0.7, base_sl_pct))
            
            # Dynamic risk-reward ratio based on confidence and market conditions
            base_rr = 1.5 + (confidence * 1.0)
            market_volatility = confluence.get('market_context', {}).get('volatility_score', 0.5)
            adjusted_rr = base_rr * (1 + market_volatility)
            
            # Calculate target
            target_pct = stop_loss_pct * adjusted_rr
            
            stop_loss = entry_price * (1 - stop_loss_pct)
            target = entry_price * (1 + target_pct)
            
            return round(stop_loss, 2), round(target, 2)
            
        except Exception as e:
            logging.error(f"Error calculating SL/Target: {e}")
            return entry_price * 0.8, entry_price * 1.5  # Default values

    def _generate_reasoning(self, confluence: Dict) -> str:
        direction = confluence['direction']
        score = confluence['score']
        signals = confluence['bullish_signals'] if direction == 'BULLISH' else confluence['bearish_signals']
        total = confluence['total_signals']
        context = confluence['market_context']
        ai_preds = confluence['ai_predictions']
        reasoning = [
            f"{direction} signal with {score:.1%} confidence ({signals}/{total} indicators aligned)."
        ]
        if context.get('trend_bullish'):
            reasoning.append("Bullish trend confirmed.")
        if context.get('volume_spike'):
            reasoning.append("Volume spike detected.")
        if context.get('bb_squeeze'):
            reasoning.append("Bollinger Band squeeze suggests breakout.")
        if context.get('high_volatility'):
            reasoning.append("High volatility regime.")
        if ai_preds.get('aria_lstm', {}).get('confidence', 0) > 0.7:
            reasoning.append("Strong AI model conviction.")
        if ai_preds.get('finbert', {}).get('sentiment', '').upper() == 'BULLISH':
            reasoning.append("Positive news sentiment.")
        if ai_preds.get('gemini', {}).get('analysis', '').startswith('BULLISH'):
            reasoning.append("Gemini LLM bullish context.")
        return ' '.join(reasoning)

    def _get_default_predictions(self) -> Dict[str, Any]:
        """Default predictions when AI models fail"""
        return {
            'aria_lstm': {'direction': 'NEUTRAL', 'confidence': 0.5},
            'prophet': {'trend': 'SIDEWAYS', 'confidence': 0.5},
            'xgboost': {'volatility': 'MEDIUM', 'score': 0.5},
            'finbert': {'sentiment': 'NEUTRAL', 'score': 0.5},
            'gemini': {'analysis': 'NEUTRAL|0.5|Default analysis'}
        }

    def _validate_signal_conditions(self, confluence: Dict, market_context: Dict) -> bool:
        """Advanced validation of signal conditions"""
        try:
            # Check trend strength
            if market_context.get('trend_strength', 0) < self.trend_strength_threshold:
                return False
            
            # Check volatility conditions
            if market_context.get('bb_squeeze', False) and not market_context.get('volume_spike', False):
                return False
            
            # Check market regime
            if market_context.get('high_volatility', False) and confluence['score'] < 0.75:
                return False
            
            # Check time-based conditions
            current_hour = datetime.now().hour
            if current_hour < 9 or current_hour > 15:  # Outside market hours
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error in signal validation: {e}")
            return False

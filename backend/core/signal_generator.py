
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
        
        # Strategy parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.bb_squeeze_threshold = 0.02
        self.volume_spike_multiplier = 1.5
        self.min_confluence_score = 0.65
        self.trend_strength_threshold = 0.6
        
        logging.info("SignalGenerator initialized with Multi-Model Confluence Strategy.")

    async def generate_signals(self, market_data: Dict[str, Any], option_chain: List[Dict]) -> List[Dict[str, Any]]:
        """
        Main signal generation method using Multi-Model Confluence Strategy
        """
        signals = []
        
        try:
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv_5min', [])
            if len(ohlcv_data) < 50:  # Need sufficient history
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
            
            # Generate confluence score
            confluence_analysis = self._calculate_confluence_score(
                indicators, ai_predictions, market_context
            )
            
            # Generate signals based on confluence
            if confluence_analysis['score'] >= self.min_confluence_score:
                signal = self._create_trading_signal(
                    confluence_analysis, market_data, option_chain
                )
                if signal:
                    signals.append(signal)
            
            logging.info(f"Generated {len(signals)} signals with confluence score: {confluence_analysis['score']:.3f}")
            
        except Exception as e:
            logging.error(f"Error in signal generation: {str(e)}")
        
        return signals

    def _prepare_dataframe(self, ohlcv_data: List[Dict]) -> pd.DataFrame:
        """Convert OHLCV data to pandas DataFrame"""
        df = pd.DataFrame(ohlcv_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df.sort_index()

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all required technical indicators"""
        indicators = {}
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
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
                'strategy': 'Multi-Model Confluence',
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
        """Find the most suitable option based on strategy"""
        if not option_chain:
            return None
        
        is_call = direction == 'BULLISH'
        
        # Filter options based on liquidity and moneyness
        suitable_options = []
        for option in option_chain:
            option_data = option.get('call' if is_call else 'put', {})
            
            # Check liquidity
            if option_data.get('volume', 0) < 10000 or option_data.get('oi', 0) < 50000:
                continue
            
            # Check reasonable IV
            if not (10 <= option_data.get('iv', 0) <= 40):
                continue
            
            suitable_options.append({
                'symbol': f"NIFTY {option['strike']} {'CE' if is_call else 'PE'}",
                'strike': option['strike'],
                'ltp': option_data.get('ltp', 0),
                'iv': option_data.get('iv', 0),
                'volume': option_data.get('volume', 0),
                'oi': option_data.get('oi', 0),
                'delta': abs(option_data.get('delta', 0)),
                'type': 'CE' if is_call else 'PE',
                'expiry': option.get('expiry', '')
            })
        
        if not suitable_options:
            return None
        
        # Select based on delta and liquidity
        target_delta = 0.3 + (confidence * 0.4)  # Higher confidence -> higher delta
        best_option = min(suitable_options, 
                         key=lambda x: abs(x['delta'] - target_delta) + (1 / (x['volume'] + 1)))
        
        return best_option

    def _calculate_position_size(self, option: Dict, confidence: float) -> int:
        """Calculate optimal position size"""
        capital = self.strategy_params.get('capital_allocation', 100000)
        max_risk_per_trade = self.strategy_params.get('max_risk_per_trade', 2.5) / 100
        
        # Adjust risk based on confidence
        adjusted_risk = max_risk_per_trade * (0.5 + confidence * 0.5)
        
        # Calculate max affordable quantity
        max_risk_amount = capital * adjusted_risk
        option_price = option['ltp']
        
        if option_price <= 0:
            return 0
        
        max_quantity = int(max_risk_amount / option_price)
        
        # Minimum viable quantity
        min_quantity = max(1, int(10000 / option_price))
        
        return max(min_quantity, min(max_quantity, 1000))  # Cap at 1000 lots

    def _calculate_sl_target(self, option: Dict, direction: str, confluence: Dict) -> Tuple[float, float]:
        """Calculate stop loss and target prices"""
        entry_price = option['ltp']
        confidence = confluence['score']
        
        # Dynamic stop loss based on IV and confidence
        iv = option.get('iv', 20) / 100
        stop_loss_pct = 0.2 + (iv * 0.5) - (confidence * 0.1)  # 20-70% range
        stop_loss_pct = max(0.15, min(0.7, stop_loss_pct))
        
        # Target based on risk-reward ratio
        risk_reward_ratio = 1.5 + (confidence * 1.0)  # 1.5:1 to 2.5:1
        target_pct = stop_loss_pct * risk_reward_ratio
        
        stop_loss = entry_price * (1 - stop_loss_pct)
        target = entry_price * (1 + target_pct)
        
        return round(stop_loss, 2), round(target, 2)

    def _generate_reasoning(self, confluence: Dict) -> str:
        """Generate human-readable reasoning for the signal"""
        direction = confluence['direction']
        score = confluence['score']
        signals = confluence['bullish_signals'] if direction == 'BULLISH' else confluence['bearish_signals']
        total = confluence['total_signals']
        
        reasoning = f"{direction} signal with {score:.1%} confidence ({signals}/{total} indicators aligned). "
        
        context = confluence['market_context']
        if context.get('trend_bullish'):
            reasoning += "Bullish trend confirmed. "
        if context.get('volume_spike'):
            reasoning += "Volume spike detected. "
        if context.get('bb_squeeze'):
            reasoning += "Bollinger Band squeeze suggests breakout. "
        
        ai_preds = confluence['ai_predictions']
        if ai_preds.get('aria_lstm', {}).get('confidence', 0) > 0.7:
            reasoning += "Strong AI model conviction. "
        
        return reasoning.strip()

    def _get_default_predictions(self) -> Dict[str, Any]:
        """Default predictions when AI models fail"""
        return {
            'aria_lstm': {'direction': 'NEUTRAL', 'confidence': 0.5},
            'prophet': {'trend': 'SIDEWAYS', 'confidence': 0.5},
            'xgboost': {'volatility': 'MEDIUM', 'score': 0.5},
            'finbert': {'sentiment': 'NEUTRAL', 'score': 0.5},
            'gemini': {'analysis': 'NEUTRAL|0.5|Default analysis'}
        }

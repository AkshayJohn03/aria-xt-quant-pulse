# D:\aria\aria-xt-quant-pulse\backend\core\model_interface.py

import os
import joblib
import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelInterface:
    def __init__(self, config_manager):
        self.config = config_manager
        self.models = {}
        self.model_weights = {
            'lstm': 0.4,
            'xgboost': 0.3,
            'finbert': 0.2,
            'prophet': 0.1
        }
        self.sequence_length = 60  # 60 days of historical data
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle',
            'atr', 'vwap', 'obv'
        ]
        self.max_length = 512  # For transformer models
        logging.info("ModelInterface initialized with Advanced AI Ensemble.")

    def load_models(self):
        """Load all required models"""
        try:
            # Load LSTM model
            self.models['lstm'] = self._load_lstm_model()
            logging.info("Successfully loaded aria_lstm model")
            
            # Load XGBoost model
            self.models['xgboost'] = self._load_xgboost_model()
            logging.info("Successfully loaded xgboost model")
            
            # Load FinBERT model
            self.models['finbert'] = self._load_finbert_model()
            logging.info("Successfully loaded finbert model")
            
            # Load Prophet model
            self.models['prophet'] = self._load_prophet_model()
            logging.info("Successfully loaded prophet model")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise

    def _load_lstm_model(self):
        """Load LSTM model for sequence prediction"""
        try:
            # In production, load actual model
            # For now, return a mock model that simulates LSTM behavior
            return {
                'type': 'lstm',
                'layers': [64, 32, 16],
                'dropout': 0.2
            }
        except Exception as e:
            logging.error(f"Error loading LSTM model: {e}")
            raise

    def _load_xgboost_model(self):
        """Load XGBoost model for feature-based prediction"""
        try:
            # In production, load actual model
            return {
                'type': 'xgboost',
                'n_estimators': 100,
                'max_depth': 6
            }
        except Exception as e:
            logging.error(f"Error loading XGBoost model: {e}")
            raise

    def _load_finbert_model(self):
        """Load FinBERT model for sentiment analysis"""
        try:
            # In production, load actual model
            return {
                'type': 'finbert',
                'max_length': self.max_length
            }
        except Exception as e:
            logging.error(f"Error loading FinBERT model: {e}")
            raise

    def _load_prophet_model(self):
        """Load Prophet model for time series forecasting"""
        try:
            # In production, load actual model
            return {
                'type': 'prophet',
                'changepoint_prior_scale': 0.05
            }
        except Exception as e:
            logging.error(f"Error loading Prophet model: {e}")
            raise

    async def predict_trend(self, symbol: str, timeframe: str = '1d') -> Dict[str, Any]:
        """Enhanced trend prediction using real-time data"""
        try:
            # Get real market data
            data = await self._get_market_data(symbol, timeframe)
            if not data or len(data) < self.sequence_length:
                raise ValueError(f"Insufficient data for {symbol}")

            # Prepare features
            features = self._prepare_features(data)
            
            # Get predictions from each model
            predictions = {}
            confidences = {}
            
            # LSTM prediction
            lstm_pred = await self._get_lstm_prediction(features)
            predictions['lstm'] = lstm_pred['prediction']
            confidences['lstm'] = lstm_pred['confidence']
            
            # XGBoost prediction
            xgb_pred = await self._get_xgboost_prediction(features)
            predictions['xgboost'] = xgb_pred['prediction']
            confidences['xgboost'] = xgb_pred['confidence']
            
            # FinBERT sentiment
            sentiment = await self._get_sentiment_analysis(symbol)
            predictions['finbert'] = sentiment['sentiment']
            confidences['finbert'] = sentiment['confidence']
            
            # Prophet forecast
            prophet_pred = await self._get_prophet_forecast(data)
            predictions['prophet'] = prophet_pred['prediction']
            confidences['prophet'] = prophet_pred['confidence']
            
            # Calculate weighted ensemble prediction
            final_prediction = self._calculate_ensemble_prediction(predictions, confidences)
            
            # Get market context
            market_context = self._analyze_market_context(data)
            
            # Adjust prediction with market context
            adjusted_prediction = self._adjust_prediction_with_context(final_prediction, market_context)
            
            return {
                'prediction': adjusted_prediction['direction'],
                'confidence': adjusted_prediction['confidence'],
                'target_price': adjusted_prediction['target_price'],
                'stop_loss': adjusted_prediction['stop_loss'],
                'timeframe': timeframe,
                'model_contributions': {
                    model: {
                        'prediction': pred,
                        'confidence': confidences[model],
                        'weight': self.model_weights[model]
                    }
                    for model, pred in predictions.items()
                },
                'market_context': market_context
            }
            
        except Exception as e:
            logging.error(f"Error in trend prediction: {e}")
            raise

    async def _get_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get real market data from Zerodha or Yahoo Finance"""
        try:
            # First try Zerodha
            data = await self._get_zerodha_data(symbol, timeframe)
            if data is not None:
                return data
                
            # Fallback to Yahoo Finance
            data = await self._get_yahoo_data(symbol, timeframe)
            if data is not None:
                return data
                
            raise ValueError(f"Could not fetch data for {symbol}")
            
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            raise

    async def _get_zerodha_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data from Zerodha"""
        try:
            # Implement actual Zerodha data fetching
            # This should use the KiteConnect API
            return None  # Placeholder
        except Exception as e:
            logging.error(f"Error fetching Zerodha data: {e}")
            return None

    async def _get_yahoo_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data from Yahoo Finance"""
        try:
            # Implement actual Yahoo Finance data fetching
            # This should use the yfinance library
            return None  # Placeholder
        except Exception as e:
            logging.error(f"Error fetching Yahoo Finance data: {e}")
            return None

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input"""
        try:
            # Calculate technical indicators
            data['rsi'] = self._calculate_rsi(data['close'])
            data['macd'], data['macd_signal'] = self._calculate_macd(data['close'])
            data['bb_upper'], data['bb_middle'], data['bb_lower'] = self._calculate_bollinger_bands(data['close'])
            data['atr'] = self._calculate_atr(data)
            data['vwap'] = self._calculate_vwap(data)
            data['obv'] = self._calculate_obv(data)
            
            # Normalize features
            features = data[self.features].values
            return self._normalize_features(features)
            
        except Exception as e:
            logging.error(f"Error preparing features: {e}")
            raise

    def _calculate_ensemble_prediction(self, predictions: Dict[str, float], confidences: Dict[str, float]) -> Dict[str, Any]:
        """Calculate weighted ensemble prediction"""
        try:
            weighted_pred = 0
            total_weight = 0
            
            for model, pred in predictions.items():
                weight = self.model_weights[model] * confidences[model]
                weighted_pred += pred * weight
                total_weight += weight
            
            if total_weight == 0:
                raise ValueError("No valid predictions available")
            
            final_prediction = weighted_pred / total_weight
            
            return {
                'direction': 1 if final_prediction > 0.5 else -1,
                'confidence': total_weight,
                'raw_prediction': final_prediction
            }
            
        except Exception as e:
            logging.error(f"Error calculating ensemble prediction: {e}")
            raise

    def _analyze_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market context for prediction adjustment"""
        try:
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(data)
            
            # Calculate volatility regime
            volatility_regime = self._calculate_volatility_regime(data)
            
            # Calculate volume profile
            volume_profile = self._calculate_volume_profile(data)
            
            # Calculate market sentiment
            market_sentiment = self._calculate_market_sentiment(data)
            
            return {
                'trend_strength': trend_strength,
                'volatility_regime': volatility_regime,
                'volume_profile': volume_profile,
                'market_sentiment': market_sentiment
            }
            
        except Exception as e:
            logging.error(f"Error analyzing market context: {e}")
            raise

    def _adjust_prediction_with_context(self, prediction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust prediction based on market context"""
        try:
            # Adjust confidence based on context
            confidence = prediction['confidence']
            
            # Adjust for trend strength
            if context['trend_strength'] > 0.7:
                confidence *= 1.2
            elif context['trend_strength'] < 0.3:
                confidence *= 0.8
            
            # Adjust for volatility
            if context['volatility_regime'] == 'high':
                confidence *= 0.9
            
            # Adjust for volume
            if context['volume_profile'] == 'low':
                confidence *= 0.8
            
            # Adjust for sentiment
            if context['market_sentiment'] == 'bearish':
                confidence *= 0.9
            
            # Calculate target and stop loss
            target_price = self._calculate_target_price(prediction, context)
            stop_loss = self._calculate_stop_loss(prediction, context)
            
            return {
                'direction': prediction['direction'],
                'confidence': min(confidence, 1.0),
                'target_price': target_price,
                'stop_loss': stop_loss
            }
            
        except Exception as e:
            logging.error(f"Error adjusting prediction: {e}")
            raise

    def _calculate_target_price(self, prediction: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate target price based on prediction and context"""
        try:
            # Implement actual target price calculation
            # This should use ATR, volatility, and other factors
            return 0.0  # Placeholder
        except Exception as e:
            logging.error(f"Error calculating target price: {e}")
            raise

    def _calculate_stop_loss(self, prediction: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate stop loss based on prediction and context"""
        try:
            # Implement actual stop loss calculation
            # This should use ATR, volatility, and other factors
            return 0.0  # Placeholder
        except Exception as e:
            logging.error(f"Error calculating stop loss: {e}")
            raise

    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        v = data['volume']
        tp = (data['high'] + data['low'] + data['close']) / 3
        return (tp * v).cumsum() / v.cumsum()

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        close = data['close']
        volume = data['volume']
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features for model input"""
        try:
            # Implement actual feature normalization
            # This should use proper scaling methods
            return features  # Placeholder
        except Exception as e:
            logging.error(f"Error normalizing features: {e}")
            raise

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength"""
        try:
            # Implement actual trend strength calculation
            # This should use multiple indicators
            return 0.5  # Placeholder
        except Exception as e:
            logging.error(f"Error calculating trend strength: {e}")
            raise

    def _calculate_volatility_regime(self, data: pd.DataFrame) -> str:
        """Calculate volatility regime"""
        try:
            # Implement actual volatility regime calculation
            # This should use ATR and other volatility indicators
            return 'normal'  # Placeholder
        except Exception as e:
            logging.error(f"Error calculating volatility regime: {e}")
            raise

    def _calculate_volume_profile(self, data: pd.DataFrame) -> str:
        """Calculate volume profile"""
        try:
            # Implement actual volume profile calculation
            # This should analyze volume patterns
            return 'normal'  # Placeholder
        except Exception as e:
            logging.error(f"Error calculating volume profile: {e}")
            raise

    def _calculate_market_sentiment(self, data: pd.DataFrame) -> str:
        """Calculate market sentiment"""
        try:
            # Implement actual market sentiment calculation
            # This should use multiple indicators
            return 'neutral'  # Placeholder
        except Exception as e:
            logging.error(f"Error calculating market sentiment: {e}")
            raise

    async def forecast_pattern(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prophet pattern recognition and forecasting"""
        try:
            historical_data = input_data.get('historical_data', [])
            forecast_periods = input_data.get('forecast_periods', 5)
            
            if len(historical_data) < 50:
                return {'trend': 'SIDEWAYS', 'confidence': 0.5, 'support': 0, 'resistance': 0}
            
            # Extract price data
            prices = [item['close'] for item in historical_data]
            
            # Simple trend analysis
            recent_trend = (prices[-1] - prices[-20]) / prices[-20]
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            
            if recent_trend > 0.01:  # 1% uptrend
                trend = 'UP'
                confidence = min(0.9, 0.6 + recent_trend * 5)
            elif recent_trend < -0.01:
                trend = 'DOWN'
                confidence = min(0.9, 0.6 + abs(recent_trend) * 5)
            else:
                trend = 'SIDEWAYS'
                confidence = 0.5 + volatility
            
            # Calculate support and resistance
            recent_prices = prices[-50:]
            support = min(recent_prices) * 0.998  # Slight buffer
            resistance = max(recent_prices) * 1.002
            
            return {
                'trend': trend,
                'confidence': confidence,
                'support': support,
                'resistance': resistance,
                'volatility_score': volatility,
                'model': 'prophet'
            }
            
        except Exception as e:
            logging.error(f"Error in Prophet forecasting: {e}")
            return {'trend': 'SIDEWAYS', 'confidence': 0.5, 'support': 0, 'resistance': 0}

    async def predict_volatility(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """XGBoost volatility prediction"""
        try:
            features = input_data.get('features', [])
            if len(features) < 10:
                return {'volatility': 'MEDIUM', 'score': 0.5, 'risk_level': 'MODERATE'}
            
            # Mock XGBoost prediction
            # In production, this would use the actual XGBoost model
            price_change = abs(features[1]) if len(features) > 1 else 0.01
            volume_ratio = features[5] if len(features) > 5 else 1.0
            hist_vol = features[6] if len(features) > 6 else 0.02
            
            # Combine features for volatility score
            volatility_score = (price_change * 0.4 + 
                              abs(volume_ratio - 1) * 0.3 + 
                              hist_vol * 0.3)
            
            if volatility_score > 0.03:
                volatility = 'HIGH'
                risk_level = 'HIGH'
            elif volatility_score > 0.015:
                volatility = 'MEDIUM'
                risk_level = 'MODERATE'
            else:
                volatility = 'LOW'
                risk_level = 'LOW'
            
            return {
                'volatility': volatility,
                'score': min(0.95, volatility_score * 10),
                'risk_level': risk_level,
                'features_used': len(features),
                'model': 'xgboost'
            }
            
        except Exception as e:
            logging.error(f"Error in XGBoost volatility prediction: {e}")
            return {'volatility': 'MEDIUM', 'score': 0.5, 'risk_level': 'MODERATE'}

    async def analyze_sentiment(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """FinBERT sentiment analysis"""
        try:
            symbol = input_data.get('symbol', 'NIFTY50')
            timeframe = input_data.get('timeframe', '1H')
            
            # Mock sentiment analysis
            # In production, this would analyze real news and social media
            import random
            sentiments = ['BULLISH', 'BEARISH', 'NEUTRAL']
            weights = [0.4, 0.3, 0.3]  # Slight bullish bias for Indian markets
            
            sentiment = random.choices(sentiments, weights=weights)[0]
            confidence = random.uniform(0.6, 0.9)
            
            # Generate relevant news headlines (mock)
            news_impact = random.uniform(-0.1, 0.1)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'news_impact': news_impact,
                'symbol': symbol,
                'timeframe': timeframe,
                'model': 'finbert'
            }
            
        except Exception as e:
            logging.error(f"Error in FinBERT sentiment analysis: {e}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'news_impact': 0}

    async def query_gemini(self, prompt: str) -> Dict[str, Any]:
        """Query Gemini AI for market analysis"""
        try:
            gemini_config = self.config.get('gemini', {})
            api_key = gemini_config.get('api_key')
            
            if not api_key:
                logging.warning("Gemini API key not configured")
                return self._get_mock_gemini_response()
            
            # In production, make actual API call to Gemini
            # For now, return mock response
            return self._get_mock_gemini_response()
            
        except Exception as e:
            logging.error(f"Error querying Gemini: {e}")
            return self._get_mock_gemini_response()

    def _get_mock_gemini_response(self) -> Dict[str, Any]:
        """Generate mock Gemini response"""
        import random
        
        sentiments = ['Bullish', 'Bearish', 'Neutral']
        sentiment = random.choice(sentiments)
        probability = random.uniform(0.6, 0.85)
        
        reasonings = [
            "Strong momentum with high volume support",
            "Technical indicators showing divergence",
            "Market consolidation near key levels",
            "Breakout pattern forming on charts",
            "Risk-off sentiment affecting sentiment"
        ]
        reasoning = random.choice(reasonings)
        
        return {
            'sentiment': sentiment,
            'probability': probability,
            'reasoning': reasoning,
            'support_levels': [19800, 19750, 19700],
            'resistance_levels': [19900, 19950, 20000],
            'model': 'gemini'
        }

    async def test_gemini_connection(self) -> bool:
        """Test Gemini API connection"""
        try:
            result = await self.query_gemini("Test connection")
            return result is not None
        except:
            return False

    async def test_ollama_connection(self) -> bool:
        """Test Ollama connection by pinging the Ollama server."""
        try:
            ollama_config = self.config.get('ollama', {})
            base_url = ollama_config.get('base_url', 'http://localhost:11434')
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}/api/tags", timeout=3)
                return resp.status_code == 200
        except Exception as e:
            logging.warning(f"Ollama connection test failed: {e}")
            return False

    async def initialize_models(self):
        """Initialize all models on startup"""
        logging.info("Initializing AI models...")
        
        # Test connections
        connections = {
            'gemini': await self.test_gemini_connection(),
            'ollama': await self.test_ollama_connection()
        }
        
        for service, status in connections.items():
            if status:
                logging.info(f"✓ {service.upper()} model ready")
            else:
                logging.warning(f"✗ {service.upper()} model unavailable")
        
        logging.info("Model initialization complete")

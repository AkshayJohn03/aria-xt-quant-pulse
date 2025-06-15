import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Trading Parameters
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
    MAX_TRADES_PER_DAY = int(os.getenv('MAX_TRADES_PER_DAY', '3'))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))
    
    # Strategy Parameters
    STRATEGY_PROFILE = os.getenv('STRATEGY_PROFILE', 'hybrid')
    TRADING_MODE = os.getenv('TRADING_MODE', 'safe')
    
    # Technical Indicators
    SUPERTREND_PERIOD = int(os.getenv('SUPERTREND_PERIOD', '10'))
    SUPERTREND_MULTIPLIER = float(os.getenv('SUPERTREND_MULTIPLIER', '3.0'))
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
    RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', '70.0'))
    RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', '30.0'))
    EMA_SHORT_PERIOD = int(os.getenv('EMA_SHORT_PERIOD', '9'))
    EMA_LONG_PERIOD = int(os.getenv('EMA_LONG_PERIOD', '21'))
    MACD_FAST_PERIOD = int(os.getenv('MACD_FAST_PERIOD', '12'))
    MACD_SLOW_PERIOD = int(os.getenv('MACD_SLOW_PERIOD', '26'))
    MACD_SIGNAL_PERIOD = int(os.getenv('MACD_SIGNAL_PERIOD', '9'))
    ADX_PERIOD = int(os.getenv('ADX_PERIOD', '14'))
    ADX_THRESHOLD = float(os.getenv('ADX_THRESHOLD', '25.0'))
    VWAP_PERIOD = int(os.getenv('VWAP_PERIOD', '14'))
    
    # Black-Scholes Parameters
    RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', '0.05'))
    TIME_TO_EXPIRY_DAYS = int(os.getenv('TIME_TO_EXPIRY_DAYS', '30'))
    VOLATILITY_WINDOW = int(os.getenv('VOLATILITY_WINDOW', '20'))
    
    # Delta Thresholds
    MIN_DELTA = float(os.getenv('MIN_DELTA', '0.3'))
    MAX_DELTA = float(os.getenv('MAX_DELTA', '0.7'))
    
    # Signal Generation
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
    MIN_SIGNAL_AGREEMENT = float(os.getenv('MIN_SIGNAL_AGREEMENT', '0.6'))
    STOP_LOSS_ATR_MULTIPLIER = float(os.getenv('STOP_LOSS_ATR_MULTIPLIER', '2.0'))
    TAKE_PROFIT_ATR_MULTIPLIER = float(os.getenv('TAKE_PROFIT_ATR_MULTIPLIER', '3.0'))
    ATR_PERIOD = int(os.getenv('ATR_PERIOD', '14'))
    
    # Flask Settings
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = bool(int(os.getenv('FLASK_DEBUG', '1')))
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24).hex())
    
    # Model Weights
    MODEL_WEIGHTS = {
        'aria_lstm': float(os.getenv('ARIA_LSTM_WEIGHT', '0.3')),
        'prophet': float(os.getenv('PROPHET_WEIGHT', '0.2')),
        'xgboost': float(os.getenv('XGBOOST_WEIGHT', '0.2')),
        'qwen': float(os.getenv('QWEN_WEIGHT', '0.3'))
    }
    
    # Kalman Filter Parameters
    KALMAN_PROCESS_VARIANCE = float(os.getenv('KALMAN_PROCESS_VARIANCE', '0.01'))
    KALMAN_MEASUREMENT_VARIANCE = float(os.getenv('KALMAN_MEASUREMENT_VARIANCE', '0.1'))
    
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        required_keys = ['TWELVEDATA_API_KEY']
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
        
        # Validate model weights sum to 1
        model_weights_sum = sum(cls.MODEL_WEIGHTS.values())
        if not 0.99 <= model_weights_sum <= 1.01:  # Allow for floating-point imprecision
            raise ValueError(f"Model weights must sum to 1.0 (current sum: {model_weights_sum})")
        
        # Validate trading parameters
        if cls.INITIAL_CAPITAL <= 0:
            raise ValueError("Initial capital must be positive")
        
        if not 0 < cls.RISK_PER_TRADE < 1:
            raise ValueError("Risk per trade must be between 0 and 1")
        
        if cls.MAX_TRADES_PER_DAY < 1:
            raise ValueError("Maximum trades per day must be at least 1")
        
        # Validate technical indicator parameters
        if cls.SUPERTREND_PERIOD < 1 or cls.RSI_PERIOD < 1:
            raise ValueError("Technical indicator periods must be positive")
        
        if not 0 <= cls.RSI_OVERSOLD < cls.RSI_OVERBOUGHT <= 100:
            raise ValueError("Invalid RSI overbought/oversold levels")
        
        # Validate Black-Scholes parameters
        if cls.RISK_FREE_RATE < 0:
            raise ValueError("Risk-free rate cannot be negative")
        
        if cls.TIME_TO_EXPIRY_DAYS < 1:
            raise ValueError("Time to expiry must be positive")
        
        # Validate delta thresholds
        if not 0 <= cls.MIN_DELTA < cls.MAX_DELTA <= 1:
            raise ValueError("Invalid delta thresholds")
        
        # Validate signal generation parameters
        if not 0 < cls.CONFIDENCE_THRESHOLD <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if not 0 < cls.MIN_SIGNAL_AGREEMENT <= 1:
            raise ValueError("Minimum signal agreement must be between 0 and 1")
        
        if cls.STOP_LOSS_ATR_MULTIPLIER <= 0 or cls.TAKE_PROFIT_ATR_MULTIPLIER <= 0:
            raise ValueError("ATR multipliers must be positive")
        
        return True 
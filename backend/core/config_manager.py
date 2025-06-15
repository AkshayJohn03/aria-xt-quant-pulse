import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dotenv import load_dotenv # ADD THIS IMPORT

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration and settings"""
    
    def __init__(self, config_file: str = "config.json"):
        load_dotenv() # ADD THIS LINE: Loads environment variables from .env
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self._merge_env_variables() # ADD THIS LINE: Merge env vars after loading JSON
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
                return config
            except Exception as e:
                logger.error(f"Error loading config from {self.config_file}: {e}. Falling back to default.")
                return self.get_default_config() # Still fall back if JSON is corrupted
        else:
            config = self.get_default_config()
            self.save_config(config)
            logger.info(f"No {self.config_file} found. Created with default configuration.")
            return config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file"""
        try:
            config_to_save = config or self.config
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return a base default configuration without directly reading env vars.
           Env vars will be merged by _merge_env_variables."""
        # Changed this to return a default structure. _merge_env_variables will
        # fill in the env specific keys.
        return {
            "apis": {
                "zerodha": {
                    "api_key": None, # Set to None, will be filled by env var
                    "api_secret": None, # Set to None, will be filled by env var
                    "access_token": None, # Set to None, will be filled by env var
                    "base_url": "https://api.kite.trade"
                },
                "twelve_data": {
                    "api_key": None, # Set to None, will be filled by env var
                    "base_url": "https://api.twelvedata.com"
                },
                "gemini": {
                    "api_key": None, # Set to None, will be filled by env var
                    "model": "gemini-2.0-flash-exp",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta"
                },
                "telegram": {
                    "bot_token": None, # Set to None, will be filled by env var
                    "chat_id": None # Set to None, will be filled by env var
                }
            },
            "trading": {
                "max_risk_per_trade": 2.5,
                "trailing_stop_percent": 5.0,
                "max_positions": 5,
                "capital_allocation": 100000,
                "min_confidence_threshold": 75.0,
                "enable_auto_trading": False
            },
            "models": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "qwen2.5:0.5b",
                    "timeout": 30
                },
                "aria_lstm": {
                    "model_path": "models/aria_lstm.pth",
                    "sequence_length": 60,
                    "features": ["open", "high", "low", "close", "volume"]
                },
                "finbert": {
                    "model_path": "models/finbert-quantized",
                    "max_length": 512
                },
                "prophet": {
                    "seasonality_mode": "multiplicative",
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False
                },
                "xgboost": {
                    "model_path": "models/xgboost_model.pkl",
                    "feature_importance_threshold": 0.01
                }
            },
            "data": {
                "symbols": ["NIFTY50", "BANKNIFTY"],
                "timeframes": ["1min", "5min", "15min"],
                "max_history_days": 30,
                "cache_duration_minutes": 5
            },
            "risk_management": {
                "max_drawdown_percent": 15.0,
                "position_sizing_method": "fixed_percentage",
                "volatility_adjustment": True,
                "correlation_threshold": 0.7
            },
            "notifications": {
                "trade_entry": True,
                "trade_exit": True,
                "profit_target": True,
                "stop_loss": True,
                "system_errors": True,
                "daily_summary": True
            }
        }
    
    def _merge_env_variables(self):
        """Helper to explicitly merge environment variables into the loaded config."""
        env_map = {
            "apis.zerodha.api_key": "ZERODHA_API_KEY",
            "apis.zerodha.api_secret": "ZERODHA_API_SECRET",
            "apis.zerodha.access_token": "ZERODHA_ACCESS_TOKEN",
            "apis.twelve_data.api_key": "TWELVE_DATA_API_KEY",
            "apis.gemini.api_key": "GEMINI_API_KEY",
            "apis.telegram.bot_token": "TELEGRAM_BOT_TOKEN",
            "apis.telegram.chat_id": "TELEGRAM_CHAT_ID",
        }

        for config_path, env_var_name in env_map.items():
            # Patch: treat empty string as None so env var is used
            current_value = self.get(config_path)
            env_value = os.getenv(env_var_name)
            if (current_value is None or current_value == "") and env_value is not None:
                self.set(config_path, env_value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        # Don't save immediately here unless you want every 'set' to write to file
        # The update_config method handles saving.
        return True # Indicate success
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values and save to file"""
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        update_nested_dict(self.config, updates)
        return self.save_config() # Save after all updates
    
    def validate_config(self) -> bool:
        """Validate current configuration"""
        required_fields = [
            "apis.zerodha.api_key",
            "apis.telegram.bot_token",
            "trading.capital_allocation"
        ]
        
        for field in required_fields:
            value = self.get(field)
            if not value: # This check is good, as it catches both None and ""
                logger.error(f"Missing required configuration: {field}")
                return False
        
        # Validate numeric ranges
        if not (0 < self.get("trading.max_risk_per_trade", 0) <= 10):
            logger.error("max_risk_per_trade must be between 0 and 10")
            return False
        
        if not (0 < self.get("trading.capital_allocation", 0)):
            logger.error("capital_allocation must be greater than 0")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_api_credentials(self, service: str) -> Dict[str, str]:
        """Get API credentials for a specific service"""
        return self.get(f"apis.{service}", {})
    
    def get_trading_params(self) -> Dict[str, Any]:
        """Get trading configuration parameters"""
        return self.get("trading", {})
    
    def get_model_config(self, model: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self.get(f"models.{model}", {})
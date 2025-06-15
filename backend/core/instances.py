"""
This module contains shared instances used across the application.
These instances are initialized in app.py and can be imported by other modules.
"""

from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Import core components
from core.config_manager import ConfigManager
from core.data_fetcher import DataFetcher
from core.model_interface import ModelInterface
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator
from core.trade_executor import TradeExecutor
from core.telegram_notifier import TelegramNotifier

# Initialize shared instances
config_manager = None
data_fetcher = None
model_interface = None
risk_manager = None
trade_executor = None
signal_generator = None
telegram_notifier = None

# System status
system_status = {
    "is_running": False,
    "last_update": None,
    "active_trades": 0,
    "total_pnl": 0.0,
    "system_health": "OK",
    "components": {
        "config_manager": False,
        "data_fetcher": False,
        "model_interface": False,
        "risk_manager": False,
        "trade_executor": False,
        "signal_generator": False,
        "telegram_notifier": False
    }
}

def init_instances():
    """Initialize all instances in the correct order"""
    global config_manager, data_fetcher, model_interface, risk_manager, trade_executor, signal_generator, telegram_notifier
    
    try:
        # Initialize config manager first
        config_manager = ConfigManager()
        if not config_manager.load_config():
            raise Exception("Failed to load configuration")
        system_status["components"]["config_manager"] = True
        logging.info("Config manager initialized")
        
        # Initialize data fetcher
        data_fetcher = DataFetcher(config_manager)
        if not data_fetcher.initialize():
            logging.warning("Data fetcher initialization failed, will use fallback data")
        else:
            system_status["components"]["data_fetcher"] = True
            logging.info("Data fetcher initialized")
        
        # Initialize model interface
        model_interface = ModelInterface(config_manager)
        if not model_interface.load_models():
            logging.warning("Model interface initialization failed, will use fallback predictions")
        else:
            system_status["components"]["model_interface"] = True
            logging.info("Model interface initialized")
        
        # Initialize trade executor first since risk manager needs it
        trade_executor = TradeExecutor(config_manager)
        if not trade_executor.initialize():
            logging.warning("Trade executor initialization failed, will use paper trading mode")
        else:
            system_status["components"]["trade_executor"] = True
            logging.info("Trade executor initialized")
        
        # Initialize risk manager with both required arguments
        risk_manager = RiskManager(trade_executor, config_manager)
        system_status["components"]["risk_manager"] = True
        logging.info("Risk manager initialized")
        
        # Initialize signal generator
        signal_generator = SignalGenerator(config_manager, model_interface, risk_manager)
        system_status["components"]["signal_generator"] = True
        logging.info("Signal generator initialized")
        
        # Initialize telegram notifier
        telegram_notifier = TelegramNotifier(config_manager)
        system_status["components"]["telegram_notifier"] = True
        logging.info("Telegram notifier initialized")
        
        # Update system status
        system_status["is_running"] = True
        system_status["last_update"] = datetime.now()
        logging.info("All components initialized successfully")
        
    except Exception as e:
        logging.error(f"Error initializing instances: {e}")
        raise

def get_system_status() -> Dict[str, Any]:
    """Get the current system status"""
    return {
        **system_status,
        "last_update": datetime.now().isoformat()
    }

def update_system_status(updates: Dict[str, Any]):
    """Update the system status"""
    global system_status
    system_status.update(updates)
    system_status["last_update"] = datetime.now().isoformat() 
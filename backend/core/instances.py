
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global instances
config_manager = None
data_fetcher = None
model_interface = None
risk_manager = None
signal_generator = None
trade_executor = None
telegram_notifier = None

def init_instances() -> bool:
    """Initialize all shared instances"""
    global config_manager, data_fetcher, model_interface, risk_manager, signal_generator, trade_executor, telegram_notifier
    
    try:
        # Import here to avoid circular imports
        from .data_fetcher import DataFetcher
        
        # Initialize basic components
        data_fetcher = DataFetcher()
        logger.info("Data fetcher initialized")
        
        # Initialize other components as needed
        # For now, we'll keep them as None to avoid import errors
        config_manager = None
        model_interface = None
        risk_manager = None
        signal_generator = None
        trade_executor = None
        telegram_notifier = None
        
        logger.info("Core instances initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing instances: {e}")
        return False

def get_system_status():
    """Get current system status"""
    return {
        "data_fetcher": data_fetcher is not None,
        "config_manager": config_manager is not None,
        "model_interface": model_interface is not None,
        "risk_manager": risk_manager is not None,
        "signal_generator": signal_generator is not None,
        "trade_executor": trade_executor is not None,
        "telegram_notifier": telegram_notifier is not None
    }

def update_system_status(status: dict):
    """Update system status"""
    logger.info(f"System status updated: {status}")

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
        from .data_fetcher import DataFetcher
        from .config_manager import ConfigManager
        from .model_interface import ModelInterface
        from .risk_manager import RiskManager
        from .signal_generator import SignalGenerator
        from .trade_executor import TradeExecutor
        from .telegram_notifier import TelegramNotifier

        # Initialize config manager first
        if config_manager is None:
            config_manager = ConfigManager()
            logger.info("Config manager initialized")
        # Data fetcher
        if data_fetcher is None:
            data_fetcher = DataFetcher()
            logger.info("Data fetcher initialized")
        # Trade executor
        if trade_executor is None:
            trade_executor = TradeExecutor(config_manager)
            logger.info("Trade executor initialized")
        # Model interface
        if model_interface is None:
            model_interface = ModelInterface(config_manager)
            logger.info("Model interface initialized")
        # Risk manager (needs trade_executor and config_manager)
        if risk_manager is None:
            risk_manager = RiskManager(trade_executor, config_manager)
            logger.info("Risk manager initialized")
        # Signal generator
        if signal_generator is None:
            signal_generator = SignalGenerator(config_manager)
            logger.info("Signal generator initialized")
        # Telegram notifier
        if telegram_notifier is None:
            telegram_notifier = TelegramNotifier(config_manager)
            logger.info("Telegram notifier initialized")
        logger.info("All core instances initialized successfully")
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

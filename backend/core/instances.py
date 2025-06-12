"""
This module contains shared instances used across the application.
These instances are initialized in app.py and can be imported by other modules.
"""

from typing import Optional

# Import core components
from core.config_manager import ConfigManager
from core.data_fetcher import DataFetcher
from core.model_interface import ModelInterface
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator
from core.trade_executor import TradeExecutor
from core.telegram_notifier import TelegramNotifier

# Initialize variables to None
config_manager: Optional[ConfigManager] = None
data_fetcher: Optional[DataFetcher] = None
model_interface: Optional[ModelInterface] = None
risk_manager: Optional[RiskManager] = None
signal_generator: Optional[SignalGenerator] = None
trade_executor: Optional[TradeExecutor] = None
telegram_notifier: Optional[TelegramNotifier] = None

# System status
system_status = {
    "is_running": False,
    "last_update": None,
    "active_trades": 0,
    "total_pnl": 0.0,
    "system_health": "OK"
}

def init_instances():
    """Initialize all instances. This should be called from app.py"""
    global config_manager, data_fetcher, model_interface, risk_manager, signal_generator, trade_executor, telegram_notifier
    
    # Initialize components in the correct order
    config_manager = ConfigManager()
    data_fetcher = DataFetcher(config_manager)
    model_interface = ModelInterface(config_manager)
    trade_executor = TradeExecutor(config_manager)
    risk_manager = RiskManager(trade_executor, config_manager)
    signal_generator = SignalGenerator(config_manager, model_interface, risk_manager)
    telegram_notifier = TelegramNotifier(config_manager)
    
    # Set up circular reference
    trade_executor.set_risk_manager(risk_manager) 
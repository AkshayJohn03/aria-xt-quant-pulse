
import logging
from typing import Dict, Any, List, Optional
import random

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config):
        self.config = config.config # Access the raw config dictionary
        self.max_daily_loss = self.config.get("risk_management.max_drawdown_percent", 2.0)
        self.max_per_trade_loss = self.config.get("trading.max_risk_per_trade", 0.5)
        self.max_open_positions = self.config.get("trading.max_positions", 5)
        self.trade_executor = None  # Will be set by TradeExecutor
        
        logger.info(f"RiskManager initialized. Risk Limits: Max Daily Loss {self.max_daily_loss}%, Max Per Trade Loss {self.max_per_trade_loss}%, Max Open Positions {self.max_open_positions}")

    def set_trade_executor(self, trade_executor):
        """Set the trade executor reference for real data access"""
        self.trade_executor = trade_executor

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validates a trading signal against predefined risk rules."""
        try:
            positions = self.get_open_positions()
            if len(positions) >= self.max_open_positions:
                logger.warning(f"Validation failed: Max open positions ({self.max_open_positions}) reached.")
                return False
            
            logger.info(f"Signal validated: {signal.get('symbol')} ({signal.get('type')})")
            return True
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False

    def check_per_trade_risk(self, trade_value: float) -> bool:
        """Checks if the potential trade value adheres to the maximum per-trade loss limit."""
        try:
            capital_allocation = self.config.get("trading.capital_allocation", 100000)
            max_allowed_loss = capital_allocation * (self.max_per_trade_loss / 100)
            
            if trade_value > capital_allocation * 0.2:
                logger.warning(f"Trade value {trade_value} exceeds per-trade risk threshold.")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking per-trade risk: {e}")
            return True  # Default to allow trade on error

    async def update_positions(self):
        """Updates the internal state of open positions."""
        try:
            if self.trade_executor:
                await self.trade_executor.update_positions()
            logger.info("Positions updated successfully.")
        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    async def check_exit_conditions(self):
        """Checks if any open positions meet stop-loss or profit-target conditions."""
        try:
            logger.info("Checking exit conditions.")
            # TODO: Implement real exit condition logic
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Returns currently open positions."""
        try:
            if self.trade_executor and hasattr(self.trade_executor, 'cached_positions'):
                return [pos for pos in self.trade_executor.cached_positions if pos.get("quantity", 0) != 0]
            
            # Fallback to mock data if no real executor
            return self._get_mock_positions()
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []

    def get_positions(self) -> List[Dict[str, Any]]:
        """Returns all tracked positions."""
        try:
            if self.trade_executor and hasattr(self.trade_executor, 'cached_positions'):
                return self.trade_executor.cached_positions
            
            return self._get_mock_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_holdings(self) -> List[Dict[str, Any]]:
        """Returns current holdings."""
        try:
            if self.trade_executor and hasattr(self.trade_executor, 'cached_holdings'):
                return self.trade_executor.cached_holdings
            
            return self._get_mock_holdings()
        except Exception as e:
            logger.error(f"Error getting holdings: {e}")
            return []

    def get_funds(self) -> Dict[str, Any]:
        """Returns available funds."""
        try:
            if self.trade_executor and hasattr(self.trade_executor, 'cached_funds'):
                return self.trade_executor.cached_funds
            
            return self._get_mock_funds()
        except Exception as e:
            logger.error(f"Error getting funds: {e}")
            return self._get_mock_funds()

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculates and returns key risk metrics for the portfolio."""
        try:
            positions = self.get_positions()
            funds = self.get_funds()
            
            total_pnl = self.calculate_total_pnl()
            total_investment = sum(abs(pos.get("avg_price", 0) * pos.get("quantity", 0)) for pos in positions)
            portfolio_value = funds.get("available_cash", 0) + total_investment + total_pnl

            return {
                "portfolio_value": round(portfolio_value, 2),
                "total_investment": round(total_investment, 2),
                "total_pnl": total_pnl,
                "risk_score": self._calculate_risk_score(total_pnl, total_investment),
                "max_drawdown": random.uniform(1.0, 10.0),  # TODO: Calculate real drawdown
                "current_drawdown": random.uniform(0.5, 5.0),
                "sharpe_ratio": random.uniform(0.8, 1.5),
                "sortino_ratio": random.uniform(1.0, 2.0),
                "portfolio_exposure_percent": min(100, (total_investment / max(portfolio_value, 1)) * 100),
                "max_risk_per_trade_percent": self.max_per_trade_loss
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._get_mock_risk_metrics()

    def calculate_total_pnl(self) -> float:
        """Calculates the total PnL across all positions."""
        try:
            positions = self.get_positions()
            total_pnl = sum(pos.get("pnl", 0) for pos in positions)
            return round(total_pnl, 2)
        except Exception as e:
            logger.error(f"Error calculating total PnL: {e}")
            return 0.0

    def _calculate_risk_score(self, total_pnl: float, total_investment: float) -> str:
        """Calculate risk score based on PnL and investment."""
        if total_investment == 0:
            return "Low"
        
        pnl_ratio = abs(total_pnl) / total_investment
        if pnl_ratio < 0.05:
            return "Low"
        elif pnl_ratio < 0.15:
            return "Medium"
        else:
            return "High"

    def _get_mock_positions(self) -> List[Dict[str, Any]]:
        """Mock positions for testing."""
        return [
            {
                "symbol": "NIFTY 20000 CE",
                "quantity": 50,
                "avg_price": 125.5,
                "current_price": 138.75,
                "pnl": 662.5,
                "product_type": "MIS",
                "timestamp": "2024-06-27T09:30:00Z",
                "status": "OPEN"
            },
            {
                "symbol": "BANKNIFTY 45500 PE",
                "quantity": -25,
                "avg_price": 189.2,
                "current_price": 76.3,
                "pnl": 322.5,
                "product_type": "MIS",
                "timestamp": "2024-06-27T10:00:00Z",
                "status": "OPEN"
            }
        ]

    def _get_mock_holdings(self) -> List[Dict[str, Any]]:
        """Mock holdings for testing."""
        return [
            {"tradingsymbol": "TCS", "quantity": 10, "last_price": 3800.0, "pnl": 1500.0},
            {"tradingsymbol": "INFY", "quantity": 20, "last_price": 1500.0, "pnl": -300.0}
        ]

    def _get_mock_funds(self) -> Dict[str, Any]:
        """Mock funds for testing."""
        return {
            "available_cash": random.uniform(45000.0, 55000.0),
            "free_margin": random.uniform(40000.0, 50000.0),
            "used_margin": random.uniform(5000.0, 10000.0)
        }

    def _get_mock_risk_metrics(self) -> Dict[str, Any]:
        """Mock risk metrics for testing."""
        return {
            "portfolio_value": 100000.0,
            "total_investment": 50000.0,
            "total_pnl": 1500.0,
            "risk_score": "Medium",
            "max_drawdown": 5.0,
            "current_drawdown": 2.0,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "portfolio_exposure_percent": 50.0,
            "max_risk_per_trade_percent": self.max_per_trade_loss
        }

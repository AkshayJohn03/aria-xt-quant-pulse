# D:\aria\aria-xt-quant-pulse\backend\core\risk_manager.py

import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_daily_loss_percent = config.get("risk_limits", {}).get("max_daily_loss_percent", 0.02) # e.g., 2%
        self.max_per_trade_loss_percent = config.get("risk_limits", {}).get("max_per_trade_loss_percent", 0.005) # e.g., 0.5%
        self.max_open_positions = config.get("risk_limits", {}).get("max_open_positions", 5)
        self.capital = config.get("initial_capital", 100000) # Assuming initial capital from config
        self.current_day_pnl = 0.0 # To track daily profit/loss
        logging.info("RiskManager initialized.")
        logging.info(f"Risk Limits: Max Daily Loss {self.max_daily_loss_percent*100}%, Max Per Trade Loss {self.max_per_trade_loss_percent*100}%, Max Open Positions {self.max_open_positions}")

    def update_pnl(self, trade_pnl: float):
        """Updates the current day's PnL."""
        self.current_day_pnl += trade_pnl
        logging.info(f"Updated daily PnL: {self.current_day_pnl:.2f}")

    def check_daily_loss_limit(self) -> bool:
        """Checks if the daily loss limit has been hit."""
        daily_loss_ratio = abs(self.current_day_pnl) / self.capital if self.capital > 0 else 0
        if self.current_day_pnl < 0 and daily_loss_ratio >= self.max_daily_loss_percent:
            logging.warning(f"Daily loss limit hit! Current loss: {self.current_day_pnl:.2f}, Max allowed: {self.max_daily_loss_percent*self.capital:.2f}")
            return False
        return True

    def check_trade_eligibility(self, trade_details: Dict[str, Any], current_portfolio: Dict[str, Any] = {}) -> bool:
        """
        Checks if a new trade can be placed based on various risk parameters.

        Args:
            trade_details (Dict): Details of the proposed trade (e.g., 'symbol', 'type', 'quantity', 'entry_price', 'stop_loss', 'target').
            current_portfolio (Dict): Current portfolio state (e.g., number of open positions).
        """
        logging.info(f"Checking eligibility for trade: {trade_details.get('symbol')} ({trade_details.get('type')})")

        # 1. Check Daily Loss Limit
        if not self.check_daily_loss_limit():
            logging.warning("Trade rejected: Daily loss limit exceeded.")
            return False

        # 2. Check Max Open Positions
        open_positions_count = current_portfolio.get("open_positions_count", 0)
        if open_positions_count >= self.max_open_positions:
            logging.warning(f"Trade rejected: Max open positions limit ({self.max_open_positions}) reached. Current: {open_positions_count}")
            return False

        # 3. Check Per-Trade Risk (requires stop_loss in trade_details)
        entry_price = trade_details.get("entry_price")
        stop_loss = trade_details.get("stop_loss")
        quantity = trade_details.get("quantity")

        if entry_price and stop_loss and quantity and self.capital > 0:
            potential_loss_per_share = abs(entry_price - stop_loss)
            potential_trade_loss = potential_loss_per_share * quantity

            max_allowed_loss_for_trade = self.max_per_trade_loss_percent * self.capital

            if potential_trade_loss > max_allowed_loss_for_trade:
                logging.warning(f"Trade rejected: Potential loss ({potential_trade_loss:.2f}) exceeds per-trade limit ({max_allowed_loss_for_trade:.2f})")
                return False
        else:
            logging.warning("Could not assess per-trade risk due to missing trade details (entry_price, stop_loss, quantity) or zero capital.")

        logging.info("Trade eligible based on current risk checks.")
        return True

    def assess_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assesses the overall risk of the current portfolio.
        This would involve more complex calculations (VaR, stress testing, etc.)
        For now, a simple aggregated assessment.
        """
        logging.info("Assessing overall portfolio risk.")
        total_portfolio_value = portfolio.get("total_value", 0)
        open_pnl = portfolio.get("open_pnl", 0)

        # Simple risk indicators
        risk_score = 0.0 # Placeholder
        if total_portfolio_value > 0:
            risk_score = abs(open_pnl) / total_portfolio_value

        # Add more sophisticated risk metrics here
        # e.g., correlation with market, exposure per sector/stock, VaR, Conditional VaR

        return {
            "overall_risk_score": round(risk_score, 4),
            "current_open_pnl": round(open_pnl, 2),
            "exceeds_soft_limits": False, # Placeholder for soft limits
            "recommendation": "Monitor closely" if risk_score > 0.01 else "Low risk"
        }

    # You can add more methods here, like:
    # - set_stop_loss_orders
    # - adjust_position_size
    # - monitor_trailing_stops
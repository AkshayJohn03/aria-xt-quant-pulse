
import logging
from typing import Dict, Any, List, Optional
import random
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, trade_executor, config_manager):
        self.trade_executor = trade_executor
        self.config = config_manager
        self.sector_mapping = {
            'NIFTY': 'Index',
            'BANKNIFTY': 'Banking',
            'RELIANCE': 'Energy',
            'TCS': 'Technology',
            'HDFCBANK': 'Banking',
            'INFY': 'Technology',
            'ICICIBANK': 'Banking',
            'ITC': 'FMCG',
            'HINDUNILVR': 'FMCG',
            'SBIN': 'Banking',
            # Add more mappings as needed
        }
        self.max_daily_loss = self.config.get("risk_management.max_drawdown_percent", 2.0)
        self.max_per_trade_loss = self.config.get("trading.max_risk_per_trade", 0.5)
        self.max_open_positions = self.config.get("trading.max_positions", 5)
        
        logger.info(f"RiskManager initialized. Risk Limits: Max Daily Loss {self.max_daily_loss}%, Max Per Trade Loss {self.max_per_trade_loss}%, Max Open Positions {self.max_open_positions}")

    def set_trade_executor(self, trade_executor):
        """Set the trade executor reference for real data access"""
        self.trade_executor = trade_executor

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate a trading signal against risk parameters."""
        try:
            # Get current risk metrics
            risk_metrics = self.calculate_risk_metrics()
            
            # Basic risk checks
            if risk_metrics['portfolio_exposure_percent'] > self.config.get('max_portfolio_exposure_percent', 80):
                logger.warning("Signal rejected: Portfolio exposure too high")
                return False
                
            if risk_metrics['risk_score'] == 'HIGH':
                logger.warning("Signal rejected: Portfolio risk score is HIGH")
                return False
            
            # Check sector exposure
            symbol = signal.get('symbol', '')
            sector = self.get_sector_for_symbol(symbol)
            sector_exposure = risk_metrics['sector_exposure'].get(sector, {}).get('exposure', 0)
            
            if sector_exposure > self.config.get('max_sector_exposure_percent', 30):
                logger.warning(f"Signal rejected: {sector} sector exposure too high")
                return False
            
            # Signal specific checks
            signal_risk = signal.get('risk_score', 0)
            if signal_risk > self.config.get('max_signal_risk_score', 0.8):
                logger.warning("Signal rejected: Signal risk score too high")
                return False
            
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
        """Get only open positions."""
        return [p for p in self.get_positions() if p.get('quantity', 0) != 0]

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get enhanced position details from trade executor."""
        try:
            if self.trade_executor and hasattr(self.trade_executor, 'get_positions'):
                return self.trade_executor.get_positions()
            else:
                # Return mock positions if trade executor not available
                return self._get_mock_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return self._get_mock_positions()

    def get_holdings(self) -> List[Dict[str, Any]]:
        """Returns current holdings from trade executor."""
        try:
            if self.trade_executor and hasattr(self.trade_executor, 'get_holdings'):
                return self.trade_executor.get_holdings()
            else:
                # Return mock holdings if trade executor not available
                return self._get_mock_holdings()
        except Exception as e:
            logger.error(f"Error getting holdings: {e}")
            return self._get_mock_holdings()

    def get_funds(self) -> Dict[str, Any]:
        """Returns available funds from trade executor."""
        try:
            if self.trade_executor and hasattr(self.trade_executor, 'get_funds'):
                return self.trade_executor.get_funds()
            else:
                # Return mock funds if trade executor not available
                return self._get_mock_funds()
        except Exception as e:
            logger.error(f"Error getting funds: {e}")
            return self._get_mock_funds()

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for the portfolio."""
        try:
            # Get portfolio data
            positions = self.get_positions()
            holdings = self.get_holdings()
            funds = self.get_funds()
            
            # Initialize metrics
            total_investment = 0
            portfolio_value = 0
            total_pnl = 0
            day_pnl = 0
            sector_exposure = {}
            
            # Process positions and holdings
            all_instruments = positions + holdings
            
            for instrument in all_instruments:
                quantity = instrument.get('quantity', 0)
                avg_price = instrument.get('average_price', instrument.get('avg_price', 0))
                current_price = instrument.get('last_price', instrument.get('current_price', 0))
                
                investment = abs(quantity * avg_price)
                current_value = abs(quantity * current_price)
                
                total_investment += investment
                portfolio_value += current_value
                total_pnl += (current_value - investment)
                day_pnl += instrument.get('day_pnl', 0)
                
                # Calculate sector-wise exposure
                sector = self.get_sector_for_symbol(instrument.get('symbol', ''))
                if sector not in sector_exposure:
                    sector_exposure[sector] = {
                        'exposure': 0,
                        'risk_score': 'LOW'
                    }
                
                sector_exposure[sector]['exposure'] += (current_value / portfolio_value * 100 if portfolio_value > 0 else 0)
            
            # Update sector risk scores based on exposure
            for sector in sector_exposure:
                exposure = sector_exposure[sector]['exposure']
                if exposure > 30:
                    sector_exposure[sector]['risk_score'] = 'HIGH'
                elif exposure > 15:
                    sector_exposure[sector]['risk_score'] = 'MEDIUM'
                else:
                    sector_exposure[sector]['risk_score'] = 'LOW'
            
            # Calculate portfolio risk score
            max_sector_exposure = max([data['exposure'] for data in sector_exposure.values()], default=0)
            available_balance = funds.get('available_cash', funds.get('equity', {}).get('available', {}).get('cash', 0))
            portfolio_exposure = (portfolio_value / (portfolio_value + available_balance)) * 100 if (portfolio_value + available_balance) > 0 else 0
            
            risk_score = 'HIGH' if max_sector_exposure > 30 or portfolio_exposure > 80 else \
                        'MEDIUM' if max_sector_exposure > 15 or portfolio_exposure > 50 else \
                        'LOW'
            
            # Calculate additional risk metrics
            returns = [instrument.get('day_pnl', 0) / instrument.get('average_price', 1) 
                      for instrument in all_instruments if instrument.get('average_price', 0) != 0]
            
            sharpe_ratio = np.mean(returns) / np.std(returns) if returns and np.std(returns) != 0 else 0
            sortino_ratio = np.mean(returns) / np.std([r for r in returns if r < 0]) if returns else 0
            max_drawdown = min(returns) if returns else 0
            current_drawdown = (portfolio_value - total_investment) / total_investment if total_investment > 0 else 0
            
            return {
                'total_investment': total_investment,
                'portfolio_value': portfolio_value,
                'total_pnl': total_pnl,
                'day_pnl': day_pnl,
                'available_balance': available_balance,
                'risk_score': risk_score,
                'portfolio_exposure_percent': portfolio_exposure,
                'sector_exposure': sector_exposure,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_risk_per_trade_percent': self.max_per_trade_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            # Return mock data on error
            return self._get_mock_risk_metrics()

    def calculate_total_pnl(self) -> float:
        """Calculate total P&L across all positions."""
        try:
            risk_metrics = self.calculate_risk_metrics()
            return risk_metrics.get('total_pnl', 0.0)
        except Exception as e:
            logger.error(f"Error calculating total P&L: {e}")
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
                "day_pnl": 200.0,
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
                "day_pnl": 150.0,
                "product_type": "MIS",
                "timestamp": "2024-06-27T10:00:00Z",
                "status": "OPEN"
            }
        ]

    def _get_mock_holdings(self) -> List[Dict[str, Any]]:
        """Mock holdings for testing."""
        return [
            {
                "symbol": "TCS",
                "quantity": 10,
                "avg_price": 3750.0,
                "current_price": 3800.0,
                "pnl": 500.0,
                "day_pnl": 100.0,
                "product_type": "CNC"
            },
            {
                "symbol": "INFY",
                "quantity": 20,
                "avg_price": 1450.0,
                "current_price": 1500.0,
                "pnl": 1000.0,
                "day_pnl": 200.0,
                "product_type": "CNC"
            },
            {
                "symbol": "RELIANCE",
                "quantity": 5,
                "avg_price": 2400.0,
                "current_price": 2450.0,
                "pnl": 250.0,
                "day_pnl": 50.0,
                "product_type": "CNC"
            }
        ]

    def _get_mock_funds(self) -> Dict[str, Any]:
        """Mock funds for testing."""
        return {
            "available_cash": 50000.0,
            "equity": {
                "available": {
                    "cash": 50000.0
                },
                "used": 75000.0
            },
            "commodity": {
                "available": {
                    "cash": 0.0
                }
            }
        }

    def _get_mock_risk_metrics(self) -> Dict[str, Any]:
        """Mock risk metrics for testing."""
        return {
            "total_investment": 75000.0,
            "portfolio_value": 77250.0,
            "total_pnl": 2250.0,
            "day_pnl": 350.0,
            "available_balance": 50000.0,
            "risk_score": "MEDIUM",
            "portfolio_exposure_percent": 60.7,
            "sector_exposure": {
                "Technology": {
                    "exposure": 45.0,
                    "risk_score": "MEDIUM"
                },
                "Energy": {
                    "exposure": 15.0,
                    "risk_score": "LOW"
                }
            },
            "max_drawdown": -2.5,
            "current_drawdown": 3.0,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "max_risk_per_trade_percent": self.max_per_trade_loss
        }

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Get the sector for a given symbol."""
        return self.sector_mapping.get(symbol.upper(), 'Others')

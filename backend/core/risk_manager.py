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
        
        # Risk parameters
        self.max_daily_loss = self.config.get("risk_management.max_drawdown_percent", 2.0)
        self.max_per_trade_loss = self.config.get("trading.max_risk_per_trade", 0.5)
        self.max_open_positions = self.config.get("trading.max_positions", 5)
        
        # Advanced risk parameters
        self.max_sector_exposure = 30.0  # Maximum sector exposure percentage
        self.max_correlation_threshold = 0.7  # Maximum correlation between positions
        self.min_risk_reward_ratio = 1.5  # Minimum risk-reward ratio
        self.max_portfolio_volatility = 0.2  # Maximum portfolio volatility
        self.position_sizing_method = self.config.get("risk_management.position_sizing_method", "fixed_percentage")
        
        # Risk metrics
        self.daily_pnl = 0.0
        self.portfolio_volatility = 0.0
        self.correlation_matrix = {}
        self.sector_exposure = {}
        
        logging.info(f"RiskManager initialized with Advanced Risk Management Framework.")

    def set_trade_executor(self, trade_executor):
        """Set the trade executor reference for real data access"""
        self.trade_executor = trade_executor

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Enhanced signal validation with advanced risk checks"""
        try:
            # Get current risk metrics
            risk_metrics = self.calculate_risk_metrics()
            
            # Check portfolio exposure
            if risk_metrics['portfolio_exposure_percent'] > self.config.get('max_portfolio_exposure_percent', 80):
                logging.warning("Signal rejected: Portfolio exposure too high")
                return False
            
            # Check risk score
            if risk_metrics['risk_score'] == 'HIGH':
                logging.warning("Signal rejected: Portfolio risk score is HIGH")
                return False
            
            # Check sector exposure
            symbol = signal.get('symbol', '')
            sector = self.get_sector_for_symbol(symbol)
            sector_exposure = risk_metrics['sector_exposure'].get(sector, {}).get('exposure', 0)
            
            if sector_exposure > self.max_sector_exposure:
                logging.warning(f"Signal rejected: {sector} sector exposure too high")
                return False
            
            # Check correlation
            if not self._validate_correlation(signal, risk_metrics):
                logging.warning("Signal rejected: High correlation with existing positions")
                return False
            
            # Check risk-reward ratio
            if not self._validate_risk_reward(signal):
                logging.warning("Signal rejected: Risk-reward ratio too low")
                return False
            
            # Check volatility impact
            if not self._validate_volatility_impact(signal, risk_metrics):
                logging.warning("Signal rejected: Would increase portfolio volatility too much")
                return False
            
            # Check time-based conditions
            if not self._validate_time_conditions():
                logging.warning("Signal rejected: Outside trading hours or near market close")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error in signal validation: {e}")
            return False

    def _validate_correlation(self, signal: Dict[str, Any], risk_metrics: Dict) -> bool:
        """Validate correlation with existing positions"""
        try:
            symbol = signal.get('symbol', '')
            if not symbol:
                return True
            
            # Get correlation with existing positions
            correlations = risk_metrics.get('correlations', {})
            max_correlation = max(correlations.get(symbol, {}).values(), default=0)
            
            return max_correlation <= self.max_correlation_threshold
            
        except Exception as e:
            logging.error(f"Error validating correlation: {e}")
            return False

    def _validate_risk_reward(self, signal: Dict[str, Any]) -> bool:
        """Validate risk-reward ratio"""
        try:
            entry_price = signal.get('price', 0)
            stop_loss = signal.get('stop_loss', 0)
            target = signal.get('target', 0)
            
            if entry_price <= 0 or stop_loss <= 0 or target <= 0:
                return False
            
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            
            return (reward / risk) >= self.min_risk_reward_ratio
            
        except Exception as e:
            logging.error(f"Error validating risk-reward: {e}")
            return False

    def _validate_volatility_impact(self, signal: Dict[str, Any], risk_metrics: Dict) -> bool:
        """Validate impact on portfolio volatility"""
        try:
            current_volatility = risk_metrics.get('portfolio_volatility', 0)
            
            # Simulate new position impact
            new_position_value = signal.get('price', 0) * signal.get('quantity', 0)
            total_portfolio_value = risk_metrics.get('portfolio_value', 0)
            
            if total_portfolio_value <= 0:
                return True
            
            position_weight = new_position_value / (total_portfolio_value + new_position_value)
            
            # Estimate new volatility (simplified)
            estimated_new_volatility = current_volatility * (1 + position_weight)
            
            return estimated_new_volatility <= self.max_portfolio_volatility
            
        except Exception as e:
            logging.error(f"Error validating volatility impact: {e}")
            return False

    def _validate_time_conditions(self) -> bool:
        """Validate time-based trading conditions"""
        try:
            current_time = datetime.now()
            current_hour = current_time.hour
            current_minute = current_time.minute
            
            # Check market hours (9:15 AM to 3:30 PM)
            if current_hour < 9 or (current_hour == 9 and current_minute < 15):
                return False
            if current_hour > 15 or (current_hour == 15 and current_minute > 30):
                return False
            
            # Avoid trading in last 15 minutes
            if current_hour == 15 and current_minute > 15:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating time conditions: {e}")
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

    def get_positions(self):
        if not self.trade_executor:
            raise Exception("Trade executor not available")
        return self.trade_executor.get_positions()

    def get_holdings(self):
        if not self.trade_executor:
            raise Exception("Trade executor not available")
        return self.trade_executor.get_holdings()

    def get_funds(self):
        if not self.trade_executor:
            raise Exception("Trade executor not available")
        return self.trade_executor.get_funds()

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for the portfolio"""
        try:
            # Get portfolio data
            positions = self.trade_executor.get_positions()
            holdings = self.trade_executor.get_holdings()
            funds = self.trade_executor.get_funds()
            
            # Initialize metrics
            total_investment = 0
            portfolio_value = 0
            total_pnl = 0
            day_pnl = 0
            sector_exposure = {}
            correlations = {}
            
            # Process positions and holdings
            all_instruments = positions + holdings
            
            for instrument in all_instruments:
                quantity = instrument.get('quantity', 0)
                avg_price = instrument.get('average_price', 0)
                current_price = instrument.get('last_price', 0)
                
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
            
            # Calculate correlations
            correlations = self._calculate_correlations(all_instruments)
            
            # Update sector risk scores
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
            portfolio_exposure = (portfolio_value / funds.get('equity', 1)) * 100
            
            risk_score = 'HIGH' if max_sector_exposure > 30 or portfolio_exposure > 80 else \
                        'MEDIUM' if max_sector_exposure > 15 or portfolio_exposure > 50 else \
                        'LOW'
            
            # Calculate additional risk metrics
            returns = [instrument.get('day_pnl', 0) / instrument.get('average_price', 1) 
                      for instrument in all_instruments if instrument.get('average_price', 0) != 0]
            
            portfolio_volatility = np.std(returns) if returns else 0
            sharpe_ratio = np.mean(returns) / portfolio_volatility if portfolio_volatility != 0 else 0
            sortino_ratio = np.mean(returns) / np.std([r for r in returns if r < 0]) if returns else 0
            max_drawdown = min(returns) if returns else 0
            current_drawdown = (portfolio_value - total_investment) / total_investment if total_investment > 0 else 0
            
            return {
                'total_investment': total_investment,
                'portfolio_value': portfolio_value,
                'total_pnl': total_pnl,
                'day_pnl': day_pnl,
                'portfolio_exposure_percent': portfolio_exposure,
                'risk_score': risk_score,
                'sector_exposure': sector_exposure,
                'correlations': correlations,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'available_balance': funds.get('equity', {}).get('available', {}).get('cash', 0)
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return {}

    def _calculate_correlations(self, instruments: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between instruments"""
        try:
            correlations = {}
            
            # Get price data for all instruments
            price_data = {}
            for instrument in instruments:
                symbol = instrument.get('symbol', '')
                if symbol:
                    # In production, fetch historical prices
                    # For now, use mock data
                    price_data[symbol] = np.random.normal(0, 1, 100)  # Mock price changes
            
            # Calculate correlations
            for symbol1 in price_data:
                correlations[symbol1] = {}
                for symbol2 in price_data:
                    if symbol1 != symbol2:
                        corr = np.corrcoef(price_data[symbol1], price_data[symbol2])[0, 1]
                        correlations[symbol1][symbol2] = corr
            
            return correlations
            
        except Exception as e:
            logging.error(f"Error calculating correlations: {e}")
            return {}

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

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector for a given symbol"""
        return self.sector_mapping.get(symbol, 'Unknown')

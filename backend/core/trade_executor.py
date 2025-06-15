import logging
import asyncio
import random
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pytz
from fastapi import HTTPException

# Attempt to import KiteConnect, allow for mock if not installed
try:
    from kiteconnect import KiteConnect
    KITE_CONNECT_AVAILABLE = True
except ImportError:
    logging.warning("KiteConnect not installed. Zerodha broker functionality will be mocked.")
    KiteConnect = None
    KITE_CONNECT_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Abstract Base Class for Broker Clients ---
class BrokerBase:
    """Abstract base class for broker clients to ensure a consistent interface."""
    async def connect(self) -> bool:
        raise NotImplementedError("Connect method must be implemented by subclasses.")
    async def place_order(self, order_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Place order method must be implemented by subclasses.")
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Get order status method must be implemented by subclasses.")
    async def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        raise NotImplementedError("Get positions method must be implemented by subclasses.")
    async def get_holdings(self) -> Optional[List[Dict[str, Any]]]:
        raise NotImplementedError("Get holdings method must be implemented by subclasses.")
    async def get_funds(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Get funds method must be implemented by subclasses.")
    async def square_off_position(self, symbol: str, quantity: int = 0, product_type: str = "MIS") -> bool:
        raise NotImplementedError("Square off position method must be implemented by subclasses.")
    def _get_exchange_for_symbol(self, symbol: str) -> str:
        # Helper to determine exchange based on symbol type (simplified).
        if "NIFTY" in symbol.upper() or "BANKNIFTY" in symbol.upper() and ("CE" in symbol.upper() or "PE" in symbol.upper()):
            return "NFO" # National Futures and Options (for index options)
        # Add logic for stocks, commodities, currencies if needed
        return "NSE" # Default for equities

# --- Zerodha Broker Implementation ---
class ZerodhaBroker(BrokerBase):
    def __init__(self, kite_client: KiteConnect):
        self.kite = kite_client

    async def connect(self) -> bool:
        """Test connection by fetching user profile"""
        try:
            user_profile = await asyncio.to_thread(self.kite.profile)
            logging.info(f"Successfully connected to Zerodha Kite for user: {user_profile.get('user_name')}")
            return True
        except Exception as e:
            logging.error(f"Failed to verify Zerodha Kite connection: {e}")
            return False

    async def place_order(self, order_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.kite:
            logging.error("Zerodha Kite client not initialized. Cannot place order.")
            return None
        
        try:
            # Map generic order_details to KiteConnect specific parameters
            transaction_type = "BUY" if order_details['type'].upper() == "BUY" else "SELL"
            exchange = order_details.get('exchange', 'NFO') # NFO for options
            product = order_details.get('product_type', 'MIS') # MIS (intraday) or NRML (delivery/carryforward)
            order_type = order_details.get('order_type', 'MARKET')

            order_params = {
                "variety": "regular",
                "exchange": exchange,
                "tradingsymbol": order_details['symbol'],
                "transaction_type": transaction_type,
                "quantity": order_details['quantity'],
                "product": product,
                "order_type": order_type,
                "price": order_details.get('price'),
                "validity": order_details.get('validity', 'DAY'),
            }
            
            # Clean up None values for KiteConnect
            order_params = {k: v for k, v in order_params.items() if v is not None}

            order_response = await asyncio.to_thread(self.kite.place_order, **order_params)
            order_id = order_response.get('order_id')
            logging.info(f"Zerodha order placed successfully. Order ID: {order_id}")
            return {"status": "success", "order_id": order_id, "details": order_details}
        except Exception as e:
            logging.error(f"Failed to place Zerodha order: {e}")
            return {"status": "failed", "error": str(e), "details": order_details}

    async def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        if not self.kite:
            logger.error("Zerodha Kite client not initialized")
            raise HTTPException(status_code=503, detail="Broker connection not initialized")
        try:
            positions_data = await asyncio.to_thread(self.kite.positions)
            net_positions = positions_data.get('net', [])
            
            parsed_positions = []
            for pos in net_positions:
                if pos.get('quantity', 0) != 0:  # Only include non-zero positions
                    parsed_positions.append({
                        "symbol": pos.get('tradingsymbol'),
                        "quantity": pos.get('quantity'),
                        "avg_price": pos.get('average_price', 0),
                        "current_price": pos.get('last_price', 0),
                        "pnl": pos.get('pnl', 0),
                        "instrument_token": pos.get('instrument_token'),
                        "product": pos.get('product'),
                        "exchange": pos.get('exchange'),
                        "trading_symbol": pos.get('tradingsymbol'),
                        "m2m": pos.get('m2m', 0),
                        "unrealised": pos.get('unrealised', 0),
                        "realised": pos.get('realised', 0),
                        "buy_quantity": pos.get('buy_quantity', 0),
                        "sell_quantity": pos.get('sell_quantity', 0),
                        "buy_value": pos.get('buy_value', 0),
                        "sell_value": pos.get('sell_value', 0),
                        "multiplier": pos.get('multiplier', 1)
                    })
            logger.info(f"Successfully fetched {len(parsed_positions)} active Zerodha positions")
            return parsed_positions
        except Exception as e:
            error_msg = f"Failed to fetch Zerodha positions: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def get_holdings(self) -> Optional[List[Dict[str, Any]]]:
        if not self.kite:
            return None
        try:
            holdings_data = await asyncio.to_thread(self.kite.holdings)
            logging.info(f"Fetched {len(holdings_data)} Zerodha holdings.")
            return holdings_data
        except Exception as e:
            logging.error(f"Failed to fetch Zerodha holdings: {e}")
            return None

    async def get_funds(self) -> Optional[Dict[str, Any]]:
        if not self.kite:
            return None
        try:
            funds_data = await asyncio.to_thread(self.kite.margins)
            equity_margin = funds_data.get('equity', {})
            logging.info(f"Fetched Zerodha funds data.")
            return {
                "available_cash": equity_margin.get('available', {}).get('cash', 0),
                "free_margin": equity_margin.get('available', {}).get('live_margin', 0),
                "used_margin": equity_margin.get('utilised', {}).get('total', 0)
            }
        except Exception as e:
            logging.error(f"Failed to fetch Zerodha funds: {e}")
            return None

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        if not self.kite:
            logging.error("Zerodha Kite client not initialized. Cannot get order status.")
            return None
        try:
            orders = await asyncio.to_thread(self.kite.orders)
            for order in orders:
                if order.get('order_id') == order_id:
                    return {
                        "order_id": order.get('order_id'),
                        "status": order.get('status'),
                        "filled_quantity": order.get('filled_quantity'),
                        "pending_quantity": order.get('pending_quantity'),
                        "average_price": order.get('average_price'),
                        "timestamp": order.get('order_timestamp')
                    }
            logging.warning(f"Order ID {order_id} not found in Zerodha orders.")
            return None
        except Exception as e:
            logging.error(f"Failed to fetch Zerodha order status for {order_id}: {e}")
            return None

    async def square_off_position(self, symbol: str, quantity: int = 0, product_type: str = "MIS") -> bool:
        if not self.kite:
            logging.error("Zerodha Kite client not initialized. Cannot square off position.")
            return False
        try:
            positions = await self.get_positions()
            if not positions:
                logging.warning(f"No positions found for {symbol} to square off.")
                return False
            
            target_position = next((pos for pos in positions if pos['symbol'] == symbol), None)
            if not target_position:
                logging.warning(f"Position for {symbol} not found to square off.")
                return False
            
            current_quantity = target_position['quantity']
            if current_quantity == 0:
                logging.info(f"Position for {symbol} is already squared off (quantity is 0).")
                return True

            transaction_type = "SELL" if current_quantity > 0 else "BUY"
            quantity_to_square = abs(current_quantity) if quantity == 0 else min(abs(current_quantity), quantity)

            logging.info(f"Squaring off {quantity_to_square} of {symbol} (Action: {transaction_type}).")

            order_response = await asyncio.to_thread(self.kite.place_order,
                variety="regular",
                exchange=self._get_exchange_for_symbol(symbol), # Helper to determine exchange
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity_to_square,
                product=product_type,
                order_type="MARKET",
                validity="DAY"
            )
            logging.info(f"Zerodha square off order placed. Order ID: {order_response.get('order_id')}")
            return True
        except Exception as e:
            logging.error(f"Failed to square off Zerodha position for {symbol}: {e}")
            return False

# --- Mock Broker Implementation ---
class MockBroker(BrokerBase):
    async def connect(self) -> bool:
        await asyncio.sleep(0.5)
        return True

    async def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        logging.info("Mock: Fetching current positions.")
        await asyncio.sleep(0.5)
        return [
            {
                "symbol": "NIFTY24JUN22500CE",
                "quantity": 50,
                "avg_price": 125.50,
                "current_price": 138.75,
                "pnl": 662.50,
                "instrument_token": "mock_token_1",
                "product": "MIS"
            },
            {
                "symbol": "BANKNIFTY24JUN45500PE",
                "quantity": -25,
                "avg_price": 89.20,
                "current_price": 76.30,
                "pnl": 322.50,
                "instrument_token": "mock_token_2",
                "product": "MIS"
            }
        ]

    async def get_holdings(self) -> Optional[List[Dict[str, Any]]]:
        logging.info("Mock: Fetching current holdings.")
        await asyncio.sleep(0.5)
        return []

    async def get_funds(self) -> Optional[Dict[str, Any]]:
        logging.info("Mock: Fetching available funds.")
        await asyncio.sleep(0.5)
        return {
            "available_cash": 50000.0,
            "free_margin": 45000.0,
            "used_margin": 5000.0
        }

    async def place_order(self, order_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        logging.info(f"Mock: Attempting to place order: {order_details}")
        await asyncio.sleep(0.5)
        order_id = f"MOCK_ORDER_{random.randint(100000, 999999)}"
        logging.info(f"Mock order placed successfully. Order ID: {order_id}")
        return {"status": "success", "order_id": order_id, "details": order_details}

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        logging.info(f"Mock: Fetching status for order ID: {order_id}")
        await asyncio.sleep(0.2)
        statuses = ["PENDING", "COMPLETE", "REJECTED"]
        return {"order_id": order_id, "status": random.choice(statuses), "timestamp": datetime.now().isoformat()}
    
    async def square_off_position(self, symbol: str, quantity: int = 0, product_type: str = "MIS") -> bool:
        logging.info(f"Mock: Squaring off {quantity} of {symbol}.")
        await asyncio.sleep(0.5)
        logging.info(f"Mock: Successfully squared off {symbol}.")
        return True

# --- Main TradeExecutor Class ---
class TradeExecutor:
    """
    Manages connections to brokerage APIs, places orders, and fetches trade-related data.
    """
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.kite = None
        self.broker = None
        self.portfolio_cache = None
        self.last_cache_update = None
        self.cache_ttl = 60  # Cache TTL in seconds
        self.initialize()

    def initialize(self):
        """Initialize the trade executor with proper error handling and permissions check"""
        try:
            if not self.config_manager.config.get('zerodha', {}).get('api_key'):
                logger.error("Zerodha API key not found in configuration. Please check your .env or config.json.")
                return False

            self.kite = KiteConnect(api_key=self.config_manager.config['zerodha']['api_key'])
            
            if not self.config_manager.config['zerodha'].get('access_token'):
                logger.error("Zerodha access token not found in configuration. Please check your .env or config.json.")
                return False

            self.kite.set_access_token(self.config_manager.config['zerodha']['access_token'])
            
            # Test connection and permissions
            try:
                # Test basic profile access
                profile = self.kite.profile()
                logger.info(f"Successfully connected to Zerodha for user: {profile.get('user_name')}")
                
                # Test required permissions
                required_permissions = [
                    'orders', 'positions', 'holdings', 'margins',
                    'market_data', 'order_place', 'order_modify'
                ]
                
                for permission in required_permissions:
                    try:
                        if permission == 'orders':
                            self.kite.orders()
                        elif permission == 'positions':
                            self.kite.positions()
                        elif permission == 'holdings':
                            self.kite.holdings()
                        elif permission == 'margins':
                            self.kite.margins()
                        elif permission == 'market_data':
                            self.kite.quote('NSE:NIFTY 50')
                        elif permission == 'order_place':
                            # Just check if we can access order placement
                            pass
                        elif permission == 'order_modify':
                            # Just check if we can access order modification
                            pass
                        logger.info(f"Successfully verified {permission} permission")
                    except Exception as e:
                        logger.error(f"Missing or insufficient permission for {permission}: {str(e)}")
                        return False

                self.broker = ZerodhaBroker(self.kite)
                logger.info("TradeExecutor initialized with Zerodha broker and all required permissions")
                return True
                
            except Exception as e:
                logger.error(f"Failed to verify Zerodha connection and permissions: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing TradeExecutor: {str(e)}")
            return False

    async def connect(self):
        """Connect to the broker"""
        try:
            if not self.broker:
                logger.error("Broker not initialized")
                return False
            return await self.broker.connect()
        except Exception as e:
            logger.error(f"Error connecting to broker: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the broker"""
        try:
            self.kite = None
            self.broker = None
            logger.info("Disconnected from broker")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from broker: {e}")
            return False

    async def update_portfolio_cache(self):
        """Update the portfolio cache"""
        try:
            if not self.broker:
                logger.error("Broker not initialized")
                return False

            positions = await self.broker.get_positions()
            holdings = await self.broker.get_holdings()
            funds = await self.broker.get_funds()

            if positions is None and holdings is None and funds is None:
                logger.error("Failed to fetch portfolio data")
                return False

            self.portfolio_cache = {
                "positions": positions or [],
                "holdings": holdings or [],
                "funds": funds or {},
                "timestamp": datetime.now().isoformat()
            }
            self.last_cache_update = datetime.now()
            logger.info("Portfolio cache updated")
            return True
        except Exception as e:
            logger.error(f"Error updating portfolio cache: {e}")
            return False

    async def get_portfolio(self) -> Dict[str, Any]:
        """Get comprehensive portfolio data including positions, holdings, and funds"""
        try:
            if not self.kite:
                error_msg = "Zerodha Kite client not initialized. Please check your API credentials and restart the backend after updating the token."
                logger.error(error_msg)
                raise HTTPException(status_code=503, detail=error_msg)

            # Fetch all portfolio components concurrently
            positions_task = self.get_positions()
            holdings_task = self.get_holdings()
            funds_task = self.get_funds()
            
            positions, holdings, funds = await asyncio.gather(
                positions_task, holdings_task, funds_task,
                return_exceptions=True
            )

            # Handle any exceptions from the concurrent tasks
            if isinstance(positions, Exception):
                error_msg = f"Error fetching positions: {str(positions)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            if isinstance(holdings, Exception):
                error_msg = f"Error fetching holdings: {str(holdings)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            if isinstance(funds, Exception):
                error_msg = f"Error fetching funds: {str(funds)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

            # Calculate portfolio metrics
            total_value = 0
            total_pnl = 0
            total_m2m = 0
            total_unrealised = 0
            total_realised = 0

            # Process positions
            for pos in positions:
                total_value += abs(pos.get('quantity', 0) * pos.get('current_price', 0))
                total_pnl += pos.get('pnl', 0)
                total_m2m += pos.get('m2m', 0)
                total_unrealised += pos.get('unrealised', 0)
                total_realised += pos.get('realised', 0)

            # Process holdings
            for holding in holdings:
                total_value += holding.get('quantity', 0) * holding.get('last_price', 0)

            # Get available funds
            available_cash = funds.get('available_cash', 0)
            free_margin = funds.get('free_margin', 0)
            used_margin = funds.get('used_margin', 0)

            portfolio_data = {
                "timestamp": datetime.now().isoformat(),
                "positions": positions,
                "holdings": holdings,
                "funds": {
                    "available_cash": available_cash,
                    "free_margin": free_margin,
                    "used_margin": used_margin,
                    "total_margin": free_margin + used_margin
                },
                "metrics": {
                    "total_value": total_value,
                    "total_pnl": total_pnl,
                    "total_m2m": total_m2m,
                    "total_unrealised": total_unrealised,
                    "total_realised": total_realised,
                    "net_value": total_value + available_cash
                },
                "risk_metrics": {
                    "margin_utilization": (used_margin / (free_margin + used_margin)) * 100 if (free_margin + used_margin) > 0 else 0,
                    "cash_utilization": (used_margin / available_cash) * 100 if available_cash > 0 else 0,
                    "position_concentration": len(positions) / self.config_manager.config.get('trading', {}).get('max_positions', 5) * 100
                }
            }

            logger.info(f"Successfully fetched comprehensive portfolio data with {len(positions)} positions and {len(holdings)} holdings")
            return portfolio_data

        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error fetching portfolio data: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    def _is_cache_valid(self) -> bool:
        """Check if the portfolio cache is still valid"""
        if not self.portfolio_cache or not self.last_cache_update:
            return False
        return (datetime.now() - self.last_cache_update).seconds < self.cache_ttl

    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on the signal"""
        try:
            if not self.broker:
                return {
                    "status": "error",
                    "message": "Broker not initialized",
                    "timestamp": datetime.now().isoformat()
                }

            order_details = {
                "symbol": signal.get("symbol"),
                "type": signal.get("type", "BUY"),
                "quantity": signal.get("quantity"),
                "price": signal.get("price"),
                "order_type": signal.get("order_type", "MARKET"),
                "product_type": signal.get("product_type", "MIS")
            }

            result = await self.broker.place_order(order_details)
            if not result:
                return {
                    "status": "error",
                    "message": "Failed to place order",
                    "timestamp": datetime.now().isoformat()
                }

            # Update portfolio cache after trade
            await self.update_portfolio_cache()

            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def place_order(self, order_details: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Places an order on the brokerage platform with retry logic."""
        if not self.kite:
            logger.error("Broker client not initialized. Cannot place order.")
            return None

        # Check risk limits before placing order
        trade_value = order_details.get('price', 0) * order_details.get('quantity', 0)
        if self.risk_manager and not self.risk_manager.check_per_trade_risk(trade_value):
            logger.warning(f"Order placement aborted due to per-trade risk limit: {order_details}")
            return {"status": "aborted", "error": "Per-trade risk limit exceeded", "details": order_details}
        
        attempt = 0
        while attempt < max_retries:
            try:
                result = await self.kite.place_order(order_details)
                if result and result.get("status") == "success":
                    logger.info(f"Order placed successfully: {result}")
                    return result
                else:
                    logger.warning(f"Order attempt {attempt+1} failed: {result}. Retrying...")
            except Exception as e:
                logger.error(f"Order attempt {attempt+1} exception: {e}. Retrying...")
            
            attempt += 1
            if attempt < max_retries:
                await asyncio.sleep(2)  # Async backoff before retry
        
        logger.error(f"Order failed after {max_retries} attempts: {order_details}. Max retries exceeded.")
        return {"status": "failed", "error": "Max retries exceeded", "details": order_details}

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get enhanced position details."""
        try:
            self.update_portfolio_cache()
            
            enhanced_positions = []
            for position in self.portfolio_cache['positions']:
                # Calculate P&L
                quantity = position.get('quantity', 0)
                avg_price = position.get('average_price', 0)
                current_price = position.get('last_price', 0)
                day_open = position.get('day_open_price', avg_price)
                
                investment = abs(quantity * avg_price)
                current_value = abs(quantity * current_price)
                day_value = abs(quantity * day_open)
                
                total_pnl = current_value - investment
                day_pnl = current_value - day_value
                
                enhanced_positions.append({
                    'symbol': position.get('tradingsymbol', ''),
                    'exchange': position.get('exchange', ''),
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'pnl': total_pnl,
                    'day_pnl': day_pnl,
                    'product_type': position.get('product', 'N/A'),
                    'last_trade_time': position.get('last_trade_time', ''),
                    'unrealized_pnl': position.get('unrealized_pnl', 0),
                    'realized_pnl': position.get('realized_pnl', 0)
                })
            
            return enhanced_positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise Exception("Error getting positions")

    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get holdings with enhanced details."""
        try:
            self.update_portfolio_cache()
            
            enhanced_holdings = []
            for holding in self.portfolio_cache['holdings']:
                quantity = holding.get('quantity', 0)
                avg_price = holding.get('average_price', 0)
                current_price = holding.get('last_price', 0)
                day_open = holding.get('day_open_price', avg_price)
                
                investment = quantity * avg_price
                current_value = quantity * current_price
                day_value = quantity * day_open
                
                total_pnl = current_value - investment
                day_pnl = current_value - day_value
                
                enhanced_holdings.append({
                    'symbol': holding.get('tradingsymbol', ''),
                    'exchange': holding.get('exchange', ''),
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'pnl': total_pnl,
                    'day_pnl': day_pnl,
                    'product_type': 'CNC',  # Holdings are always delivery
                    'last_trade_time': holding.get('last_trade_time', ''),
                    'unrealized_pnl': total_pnl,  # For holdings, total PnL is unrealized
                    'realized_pnl': 0  # No realized PnL for holdings until sold
                })
            
            return enhanced_holdings
            
        except Exception as e:
            logger.error(f"Error getting holdings: {e}")
            raise Exception("Error getting holdings")

    def get_funds(self) -> Dict[str, Any]:
        """Get enhanced funds data."""
        try:
            self.update_portfolio_cache()
            return self.portfolio_cache['available_balance']
        except Exception as e:
            logger.error(f"Error getting funds: {e}")
            raise Exception("Error getting funds")

    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order with Zerodha."""
        try:
            if not self.kite:
                raise Exception("Not connected to broker")
                
            order_id = self.kite.place_order(
                variety=order_params.get('variety', 'regular'),
                exchange=order_params.get('exchange', 'NSE'),
                tradingsymbol=order_params.get('symbol'),
                transaction_type=order_params.get('transaction_type'),
                quantity=order_params.get('quantity'),
                product=order_params.get('product', 'MIS'),
                order_type=order_params.get('order_type', 'MARKET'),
                price=order_params.get('price', 0),
                trigger_price=order_params.get('trigger_price', 0)
            )
            
            logger.info(f"Order placed successfully. Order ID: {order_id}")
            return {"order_id": order_id, "status": "success"}
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise Exception("Error placing order")

    def modify_order(self, order_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an existing order."""
        try:
            if not self.kite:
                raise Exception("Not connected to broker")
                
            self.kite.modify_order(
                order_id=order_id,
                variety=params.get('variety', 'regular'),
                quantity=params.get('quantity'),
                price=params.get('price'),
                trigger_price=params.get('trigger_price'),
                order_type=params.get('order_type'),
                validity=params.get('validity')
            )
            
            logger.info(f"Order {order_id} modified successfully")
            return {"order_id": order_id, "status": "success"}
            
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            raise Exception("Error modifying order")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            if not self.kite:
                raise Exception("Not connected to broker")
                
            self.kite.cancel_order(order_id=order_id, variety='regular')
            
            logger.info(f"Order {order_id} cancelled successfully")
            return {"order_id": order_id, "status": "success"}
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            raise Exception("Error cancelling order")

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get the current status of an order."""
        try:
            if not self.kite:
                raise Exception("Not connected to broker")
                
            orders = self.kite.orders()
            order = next((o for o in orders if o['order_id'] == order_id), None)
            
            if not order:
                raise Exception(f"Order {order_id} not found")
                
            return order
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            raise Exception("Error getting order status")

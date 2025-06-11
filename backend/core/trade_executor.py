# D:\aria\aria-xt-quant-pulse\backend\core\trade_executor.py

import logging
import asyncio
import random
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Attempt to import KiteConnect, allow for mock if not installed
try:
    from kiteconnect import KiteConnect
    KITE_CONNECT_AVAILABLE = True
except ImportError:
    logging.warning("KiteConnect not installed. Zerodha broker functionality will be mocked.")
    KiteConnect = None
    KITE_CONNECT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            return None
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
                        "product": pos.get('product')
                    })
            logging.info(f"Fetched {len(parsed_positions)} active Zerodha positions.")
            return parsed_positions
        except Exception as e:
            logging.error(f"Failed to fetch Zerodha positions: {e}")
            return None

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
    def __init__(self, config: Dict[str, Any], risk_manager):
        self.config = config
        self.risk_manager = risk_manager
        self.connected = False
        self.broker_client: Optional[BrokerBase] = None

        logging.info("TradeExecutor initialized.")

    async def connect_to_broker(self) -> bool:
        """Connect to the specified brokerage API."""
        logging.info("Attempting to connect to brokerage API...")
        
        # Get Zerodha configuration
        api_key = self.config.get("apis", {}).get("zerodha", {}).get("api_key")
        access_token = self.config.get("apis", {}).get("zerodha", {}).get("access_token")

        if api_key and access_token and KITE_CONNECT_AVAILABLE:
            try:
                # Initialize KiteConnect client
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(access_token)
                
                # Create broker adapter
                self.broker_client = ZerodhaBroker(kite)
                self.connected = await self.broker_client.connect()
                
                if self.connected:
                    logging.info("Successfully connected to Zerodha Kite.")
                    return True
                else:
                    logging.error("Failed to connect to Zerodha Kite.")
                    
            except Exception as e:
                logging.error(f"Failed to initialize Zerodha connection: {e}")
        
        # Fallback to mock broker
        logging.warning("Using Mock Broker for trading operations.")
        self.broker_client = MockBroker()
        self.connected = await self.broker_client.connect()
        return self.connected

    async def place_order(self, order_details: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Places an order on the brokerage platform with retry logic."""
        if not self.broker_client:
            logging.error("Broker client not initialized. Cannot place order.")
            return None

        # Check risk limits before placing order
        trade_value = order_details.get('price', 0) * order_details.get('quantity', 0)
        if not self.risk_manager.check_per_trade_risk(trade_value):
            logging.warning(f"Order placement aborted due to per-trade risk limit: {order_details}")
            return {"status": "aborted", "error": "Per-trade risk limit exceeded", "details": order_details}
        
        attempt = 0
        while attempt < max_retries:
            try:
                result = await self.broker_client.place_order(order_details)
                if result and result.get("status") == "success":
                    logging.info(f"Order placed successfully: {result}")
                    return result
                else:
                    logging.warning(f"Order attempt {attempt+1} failed: {result}. Retrying...")
            except Exception as e:
                logging.error(f"Order attempt {attempt+1} exception: {e}. Retrying...")
            
            attempt += 1
            if attempt < max_retries:
                time.sleep(2) # Backoff before retry
        
        logging.error(f"Order failed after {max_retries} attempts: {order_details}. Max retries exceeded.")
        return {"status": "failed", "error": "Max retries exceeded", "details": order_details}

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        if not self.broker_client:
            return None
        return await self.broker_client.get_order_status(order_id)

    async def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        if not self.broker_client:
            return None
        return await self.broker_client.get_positions()

    async def get_holdings(self) -> Optional[List[Dict[str, Any]]]:
        if not self.broker_client:
            return None
        return await self.broker_client.get_holdings()

    async def get_funds(self) -> Optional[Dict[str, Any]]:
        if not self.broker_client:
            return None
        return await self.broker_client.get_funds()

    async def square_off_position(self, symbol: str, quantity: int = 0, product_type: str = "MIS") -> bool:
        if not self.broker_client:
            return False
        return await self.broker_client.square_off_position(symbol, quantity, product_type)

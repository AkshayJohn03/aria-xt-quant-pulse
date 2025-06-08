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
    def __init__(self, broker_config: Dict[str, Any]):
        self.broker_config = broker_config
        self.kite = None # Will be initialized by TradeExecutor's connect_to_broker

    async def connect(self, kite_client: KiteConnect) -> bool:
        """
        Connects to the Zerodha brokerage API using a pre-initialized KiteConnect client.
        This assumes the KiteConnect client is passed from TradeExecutor after initial setup.
        """
        self.kite = kite_client
        try:
            # Test connection by fetching user profile
            user_profile = self.kite.profile()
            logging.info(f"Successfully connected to Zerodha Kite for user: {user_profile.get('user_name')}")
            return True
        except Exception as e:
            logging.error(f"Failed to verify Zerodha Kite connection: {e}")
            self.kite = None # Reset client if connection test fails
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
                "variety": "regular", # Or "amo", "iceberg", "bo", "co"
                "exchange": exchange,
                "tradingsymbol": order_details['symbol'], # e.g., "NIFTY24MAY22500CE"
                "transaction_type": transaction_type,
                "quantity": order_details['quantity'],
                "product": product,
                "order_type": order_type,
                "price": order_details.get('price'), # Required for LIMIT/SL orders
                "trigger_price": order_details.get('trigger_price'), # Required for SL orders
                "squareoff": order_details.get('squareoff'), # For Bracket Orders
                "stoploss": order_details.get('stoploss'), # For Bracket Orders
                "trailing_stoploss": order_details.get('trailing_stoploss'), # For Bracket Orders
                "validity": order_details.get('validity', 'DAY'), # DAY or IOC
                "disclosed_quantity": order_details.get('disclosed_quantity'),
                "kitemf_quantity": order_details.get('kitemf_quantity')
            }
            
            # Clean up None values for KiteConnect
            order_params = {k: v for k, v in order_params.items() if v is not None}

            order_response = self.kite.place_order(**order_params)
            order_id = order_response.get('order_id')
            logging.info(f"Zerodha order placed successfully. Order ID: {order_id}")
            return {"status": "success", "order_id": order_id, "details": order_details}
        except Exception as e:
            logging.error(f"Failed to place Zerodha order: {e}")
            return {"status": "failed", "error": str(e), "details": order_details}

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        if not self.kite:
            logging.error("Zerodha Kite client not initialized. Cannot get order status.")
            return None
        try:
            orders = self.kite.orders()
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

    async def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        if not self.kite:
            logging.error("Zerodha Kite client not initialized. Cannot get positions.")
            return None
        try:
            positions_data = self.kite.positions()
            net_positions = positions_data.get('net', [])
            
            parsed_positions = []
            for pos in net_positions:
                if pos.get('exchange') == 'NFO' or pos.get('exchange') == 'NSE':
                    parsed_positions.append({
                        "symbol": pos.get('tradingsymbol'),
                        "quantity": pos.get('quantity'), # Positive for long, negative for short
                        "avg_price": pos.get('average_price'),
                        "current_price": pos.get('last_price'),
                        "pnl": pos.get('pnl'),
                        "instrument_token": pos.get('instrument_token'),
                        "product": pos.get('product') # MIS, NRML, etc.
                    })
            logging.info(f"Fetched {len(parsed_positions)} Zerodha positions.")
            return parsed_positions
        except Exception as e:
            logging.error(f"Failed to fetch Zerodha positions: {e}")
            return None

    async def get_holdings(self) -> Optional[List[Dict[str, Any]]]:
        if not self.kite:
            logging.error("Zerodha Kite client not initialized. Cannot get holdings.")
            return None
        try:
            holdings_data = self.kite.holdings()
            logging.info(f"Fetched {len(holdings_data)} Zerodha holdings.")
            return holdings_data
        except Exception as e:
            logging.error(f"Failed to fetch Zerodha holdings: {e}")
            return None

    async def get_funds(self) -> Optional[Dict[str, Any]]:
        if not self.kite:
            logging.error("Zerodha Kite client not initialized. Cannot get funds.")
            return None
        try:
            funds_data = self.kite.margins()
            equity_margin = funds_data.get('equity', {})
            logging.info(f"Fetched Zerodha funds data.")
            return {
                "available_cash": equity_margin.get('available', {}).get('cash'),
                "free_margin": equity_margin.get('available', {}).get('live_margin'),
                "used_margin": equity_margin.get('utilised', {}).get('total')
            }
        except Exception as e:
            logging.error(f"Failed to fetch Zerodha funds: {e}")
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

            order_response = self.kite.place_order(
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
        if random.random() > 0.1: # 90% success rate
            logging.info("Successfully connected to mock brokerage.")
            return True
        else:
            logging.error("Failed to connect to mock brokerage.")
            return False

    async def place_order(self, order_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        logging.info(f"Mock: Attempting to place order: {order_details}")
        await asyncio.sleep(0.5)
        if random.random() > 0.05: # 95% success rate
            order_id = f"MOCK_ORDER_{random.randint(100000, 999999)}"
            logging.info(f"Mock order placed successfully. Order ID: {order_id}")
            return {"status": "success", "order_id": order_id, "details": order_details}
        else:
            logging.error("Mock order failed.")
            return {"status": "failed", "error": "Simulated failure", "details": order_details}

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        logging.info(f"Mock: Fetching status for order ID: {order_id}")
        await asyncio.sleep(0.2)
        statuses = ["PENDING", "COMPLETE", "REJECTED"]
        return {"order_id": order_id, "status": random.choice(statuses), "timestamp": datetime.now().isoformat()}

    async def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        logging.info("Mock: Fetching current positions.")
        await asyncio.sleep(0.5)
        return [
            {"symbol": "NIFTY50", "quantity": 50, "avg_price": 22400.0, "current_price": 22450.0, "pnl": 2500.0, "instrument_token": "mock_token_1", "product": "MIS"},
            {"symbol": "BANKNIFTY", "quantity": -25, "avg_price": 47900.0, "current_price": 47850.0, "pnl": 1250.0, "instrument_token": "mock_token_2", "product": "MIS"}
        ]

    async def get_holdings(self) -> Optional[List[Dict[str, Any]]]:
        logging.info("Mock: Fetching current holdings.")
        await asyncio.sleep(0.5)
        return [
            {"tradingsymbol": "RELIANCE", "quantity": 10, "last_price": 2800.0, "pnl": 500.0},
            {"tradingsymbol": "TCS", "quantity": 5, "last_price": 4000.0, "pnl": -200.0}
        ]

    async def get_funds(self) -> Optional[Dict[str, Any]]:
        logging.info("Mock: Fetching available funds.")
        await asyncio.sleep(0.5)
        return {"available_cash": 50000.0, "free_margin": 45000.0, "used_margin": 5000.0}
    
    async def square_off_position(self, symbol: str, quantity: int = 0, product_type: str = "MIS") -> bool:
        logging.info(f"Mock: Squaring off {quantity} of {symbol}.")
        await asyncio.sleep(0.5)
        if random.random() > 0.1:
            logging.info(f"Mock: Successfully squared off {symbol}.")
            return True
        else:
            logging.error(f"Mock: Failed to square off {symbol}.")
            return False

# --- Main TradeExecutor Class ---
class TradeExecutor:
    """
    Manages connections to brokerage APIs, places orders, and fetches trade-related data.
    Uses a hybrid approach with real broker integration (e.g., Zerodha) and mock fallbacks.
    """
    def __init__(self, config: Dict[str, Any], risk_manager):
        self.config = config
        self.risk_manager = risk_manager
        self.broker_config = config.get("broker_api", {})
        self.connected = False
        self.broker_client: Optional[Union[KiteConnect, BrokerBase]] = None # Union for type hinting

        logging.info("TradeExecutor initialized.")

    async def connect_to_broker(self) -> bool:
        """
        Connects to the specified brokerage API.
        Handles initial authentication flow for Zerodha KiteConnect.
        """
        logging.info("Attempting to connect to brokerage API...")
        broker_name = self.broker_config.get("provider", "mock").lower()
        api_key = self.broker_config.get("api_key")
        api_secret = self.broker_config.get("api_secret")
        request_token = self.broker_config.get("request_token")
        access_token = self.broker_config.get("access_token")

        if broker_name == "zerodha" and KITE_CONNECT_AVAILABLE:
            if not api_key:
                logging.error("Zerodha API key is missing in configuration.")
                self.connected = False
                return False
            
            try:
                # Initialize KiteConnect client
                self.broker_client = KiteConnect(api_key=api_key)

                if access_token:
                    self.broker_client.set_access_token(access_token)
                    logging.info("KiteConnect access token set from config.")
                    # Verify connection
                    if await self.broker_client.profile(): # Use profile to test access
                        self.connected = True
                        logging.info("Successfully connected to Zerodha Kite with existing access token.")
                        return True
                    else:
                        logging.warning("Existing Zerodha access token invalid or expired. Attempting session generation.")
                        # Fall through to session generation if existing token fails
                
                # If no access token or it failed, try to generate a new session
                if request_token and api_secret:
                    try:
                        data = self.broker_client.generate_session(request_token, api_secret=api_secret)
                        self.broker_config['access_token'] = data['access_token'] # Update config in memory
                        # TODO: You MUST persist this new access token to your .env or config.json file!
                        logging.info(f"New Zerodha Kite session generated. Access Token: {data['access_token'][:8]}... Please update your config.")
                        self.connected = True
                        return True
                    except Exception as gs_e:
                        logging.error(f"Failed to generate Zerodha Kite session: {gs_e}. Ensure request_token is valid.")
                        self.connected = False
                        return False
                else:
                    logging.error("Zerodha: request_token or api_secret missing for session generation. Cannot connect.")
                    self.connected = False
                    return False

            except Exception as e:
                logging.error(f"Failed to initialize KiteConnect client or connect to Zerodha: {e}")
                self.connected = False
                return False
        
        # If not Zerodha or KiteConnect not available, use mock
        logging.warning(f"Using Mock Broker: Broker '{broker_name}' not supported or KiteConnect not available.")
        self.broker_client = MockBroker()
        self.connected = await self.broker_client.connect()
        return self.connected

    def _get_active_broker_client(self) -> BrokerBase:
        """Helper to get the currently active broker client (real or mock)."""
        if self.connected and self.broker_client:
            if isinstance(self.broker_client, KiteConnect):
                # Wrap KiteConnect client in ZerodhaBroker adapter if it's the raw client
                return ZerodhaBroker(self.broker_client)
            return self.broker_client # Already an instance of BrokerBase (e.g., MockBroker or ZerodhaBroker)
        
        # Fallback to MockBroker if not connected or client is not set up
        logging.warning("Broker client not active or connected. Falling back to MockBroker.")
        return MockBroker()

    async def place_order(self, order_details: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Places an order on the brokerage platform with retry logic and risk checks.
        """
        broker = self._get_active_broker_client()

        # Check risk limits before placing order
        trade_value = order_details.get('price', 0) * order_details.get('quantity', 0)
        if not self.risk_manager.check_per_trade_risk(trade_value):
            logging.warning(f"Order placement aborted due to per-trade risk limit: {order_details}")
            return {"status": "aborted", "error": "Per-trade risk limit exceeded", "details": order_details}
        
        attempt = 0
        while attempt < max_retries:
            try:
                result = await broker.place_order(order_details)
                if result and result.get("status") == "success":
                    # TODO: Add notification hook here (e.g., TelegramNotifier.send_notification)
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
        # TODO: Add persistent notification for failure (e.g., TelegramNotifier.send_notification)
        return {"status": "failed", "error": "Max retries exceeded", "details": order_details}

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        return await self._get_active_broker_client().get_order_status(order_id)

    async def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        return await self._get_active_broker_client().get_positions()

    async def get_holdings(self) -> Optional[List[Dict[str, Any]]]:
        return await self._get_active_broker_client().get_holdings()

    async def get_funds(self) -> Optional[Dict[str, Any]]:
        return await self._get_active_broker_client().get_funds()

    async def square_off_position(self, symbol: str, quantity: int = 0, product_type: str = "MIS") -> bool:
        return await self._get_active_broker_client().square_off_position(symbol, quantity, product_type)

    # You can add other common methods here like cancel_order, modify_order, etc.
    # def _get_exchange_for_symbol(self, symbol: str) -> str:
    #     """Helper to determine exchange based on symbol type (simplified)."""
    #     # This logic is now part of BrokerBase, which ZerodhaBroker inherits
    #     return BrokerBase()._get_exchange_for_symbol(symbol)
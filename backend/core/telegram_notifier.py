import logging
import httpx # Used for async HTTP requests
from typing import Dict, Any # ADD THIS IMPORT
from core.config_manager import ConfigManager # Import ConfigManager

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, config_manager: ConfigManager): # Accept config_manager directly
        self.config_manager = config_manager
        self.bot_token = self.config_manager.get("apis.telegram.bot_token")
        self.chat_id = self.config_manager.get("apis.telegram.chat_id")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        
        if not self.bot_token or not self.chat_id:
            logger.warning("TelegramNotifier initialized without bot_token or chat_id. Notifications will not work.")
        else:
            logger.info("TelegramNotifier initialized.")

    async def send_message(self, message: str) -> bool:
        """Sends a text message to the configured Telegram chat."""
        if not self.base_url or not self.chat_id:
            logger.error("Cannot send Telegram message: Bot token or chat ID is missing.")
            return False
        
        endpoint = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML" # Optional: allows basic HTML formatting
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=payload, timeout=5)
                response.raise_for_status() # Raise an exception for HTTP errors
                data = response.json()
                if data.get("ok"):
                    logger.info(f"Telegram message sent: {message}")
                    return True
                else:
                    logger.error(f"Failed to send Telegram message: {data.get('description', 'Unknown error')}")
                    return False
        except httpx.RequestError as e:
            logger.error(f"Telegram message request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending Telegram message: {e}")
            return False

    async def send_trade_notification(self, signal: Dict[str, Any], trade_result: Dict[str, Any]) -> bool:
        """Sends a formatted notification for a trade entry."""
        if not self.config_manager.get("notifications.trade_entry"):
            return False # Notifications disabled

        success = trade_result.get("success", False)
        status_emoji = "✅" if success else "❌"
        
        message = f"{status_emoji} <b>TRADE {'EXECUTED' if success else 'FAILED'}</b>\n"
        message += f"<b>Symbol:</b> {signal.get('symbol')} ({signal.get('option_type')})\n"
        message += f"<b>Direction:</b> {signal.get('type')}\n"
        message += f"<b>Price:</b> {signal.get('price'):.2f}\n"
        message += f"<b>Quantity:</b> {signal.get('quantity')}\n"
        message += f"<b>Confidence:</b> {signal.get('confidence'):.1f}%\n"
        message += f"<b>SL:</b> {signal.get('stop_loss'):.2f} | <b>Target:</b> {signal.get('target'):.2f}\n"
        message += f"<b>Reasoning:</b> {signal.get('reasoning')}\n"
        if not success:
            message += f"<b>Error:</b> {trade_result.get('error', 'N/A')}\n"

        return await self.send_message(message)

    async def test_connection(self) -> bool:
        """Tests the Telegram bot API connection by getting bot info."""
        if not self.bot_token:
            logger.warning("Telegram bot token not configured for connection test.")
            return False

        endpoint = f"https://api.telegram.org/bot{self.bot_token}/getMe"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint, timeout=5)
                response.raise_for_status()
                data = response.json()
                if data.get("ok") and data.get("result"):
                    bot_name = data["result"].get("first_name", "Telegram Bot")
                    logger.info(f"Telegram connection test: SUCCESS. Bot Name: {bot_name}")
                    return True
                else:
                    logger.error(f"Telegram connection test failed: {data.get('description', 'Unknown error')}")
                    return False
        except httpx.RequestError as e:
            logger.error(f"Telegram connection test request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during Telegram connection test: {e}")
            return False
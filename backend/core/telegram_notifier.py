# D:\aria\aria-xt-quant-pulse\backend\core\telegram_notifier.py

import requests
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TelegramNotifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_token = config.get("telegram", {}).get("bot_token")
        self.chat_id = config.get("telegram", {}).get("chat_id")

        if not self.bot_token or not self.chat_id:
            logging.warning("TelegramNotifier initialized without bot_token or chat_id. Notifications will not work.")
            self.is_configured = False
        else:
            self.is_configured = True
            logging.info("TelegramNotifier initialized successfully.")

    def send_message(self, message: str) -> bool:
        """
        Sends a message to the configured Telegram chat.
        """
        if not self.is_configured:
            logging.error("Telegram Notifier is not configured. Cannot send message.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "MarkdownV2" # Optional: allows markdown formatting in messages
        }

        try:
            # Telegram API can be sensitive to rapid calls; consider rate limiting if used extensively
            response = requests.post(url, data=data, timeout=5)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            logging.info(f"Telegram message sent successfully: {message[:50]}...")
            return True
        except requests.exceptions.HTTPError as e:
            logging.error(f"Telegram HTTP error: {e.response.status_code} - {e.response.text}")
            return False
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Telegram connection error: {e}")
            return False
        except requests.exceptions.Timeout as e:
            logging.error(f"Telegram request timed out: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logging.error(f"An unexpected Telegram request error occurred: {e}")
            return False
        except Exception as e:
            logging.error(f"Failed to send Telegram message due to an unexpected error: {e}")
            return False

    # You might add methods for:
    # - send_photo()
    # - send_document()
    # - send_alert() with specific formatting
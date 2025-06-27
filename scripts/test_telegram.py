import asyncio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from core.config_manager import ConfigManager
from core.telegram_notifier import TelegramNotifier

async def main():
    config = ConfigManager()
    notifier = TelegramNotifier(config)
    text = "Hi from Aria XT Quant Pulse! This is a test message."
    result = await notifier.send_test_message(text)
    print("Test message sent successfully." if result else "Failed to send test message.")

if __name__ == "__main__":
    asyncio.run(main()) 
import os
from pathlib import Path
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

# Your request token from the URL
REQUEST_TOKEN = "kFFFp9xu0rOvR9MROGCUMpfxf2FHRj8y"  # The token from your URL

# Retrieve API credentials
API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")
ACCESS_TOKEN_ENV_VAR = "ZERODHA_ACCESS_TOKEN"

kite = KiteConnect(api_key=API_KEY)

try:
    # Generate session
    session_data = kite.generate_session(REQUEST_TOKEN, api_secret=API_SECRET)
    access_token = session_data["access_token"]

    logger.info(f"[✅] Access Token Fetched: {access_token[:8]}...")

    # Update the .env file
    set_key(dotenv_path, ACCESS_TOKEN_ENV_VAR, access_token)
    logger.info(f"[✅] {ACCESS_TOKEN_ENV_VAR} updated successfully in .env file.")
    
except Exception as e:
    logger.error(f"❌ Failed to generate access token: {e}")
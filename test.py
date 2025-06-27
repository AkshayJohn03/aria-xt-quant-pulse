import os
import logging
from dotenv import load_dotenv
from kiteconnect import KiteConnect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path='./backend/.env')

api_key = os.getenv("ZERODHA_API_KEY")
access_token = os.getenv("ZERODHA_ACCESS_TOKEN")

if not api_key or not access_token:
    logger.error("Error: ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN not found in .env")
    exit(1)

try:
    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    logger.info("KiteConnect client initialized. Testing connection and permissions...")

    # Test basic profile access
    profile = kite.profile()
    logger.info(f"✅ Successfully connected to Zerodha for user: {profile.get('user_name')}")

    # Test required permissions
    required_permissions = [
        'orders', 'positions', 'holdings', 'margins',
        'market_data', 'order_place', 'order_modify'
    ]

    for permission in required_permissions:
        try:
            if permission == 'orders':
                kite.orders()
            elif permission == 'positions':
                kite.positions()
            elif permission == 'holdings':
                kite.holdings()
            elif permission == 'margins':
                kite.margins()
            elif permission == 'market_data':
                kite.quote('NSE:NIFTY 50')
            logger.info(f"✅ Successfully verified {permission} permission")
        except Exception as e:
            logger.error(f"❌ Missing or insufficient permission for {permission}: {str(e)}")
            exit(1)

    # Fetch all instruments
    logger.info("\nFetching all instruments. This may take a moment...")
    all_instruments = kite.instruments()

    nifty_token = None
    sensex_token = None

    logger.info("\nSearching for NIFTY 50 and SENSEX instrument tokens...")
    for inst in all_instruments:
        if inst.get('exchange') == 'NSE' and inst.get('tradingsymbol') == 'NIFTY 50':
            nifty_token = inst.get('instrument_token')
            logger.info(f"✅ Found NIFTY 50: Instrument Token = {nifty_token}, Trading Symbol = {inst.get('tradingsymbol')}")
        if inst.get('exchange') == 'BSE' and inst.get('tradingsymbol') == 'SENSEX':
            sensex_token = inst.get('instrument_token')
            logger.info(f"✅ Found SENSEX: Instrument Token = {sensex_token}, Trading Symbol = {inst.get('tradingsymbol')}")

        if nifty_token and sensex_token:
            break

    if nifty_token and sensex_token:
        logger.info(f"\n=== IMPORTANT: Update your config.json with these tokens ===")
        logger.info(f"\"nifty_instrument_token\": \"{nifty_token}\",")
        logger.info(f"\"sensex_instrument_token\": \"{sensex_token}\"")
    else:
        logger.error("\n❌ Could not find both NIFTY 50 and SENSEX instrument tokens.")
        logger.error("Please ensure correct trading symbols and permissions.")
        exit(1)

    # Test market data access
    logger.info("\nTesting market data access...")
    try:
        nifty_quote = kite.quote(f"NSE:NIFTY 50")
        logger.info(f"✅ Successfully fetched NIFTY 50 quote: {nifty_quote}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch NIFTY 50 quote: {str(e)}")
        exit(1)

    logger.info("\n✅ All tests completed successfully!")

except Exception as e:
    logger.error(f"❌ An error occurred: {str(e)}")
    logger.error("Please ensure your API key and access token are correct and active.")
    exit(1)
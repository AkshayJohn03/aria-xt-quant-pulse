import os
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv(dotenv_path='./backend/.env') # Adjust path if your .env is elsewhere

api_key = os.getenv("ZERODHA_API_KEY")
access_token = os.getenv("ZERODHA_ACCESS_TOKEN")

if not api_key or not access_token:
    print("Error: ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN not found in .env")
    exit()

try:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    print("KiteConnect client initialized. Fetching instruments...")

    # Fetch all instruments. This can take a moment.
    all_instruments = kite.instruments()

    nifty_token = None
    sensex_token = None

    print("\nSearching for NIFTY 50 and SENSEX instrument tokens...")
    for inst in all_instruments:
        if inst.get('exchange') == 'NSE' and inst.get('tradingsymbol') == 'NIFTY 50':
            nifty_token = inst.get('instrument_token')
            print(f"Found NIFTY 50: Instrument Token = {nifty_token}, Trading Symbol = {inst.get('tradingsymbol')}")
        if inst.get('exchange') == 'BSE' and inst.get('tradingsymbol') == 'SENSEX':
            sensex_token = inst.get('instrument_token')
            print(f"Found SENSEX: Instrument Token = {sensex_token}, Trading Symbol = {inst.get('tradingsymbol')}")

        if nifty_token and sensex_token:
            break # Found both, can exit loop

    if nifty_token and sensex_token:
        print(f"\n--- IMPORTANT: Update your config.json with these tokens ---")
        print(f"\"nifty_instrument_token\": \"{nifty_token}\",")
        print(f"\"sensex_instrument_token\": \"{sensex_token}\"")
    else:
        print("\nCould not find both NIFTY 50 and SENSEX instrument tokens. Please ensure correct trading symbols.")
        print("You might need to adjust 'tradingsymbol' values if they differ in your Kite account.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Ensure your API key and access token are correct and active.")
import os
import webbrowser
import json
import logging
from pathlib import Path # Import Path for robust path handling
from dotenv import load_dotenv, set_key # Import load_dotenv and set_key

# Import Flask and KiteConnect
from flask import Flask, request
from kiteconnect import KiteConnect

# Configure logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
# CORRECTED PATH: .env is one level up from 'utils' directory
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

# --- Retrieve API credentials from environment variables ---
API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")
ACCESS_TOKEN_ENV_VAR = "ZERODHA_ACCESS_TOKEN" # The name of the env var to update

# --- Initialize Flask App and KiteConnect ---
app = Flask(__name__)

# Ensure API_KEY and API_SECRET are loaded before initializing KiteConnect
if not API_KEY or not API_SECRET:
    logger.error(f"ZERODHA_API_KEY and ZERODHA_API_SECRET must be set in your .env file. Looked in: {dotenv_path}")
    # Exit or raise error, as we cannot proceed without these
    import sys
    sys.exit(1)

kite = KiteConnect(api_key=API_KEY)

@app.route("/")
def home():
    """
    Handles the callback from Zerodha after successful login.
    Extracts the request_token and generates the access_token.
    """
    request_token = request.args.get("request_token")
    if not request_token:
        logger.error("No request_token found in URL after redirect.")
        return "<h2>❌ Login Failed!</h2><p>No request_token found in URL.</p>"

    try:
        # Generate session using the request token and API secret
        session_data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = session_data["access_token"]

        logger.info(f"[✅] Access Token Fetched: {access_token[:8]}...") # Log partial token for security

        # --- Update the .env file with the new access token ---
        # This will either update an existing line or add a new one
        set_key(dotenv_path, ACCESS_TOKEN_ENV_VAR, access_token)
        logger.info(f"[✅] {ACCESS_TOKEN_ENV_VAR} updated successfully in .env file.")

        return f"<h2>✅ Access token saved successfully!</h2>" \
               f"<p>Token: <code>{access_token}</code></p>" \
               f"<p>You may close this window. Your <code>.env</code> file has been updated.</p>" \
               f"<p><b>Important:</b> Restart your main backend application (Uvicorn) to use the new token!</p>"
    except Exception as e:
        logger.error(f"❌ Failed to generate access token: {e}")
        return f"<h2>❌ Failed to generate token:</h2><p>{e}</p>"

if __name__ == "__main__":
    logger.info("Starting Zerodha Token Updater (Flask-based)...")
    
    # Get the login URL from KiteConnect
    login_url = kite.login_url()
    logger.info(f"[🔗] Open this URL in your browser to login with Zerodha: {login_url}")
    
    # Open the URL in the default web browser
    webbrowser.open(login_url)
    
    # Run the Flask app
    # Zerodha redirects to localhost:PORT, so this must be the port Zerodha expects.
    # Default is typically 80, but can be configured in your KiteConnect app settings.
    # Ensure this port is open on your firewall if needed.
    logger.info("Flask server starting on http://localhost:80/. Waiting for Zerodha callback...")
    app.run(port=80, debug=False) # debug=False for production use

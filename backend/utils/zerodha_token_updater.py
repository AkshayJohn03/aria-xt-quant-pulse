import os
import webbrowser
import json
import logging
from pathlib import Path
from dotenv import load_dotenv, set_key

# Import Flask and KiteConnect
from flask import Flask, request
from kiteconnect import KiteConnect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

# Retrieve API credentials
API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")
ACCESS_TOKEN_ENV_VAR = "ZERODHA_ACCESS_TOKEN"

app = Flask(__name__)

if not API_KEY or not API_SECRET:
    logger.error(f"ZERODHA_API_KEY and ZERODHA_API_SECRET must be set in your .env file. Looked in: {dotenv_path}")
    import sys
    sys.exit(1)

kite = KiteConnect(api_key=API_KEY)

@app.route("/")
def home():
    request_token = request.args.get("request_token")
    if not request_token:
        logger.error("No request_token found in URL after redirect.")
        return "<h2>❌ Login Failed!</h2><p>No request_token found in URL.</p>"

    try:
        session_data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = session_data["access_token"]

        logger.info(f"[✅] Access Token Fetched: {access_token[:8]}...")

        # Update the .env file
        set_key(dotenv_path, ACCESS_TOKEN_ENV_VAR, access_token)
        logger.info(f"[✅] {ACCESS_TOKEN_ENV_VAR} updated successfully in .env file.")

        return f"""
        <h2>✅ Access token saved successfully!</h2>
        <p>Token: <code>{access_token}</code></p>
        <p>You may close this window. Your <code>.env</code> file has been updated.</p>
        <p><b>Important:</b> Restart your main backend application (Uvicorn) to use the new token!</p>
        """
    except Exception as e:
        logger.error(f"❌ Failed to generate access token: {e}")
        return f"<h2>❌ Failed to generate token:</h2><p>{e}</p>"

if __name__ == "__main__":
    logger.info("Starting Zerodha Token Updater...")
    
    # Get the login URL
    login_url = kite.login_url()
    logger.info(f"[🔗] Open this URL in your browser to login with Zerodha: {login_url}")
    
    # Open the URL in browser
    webbrowser.open(login_url)
    
    # Run Flask app on port 80 with sudo
    logger.info("Flask server starting on http://localhost/. Waiting for Zerodha callback...")
    try:
        app.run(host='0.0.0.0', port=80, debug=False)
    except PermissionError:
        logger.info("Permission denied for port 80. Trying with sudo...")
        import subprocess
        subprocess.run(['sudo', 'python3', __file__])
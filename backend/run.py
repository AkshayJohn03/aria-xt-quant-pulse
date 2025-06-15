
#!/usr/bin/env python3
"""
Startup script for Aria XT Quant Pulse Backend
"""

import sys
import os
import subprocess
import logging

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages"""
    try:
        logger.info("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False
    return True

def main():
    """Main entry point"""
    logger.info("Starting Aria XT Quant Pulse Backend...")
    
    # Check if we're in the backend directory
    if not os.path.exists("app.py"):
        logger.error("app.py not found. Please run this script from the backend directory.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements. Exiting.")
        sys.exit(1)
    
    # Start the application
    try:
        logger.info("Starting FastAPI application...")
        from app import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

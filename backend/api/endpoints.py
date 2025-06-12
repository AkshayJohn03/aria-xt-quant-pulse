from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

# Import shared instances
from core import instances
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection - using shared instances
def get_config_manager_dep():
    return instances.config_manager

def get_model_interface_dep():
    return instances.model_interface

def get_risk_manager_dep():
    return instances.risk_manager

def get_trade_executor_dep():
    return instances.trade_executor

def create_api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Helper function to create consistent API responses"""
    response_data = {
        "success": success,
        "data": data,
        "error": error
    }
    return JSONResponse(content=response_data, status_code=status_code)

# --- Configuration endpoints ---
@router.get("/config")
async def get_config(config: ConfigManager = Depends(get_config_manager_dep)):
    """Get current configuration"""
    try:
        return create_api_response(True, config.config)
    except Exception as e:
        logger.error(f"Error fetching configuration: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.post("/config")
async def update_config(updates: Dict[str, Any], config: ConfigManager = Depends(get_config_manager_dep)):
    """Update configuration"""
    try:
        success = config.update_config(updates)
        if success:
            return create_api_response(True, {"message": "Configuration updated successfully"})
        else:
            return create_api_response(False, error="Failed to update configuration", status_code=500)
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/config/validate")
async def validate_config(config: ConfigManager = Depends(get_config_manager_dep)):
    """Validate current configuration"""
    try:
        is_valid = config.validate_config()
        if is_valid:
            return create_api_response(True, {"message": "Configuration is valid"})
        else:
            return create_api_response(False, error="Configuration validation failed", status_code=400)
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return create_api_response(False, error=str(e), status_code=500)

# --- System status endpoints ---
@router.get("/status")
async def get_status():
    """Get current system status"""
    try:
        return create_api_response(True, instances.system_status)
    except Exception as e:
        logger.error(f"Error fetching system status: {e}")
        return create_api_response(False, error=str(e), status_code=500)

# --- Trading endpoints ---
@router.get("/positions")
async def get_positions(risk_manager = Depends(get_risk_manager_dep)):
    """Get current positions"""
    try:
        positions = risk_manager.get_positions()
        return create_api_response(True, positions)
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/holdings")
async def get_holdings(risk_manager = Depends(get_risk_manager_dep)):
    """Get current holdings"""
    try:
        holdings = risk_manager.get_holdings()
        return create_api_response(True, holdings)
    except Exception as e:
        logger.error(f"Error fetching holdings: {e}")
        return create_api_response(False, error=str(e), status_code=500)

@router.get("/funds")
async def get_funds(risk_manager = Depends(get_risk_manager_dep)):
    """Get available funds"""
    try:
        funds = risk_manager.get_funds()
        return create_api_response(True, funds)
    except Exception as e:
        logger.error(f"Error fetching funds: {e}")
        return create_api_response(False, error=str(e), status_code=500)

# --- Model endpoints ---
@router.get("/models/status")
async def get_model_status(model_interface = Depends(get_model_interface_dep)):
    """Get status of all models"""
    try:
        status = {
            model: model_interface.is_model_loaded(model)
            for model in ["aria_lstm", "xgboost", "finbert", "prophet"]
        }
        return create_api_response(True, status)
    except Exception as e:
        logger.error(f"Error fetching model status: {e}")
        return create_api_response(False, error=str(e), status_code=500)

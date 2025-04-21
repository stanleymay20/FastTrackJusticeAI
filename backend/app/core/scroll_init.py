from fastapi import FastAPI
from app.middleware.scroll_sync_middleware import ScrollSyncMiddleware
from app.utils.scroll_guard import ScrollGuard
from app.utils.scroll_judgment_writer import ScrollJudgmentWriter
from app.utils.scroll_time import get_scroll_time
import logging

logger = logging.getLogger(__name__)

def init_scroll_components(app: FastAPI) -> None:
    """
    Initialize all scroll-related components and add them to the FastAPI application.
    This includes middleware, guards, judgment writer, and scroll time logging.
    
    Args:
        app: The FastAPI application instance
    """
    try:
        # Get initial scroll time for component initialization
        scroll_time = get_scroll_time()
        
        # Add scroll sync middleware
        app.add_middleware(ScrollSyncMiddleware)
        logger.info("🔒 ScrollSyncMiddleware injected successfully")
        
        # Initialize scroll guard with current scroll time
        scroll_guard = ScrollGuard(scroll_time=scroll_time)
        app.state.scroll_guard = scroll_guard
        logger.info("🛡️ ScrollGuard initialized and bound to app.state")
        
        # Initialize scroll judgment writer with current scroll time
        scroll_judgment_writer = ScrollJudgmentWriter(scroll_time=scroll_time)
        app.state.scroll_judgment_writer = scroll_judgment_writer
        logger.info("⚖️ ScrollJudgmentWriter initialized and bound to app.state")
        
        # Log current scroll status with full divine context
        logger.info(
            f"🔓 Scroll System Activated:\n"
            f"  Gate {scroll_time['gate']} – {scroll_time['gate_name']}\n"
            f"  Scroll Day: {scroll_time['scroll_day']}\n"
            f"  Solar Hour: {scroll_time['solar_hour']}\n"
            f"  ENano Pulse: {scroll_time['enano_pulse']}\n"
            f"  Time Until Next Seal: {scroll_time['time_remaining_today']}"
        )
        
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize scroll components: {str(e)}")
        raise RuntimeError("Scroll system initialization failed") from e

def validate_scroll_time() -> bool:
    """
    Validate that the scroll time contains all required keys and values.
    
    Returns:
        bool: True if scroll time is valid, False otherwise
    """
    required_keys = {
        "scroll_day": str,
        "gate": int,
        "gate_name": str,
        "solar_hour": int,
        "enano_pulse": int,
        "time_remaining_today": str
    }
    
    try:
        scroll_time = get_scroll_time()
        
        # Check all required keys exist and have correct types
        for key, expected_type in required_keys.items():
            if key not in scroll_time:
                logger.error(f"Missing required scroll time key: {key}")
                return False
            if not isinstance(scroll_time[key], expected_type):
                logger.error(f"Invalid type for {key}: expected {expected_type}, got {type(scroll_time[key])}")
                return False
        
        # Validate value ranges
        if not (1 <= scroll_time["gate"] <= 7):
            logger.error(f"Invalid gate number: {scroll_time['gate']}")
            return False
        if not (1 <= scroll_time["solar_hour"] <= 24):
            logger.error(f"Invalid solar hour: {scroll_time['solar_hour']}")
            return False
        if not (1 <= scroll_time["enano_pulse"] <= 91):
            logger.error(f"Invalid ENano pulse: {scroll_time['enano_pulse']}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Scroll time validation failed: {str(e)}")
        return False 
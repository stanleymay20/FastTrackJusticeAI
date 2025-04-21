from loguru import logger
import sys
import os
from datetime import datetime
from typing import Optional

# Configure base logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler with INFO level

# Add file handlers
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Main log file with rotation
logger.add(
    os.path.join(logs_dir, "fasttrackjustice.log"),
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[scroll_phase]} | {message}"
)

# Error log file
logger.add(
    os.path.join(logs_dir, "errors.log"),
    level="ERROR",
    rotation="100 MB",
    retention="30 days",
    backtrace=True,
    diagnose=True
)

# Scroll phase specific log
logger.add(
    os.path.join(logs_dir, "scroll_phases.log"),
    filter=lambda record: "scroll_phase" in record["extra"],
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[scroll_phase]} | Gate: {extra[gate]:.0f} | {message}"
)

def get_logger(name: str, scroll_phase: Optional[str] = None):
    """Get a contextualized logger with scroll phase awareness."""
    from app.utils.scroll_time import get_scroll_time  # Import here to avoid circular imports
    
    if scroll_phase is None:
        scroll_data = get_scroll_time()
        scroll_phase = scroll_data["phase"]
    
    return logger.bind(
        context=name,
        scroll_phase=scroll_phase,
        gate=get_gate_number()
    )

def get_gate_number() -> float:
    """Calculate the current gate number based on time and scroll phase."""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    
    # Gate calculation: 1-12 based on hour, with fractional component from minutes
    gate = ((hour % 12) or 12) + (minute / 60.0)
    return gate

# Create module-level logger
log = get_logger(__name__)

# Example usage:
# log.info("Judgment request received")
# log.error("Failed to generate judgment", judgment_id="123") 
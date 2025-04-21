import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Define the log directory
LOG_DIR = "logs"
SCROLL_LOG_FILE = os.path.join(LOG_DIR, "scroll.log")

def ensure_log_directory():
    """Ensure the log directory exists."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def log_scroll_shift(scroll_data: Dict[str, Any], latitude: Optional[float] = None, longitude: Optional[float] = None):
    """
    Log a scroll shift event to the scroll log file.
    
    Args:
        scroll_data: The scroll data to log
        latitude: Optional latitude of the user's location
        longitude: Optional longitude of the user's location
    """
    ensure_log_directory()
    
    # Create a log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "phase": scroll_data.get("phase"),
        "gate": scroll_data.get("gate"),
        "gate_name": scroll_data.get("gate_name"),
        "enano_pulse": scroll_data.get("enano_pulse"),
        "solar_hour": scroll_data.get("solar_hour"),
        "is_active": scroll_data.get("is_active"),
        "location": {
            "latitude": latitude,
            "longitude": longitude
        } if latitude is not None and longitude is not None else None
    }
    
    # Append to the log file
    with open(SCROLL_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_scroll_history(days: int = 7) -> list:
    """
    Get the scroll history for the specified number of days.
    
    Args:
        days: Number of days of history to retrieve
        
    Returns:
        List of log entries
    """
    ensure_log_directory()
    
    if not os.path.exists(SCROLL_LOG_FILE):
        return []
    
    # Read the log file
    with open(SCROLL_LOG_FILE, "r") as f:
        log_entries = [json.loads(line) for line in f if line.strip()]
    
    # Filter by date
    cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
    filtered_entries = [
        entry for entry in log_entries
        if datetime.fromisoformat(entry["timestamp"]).timestamp() > cutoff_date
    ]
    
    return filtered_entries 
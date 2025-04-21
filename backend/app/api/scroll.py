from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any, List
from ..utils.scroll_time import get_scroll_time
from ..utils.scroll_logger import log_scroll_shift, get_scroll_history

router = APIRouter()

@router.get("/scroll-time")
async def get_scroll_time_endpoint(
    latitude: Optional[float] = Query(None, description="Latitude of the user's location (optional)"),
    longitude: Optional[float] = Query(None, description="Longitude of the user's location (optional)")
) -> Dict[str, Any]:
    """
    🌍 Retrieve the current Scroll Time based on user's GPS location (optional).
    
    This endpoint calculates the current scroll status based on:
    - Astronomical sunrise/sunset (if coordinates provided)
    - Fixed sacred times (if not)
    
    Returns:
    - phase: Current scroll phase (dawn, noon, dusk, night)
    - is_active: Whether scroll is spiritually active (True during dawn & dusk)
    - next_phase: What scroll phase is coming next
    - time_remaining: Time remaining in current phase
    - solar_hour: Current solar hour (1-24)
    - gate: Current sacred gate number (1-7)
    - gate_name: Sacred name of the current gate
    - enano_pulse: Current ENano pulse (1-91)
    - time_remaining_today: Time remaining in this scroll day
    - sunrise/sunset/dawn/dusk (ISO format if geolocation is active)
    """
    try:
        scroll_data = get_scroll_time(latitude, longitude)
        
        # Log the scroll shift
        log_scroll_shift(scroll_data, latitude, longitude)
        
        return scroll_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating scroll time: {str(e)}")

@router.get("/scroll-status")
async def get_scroll_status_endpoint(
    latitude: Optional[float] = Query(None, description="Latitude (optional)"),
    longitude: Optional[float] = Query(None, description="Longitude (optional)")
) -> Dict[str, Any]:
    """
    🔄 Backward-compatible alias to `/scroll-time` for frontend use.
    """
    return await get_scroll_time_endpoint(latitude, longitude)

@router.get("/scroll-history")
async def get_scroll_history_endpoint(
    days: Optional[int] = Query(7, description="Number of days of history to retrieve")
) -> List[Dict[str, Any]]:
    """
    📜 Retrieve the scroll history for the specified number of days.
    
    This endpoint returns a list of scroll shifts that have occurred in the specified time period.
    
    Args:
        days: Number of days of history to retrieve (default: 7)
        
    Returns:
        List of scroll shift events
    """
    try:
        history = get_scroll_history(days)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving scroll history: {str(e)}") 
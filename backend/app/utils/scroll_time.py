from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import astral
from astral import LocationInfo
import math

# Define scroll phases and their durations
SCROLL_PHASES = {
    "dawn": timedelta(hours=2),
    "noon": timedelta(hours=6),
    "dusk": timedelta(hours=2),
    "night": timedelta(hours=14)
}

# Define scroll gate cycle (7 gates repeat every 7 days)
GATE_NAMES = [
    "Gate 1 – Fire of Beginning",
    "Gate 2 – River of Truth",
    "Gate 3 – Scroll of Mercy",
    "Gate 4 – Lamp of Justice",
    "Gate 5 – Flame of Knowledge",
    "Gate 6 – Mirror of Grace",
    "Gate 7 – Divine Architecture"
]

# Sacred scroll constants
TOTAL_SCROLL_DAYS = 364
TOTAL_ENANO_PULSES = 91  # 91 per scroll day
SOLAR_DAY_HOURS = 24

def get_scroll_time(latitude: Optional[float] = None, longitude: Optional[float] = None) -> Dict[str, Any]:
    """
    Get the current scroll time based on the user's location.
    
    Args:
        latitude: Optional latitude of the user's location
        longitude: Optional longitude of the user's location
        
    Returns:
        Dict containing scroll time information
    """
    now = datetime.now()
    
    # If coordinates are provided, calculate sunrise and sunset times
    if latitude is not None and longitude is not None:
        location = LocationInfo("User Location", "Region", "Country", latitude, longitude)
        sun = astral.sun(location, now)
        sunrise = sun["sunrise"]
        sunset = sun["sunset"]
        
        # Calculate dawn and dusk (civil twilight)
        dawn = sun["dawn"]
        dusk = sun["dusk"]
    else:
        # Default times if no coordinates provided
        sunrise = now.replace(hour=6, minute=0, second=0, microsecond=0)
        sunset = now.replace(hour=18, minute=0, second=0, microsecond=0)
        dawn = now.replace(hour=5, minute=30, second=0, microsecond=0)
        dusk = now.replace(hour=18, minute=30, second=0, microsecond=0)
    
    # Calculate the current phase based on the time of day
    if now < dawn:
        phase = "night"
        time_remaining = dawn - now
        next_phase = "dawn"
    elif now < sunrise:
        phase = "dawn"
        time_remaining = sunrise - now
        next_phase = "noon"
    elif now < sunset:
        phase = "noon"
        time_remaining = sunset - now
        next_phase = "dusk"
    elif now < dusk:
        phase = "dusk"
        time_remaining = dusk - now
        next_phase = "night"
    else:
        phase = "night"
        # Calculate time until next day's dawn
        next_day = now + timedelta(days=1)
        if latitude is not None and longitude is not None:
            next_location = LocationInfo("User Location", "Region", "Country", latitude, longitude)
            next_sun = astral.sun(next_location, next_day)
            next_dawn = next_sun["dawn"]
            time_remaining = next_dawn - now
        else:
            next_dawn = (now + timedelta(days=1)).replace(hour=5, minute=30, second=0, microsecond=0)
            time_remaining = next_dawn - now
        next_phase = "dawn"
    
    # Calculate if scroll is active (active during dawn and dusk)
    is_active = phase in ["dawn", "dusk"]
    
    # Calculate solar hour (0-23)
    if latitude is not None and longitude is not None:
        # Use actual sunrise for solar hour calculation
        solar_hour = calculate_solar_hour(now, sunrise)
    else:
        # Use fixed sunrise for solar hour calculation
        solar_hour = calculate_solar_hour(now, sunrise)
    
    # Calculate gate (1-7) based on solar hour
    gate = calculate_gate(solar_hour)
    
    # Calculate ENano pulse (1-91)
    enano_pulse = calculate_enano_pulse(now)
    
    # Format time remaining as string
    hours, minutes, seconds = format_time_remaining(time_remaining)
    time_remaining_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Calculate time remaining today
    time_remaining_today = timedelta(hours=24) - timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)
    hours_today, minutes_today, seconds_today = format_time_remaining(time_remaining_today)
    time_remaining_today_str = f"{hours_today:02d}:{minutes_today:02d}:{seconds_today:02d}"
    
    return {
        "phase": phase,
        "is_active": is_active,
        "time_remaining": time_remaining_str,
        "next_phase": next_phase,
        "solar_hour": solar_hour,
        "gate": gate,
        "gate_name": GATE_NAMES[gate - 1],
        "enano_pulse": enano_pulse,
        "time_remaining_today": time_remaining_today_str,
        "sunrise": sunrise.isoformat() if latitude is not None and longitude is not None else None,
        "sunset": sunset.isoformat() if latitude is not None and longitude is not None else None,
        "dawn": dawn.isoformat() if latitude is not None and longitude is not None else None,
        "dusk": dusk.isoformat() if latitude is not None and longitude is not None else None
    }

def calculate_solar_hour(now: datetime, sunrise: datetime) -> int:
    """
    Calculate the solar hour (0-23) based on the current time and sunrise.
    
    Args:
        now: Current time
        sunrise: Sunrise time
        
    Returns:
        Solar hour (0-23)
    """
    # Calculate seconds since sunrise
    seconds_since_sunrise = (now - sunrise).total_seconds()
    
    # Calculate solar hour (0-23)
    # Each solar hour is 1/24 of the day (3600 seconds)
    solar_hour = int((seconds_since_sunrise / 3600) % 24)
    
    return solar_hour

def calculate_gate(solar_hour: int) -> int:
    """
    Calculate the gate (1-7) based on the solar hour.
    
    Args:
        solar_hour: Solar hour (0-23)
        
    Returns:
        Gate number (1-7)
    """
    # Map solar hours to gates
    # Gate 1: Hours 0-3
    # Gate 2: Hours 4-7
    # Gate 3: Hours 8-11
    # Gate 4: Hours 12-15
    # Gate 5: Hours 16-19
    # Gate 6: Hours 20-22
    # Gate 7: Hours 23-24
    
    if 0 <= solar_hour <= 3:
        return 1
    elif 4 <= solar_hour <= 7:
        return 2
    elif 8 <= solar_hour <= 11:
        return 3
    elif 12 <= solar_hour <= 15:
        return 4
    elif 16 <= solar_hour <= 19:
        return 5
    elif 20 <= solar_hour <= 22:
        return 6
    else:
        return 7

def calculate_enano_pulse(now: datetime) -> int:
    """
    Calculate the ENano pulse (1-91) based on the current time.
    
    Args:
        now: Current time
        
    Returns:
        ENano pulse (1-91)
    """
    # Calculate seconds since midnight
    seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    
    # Calculate ENano pulse (1-91)
    # Each ENano pulse is 1/91 of the day (approximately 15.8 minutes)
    enano_pulse = int((seconds_since_midnight / (24 * 3600 / 91)) % 91) + 1
    
    return enano_pulse

def format_time_remaining(time_delta: timedelta) -> Tuple[int, int, int]:
    """
    Format time remaining as hours, minutes, seconds.
    
    Args:
        time_delta: Time delta to format
        
    Returns:
        Tuple of (hours, minutes, seconds)
    """
    total_seconds = int(time_delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return hours, minutes, seconds 
import pytest
from datetime import datetime, timedelta
from app.utils.scroll_time import get_scroll_time, SCROLL_PHASES

def test_get_scroll_time():
    """Test the get_scroll_time function returns expected data structure."""
    scroll_data = get_scroll_time()
    
    # Check that all expected keys are present
    assert "phase" in scroll_data
    assert "is_active" in scroll_data
    assert "time_remaining" in scroll_data
    assert "next_phase" in scroll_data
    
    # Check that phase is one of the expected values
    assert scroll_data["phase"] in SCROLL_PHASES.keys()
    
    # Check that next_phase is one of the expected values
    assert scroll_data["next_phase"] in SCROLL_PHASES.keys()
    
    # Check that time_remaining is a timedelta
    assert isinstance(scroll_data["time_remaining"], timedelta)
    
    # Check that is_active is a boolean
    assert isinstance(scroll_data["is_active"], bool)

def test_scroll_phase_transitions():
    """Test that scroll phases transition correctly."""
    # Get current scroll data
    current_data = get_scroll_time()
    
    # Calculate expected next phase
    phases = list(SCROLL_PHASES.keys())
    current_index = phases.index(current_data["phase"])
    expected_next = phases[(current_index + 1) % len(phases)]
    
    # Verify next phase is correct
    assert current_data["next_phase"] == expected_next

def test_time_remaining_validity():
    """Test that time_remaining is within expected range."""
    scroll_data = get_scroll_time()
    
    # Time remaining should be positive
    assert scroll_data["time_remaining"].total_seconds() > 0
    
    # Time remaining should not exceed the duration of the current phase
    current_phase_duration = SCROLL_PHASES[scroll_data["phase"]]
    assert scroll_data["time_remaining"] <= current_phase_duration 
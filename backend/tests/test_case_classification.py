import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.utils.scroll_time import get_scroll_time

client = TestClient(app)

def test_classify_case_endpoint():
    """Test the case classification endpoint with a sample case."""
    test_case = {
        "case_text": "The defendant was found in possession of illegal substances.",
        "latitude": 40.7128,
        "longitude": -74.0060
    }
    
    response = client.post("/api/case/classify-case", json=test_case)
    assert response.status_code == 200
    
    data = response.json()
    assert "legal_category" in data
    assert "confidence" in data
    assert "scroll_phase" in data
    assert "alignment" in data
    assert "divine_title" in data

def test_classify_case_get_endpoint():
    """Test the GET endpoint for case classification."""
    params = {
        "case_text": "The defendant was found in possession of illegal substances.",
        "latitude": 40.7128,
        "longitude": -74.0060
    }
    
    response = client.get("/api/case/classify-case", params=params)
    assert response.status_code == 200
    
    data = response.json()
    assert "legal_category" in data
    assert "confidence" in data
    assert "scroll_phase" in data
    assert "alignment" in data
    assert "divine_title" in data

def test_scroll_time_integration():
    """Test that the scroll time is properly integrated with case classification."""
    scroll_data = get_scroll_time()
    assert "phase" in scroll_data
    assert "is_active" in scroll_data
    assert "time_remaining" in scroll_data
    assert "next_phase" in scroll_data 
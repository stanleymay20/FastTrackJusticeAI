from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_scroll_time_endpoint():
    """Test the GET /api/scroll/time endpoint."""
    response = client.get("/api/scroll/time")
    assert response.status_code == 200
    
    data = response.json()
    assert "phase" in data
    assert "is_active" in data
    assert "time_remaining" in data
    assert "next_phase" in data

def test_get_scroll_time_with_coordinates():
    """Test the GET /api/scroll/time endpoint with coordinates."""
    # Test with valid coordinates
    response = client.get("/api/scroll/time?latitude=40.7128&longitude=-74.0060")
    assert response.status_code == 200
    
    data = response.json()
    assert "phase" in data
    assert "is_active" in data
    assert "time_remaining" in data
    assert "next_phase" in data
    
    # Test with invalid coordinates
    response = client.get("/api/scroll/time?latitude=invalid&longitude=-74.0060")
    assert response.status_code == 422  # Validation error
    
    response = client.get("/api/scroll/time?latitude=40.7128&longitude=invalid")
    assert response.status_code == 422  # Validation error 
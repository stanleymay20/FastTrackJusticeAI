import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
from unittest.mock import patch, MagicMock

from app.main import app
from app.api.insights import router

client = TestClient(app)

# Test data
SAMPLE_INSIGHT = {
    "timestamp": datetime.now().isoformat(),
    "type": "phase_distribution",
    "phase": "dawn",
    "severity": "info",
    "message": "🌅 Dawn phase had 35% of judgments, significantly above average",
    "emoji": "🌅"
}

SAMPLE_SUMMARY = {
    "timestamp": datetime.now().isoformat(),
    "total_insights": 1,
    "phase_distribution": {"dawn": 35, "noon": 25, "dusk": 20, "night": 20},
    "severity_distribution": {"info": 1, "warning": 0, "error": 0},
    "top_insights": [SAMPLE_INSIGHT]
}

@pytest.fixture
def mock_insights_dir(tmp_path):
    """Create a temporary insights directory with test data"""
    insights_dir = tmp_path / "insights"
    insights_dir.mkdir()
    
    # Create daily summary
    summary_file = insights_dir / "daily_summary.json"
    with open(summary_file, "w") as f:
        json.dump(SAMPLE_SUMMARY, f)
    
    # Create historical insights
    for i in range(3):
        date = datetime.now() - timedelta(days=i)
        insight_file = insights_dir / f"insights_{date.strftime('%Y-%m-%d')}.json"
        with open(insight_file, "w") as f:
            json.dump([SAMPLE_INSIGHT], f)
    
    return insights_dir

@pytest.fixture
def mock_env(monkeypatch, mock_insights_dir):
    """Set up environment variables and mock the insights directory"""
    monkeypatch.setenv("INSIGHTS_DIR", str(mock_insights_dir))
    return mock_insights_dir

def test_get_daily_summary(mock_env):
    """Test retrieving the daily summary"""
    response = client.get("/api/insights/daily-summary")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data
    assert "total_insights" in data
    assert "phase_distribution" in data
    assert "severity_distribution" in data
    assert "top_insights" in data

def test_get_insights_with_filters(mock_env):
    """Test retrieving filtered insights"""
    # Test with phase filter
    response = client.get("/api/insights/insights?phase=dawn")
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data
    assert all(insight["phase"] == "dawn" for insight in data["insights"])
    
    # Test with severity filter
    response = client.get("/api/insights/insights?severity=info")
    assert response.status_code == 200
    data = response.json()
    assert all(insight["severity"] == "info" for insight in data["insights"])
    
    # Test with type filter
    response = client.get("/api/insights/insights?insight_type=phase_distribution")
    assert response.status_code == 200
    data = response.json()
    assert all(insight["type"] == "phase_distribution" for insight in data["insights"])

def test_get_historical_insights(mock_env):
    """Test retrieving historical insights"""
    start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    response = client.get(f"/api/insights/historical?start_date={start_date}&end_date={end_date}")
    assert response.status_code == 200
    data = response.json()
    assert "historical_insights" in data
    assert len(data["historical_insights"]) > 0

def test_generate_daily_summary(mock_env):
    """Test generating a new daily summary"""
    response = client.post("/api/insights/generate")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "summary" in data
    assert data["message"] == "Daily summary generated successfully"

def test_invalid_date_format():
    """Test handling of invalid date format"""
    response = client.get("/api/insights/historical?start_date=invalid&end_date=2024-03-20")
    assert response.status_code == 400
    assert "Invalid date format" in response.json()["detail"]

def test_missing_insights_directory(mock_env, monkeypatch):
    """Test handling of missing insights directory"""
    monkeypatch.setenv("INSIGHTS_DIR", "/nonexistent/directory")
    response = client.get("/api/insights/daily-summary")
    assert response.status_code == 404
    assert "Daily summary not found" in response.json()["detail"]

def test_insight_structure_validation(mock_env):
    """Test that insights have the required structure"""
    response = client.get("/api/insights/insights")
    assert response.status_code == 200
    data = response.json()
    
    for insight in data["insights"]:
        assert "timestamp" in insight
        assert "type" in insight
        assert "phase" in insight
        assert "severity" in insight
        assert "message" in insight
        assert "emoji" in insight

def test_phase_distribution_validation(mock_env):
    """Test that phase distribution percentages sum to 100"""
    response = client.get("/api/insights/daily-summary")
    assert response.status_code == 200
    data = response.json()
    
    phase_distribution = data["phase_distribution"]
    total_percentage = sum(phase_distribution.values())
    assert abs(total_percentage - 100) < 0.1  # Allow for small floating-point differences 
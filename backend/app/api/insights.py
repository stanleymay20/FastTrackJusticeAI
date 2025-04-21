from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

router = APIRouter(prefix="/insights", tags=["insights"])

# Constants
INSIGHTS_DIR = Path("backend/logs/insights")
DAILY_SUMMARY_PATH = INSIGHTS_DIR / "daily_summary.json"

@router.get("/daily-summary")
async def get_daily_summary():
    """Get the latest daily summary of scroll intelligence insights."""
    try:
        if not DAILY_SUMMARY_PATH.exists():
            raise HTTPException(status_code=404, detail="Daily summary not found")
        
        with open(DAILY_SUMMARY_PATH, 'r') as f:
            summary = json.load(f)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving daily summary: {str(e)}")

@router.get("/insights")
async def get_insights(
    days: int = Query(7, description="Number of days to look back"),
    insight_type: Optional[str] = Query(None, description="Type of insight to filter by"),
    phase: Optional[str] = Query(None, description="Scroll phase to filter by"),
    severity: Optional[str] = Query(None, description="Severity level to filter by")
):
    """Get filtered scroll intelligence insights."""
    try:
        insights = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Iterate through insight files in the directory
        for file in INSIGHTS_DIR.glob("*.json"):
            if file.name == "daily_summary.json":
                continue
                
            with open(file, 'r') as f:
                file_insights = json.load(f)
                
            # Apply filters
            for insight in file_insights:
                insight_date = datetime.fromisoformat(insight.get('timestamp', ''))
                if insight_date < cutoff_date:
                    continue
                    
                if insight_type and insight.get('type') != insight_type:
                    continue
                    
                if phase and insight.get('phase') != phase:
                    continue
                    
                if severity and insight.get('severity') != severity:
                    continue
                    
                insights.append(insight)
                
        return {"insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving insights: {str(e)}")

@router.get("/historical")
async def get_historical_insights(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    """Get historical insights over a specified period."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        historical_insights = []
        
        for file in INSIGHTS_DIR.glob("*.json"):
            if file.name == "daily_summary.json":
                continue
                
            file_date = datetime.fromtimestamp(file.stat().st_mtime)
            if start <= file_date <= end:
                with open(file, 'r') as f:
                    insights = json.load(f)
                    historical_insights.extend(insights)
                    
        return {"historical_insights": historical_insights}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving historical insights: {str(e)}")

@router.post("/generate")
async def generate_daily_summary():
    """Force generation of a new daily summary."""
    try:
        # This would typically call your insights generation logic
        # For now, we'll create a sample summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_insights": 0,
            "phase_distribution": {},
            "severity_distribution": {},
            "top_insights": []
        }
        
        # Ensure directory exists
        INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Write summary to file
        with open(DAILY_SUMMARY_PATH, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return {"message": "Daily summary generated successfully", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating daily summary: {str(e)}") 
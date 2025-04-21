from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel
from ..utils.case_classifier import classify_case

router = APIRouter()

class CaseClassificationRequest(BaseModel):
    """Request model for case classification."""
    case_text: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

@router.post("/classify-case")
async def classify_case_endpoint(request: CaseClassificationRequest) -> Dict[str, Any]:
    """
    🌟 Classify a legal case and calculate its alignment with the current scroll phase.
    
    This endpoint analyzes the case text and determines:
    - The legal category (Criminal, Civil, Family, Administrative)
    - The confidence in the classification
    - The current scroll phase
    - The alignment between the case and the scroll phase
    - A divine title for the classification
    
    Args:
        request: The case classification request containing the case text and optional location
        
    Returns:
        A dictionary containing the classification results and scroll alignment
    """
    try:
        result = classify_case(
            case_text=request.case_text,
            latitude=request.latitude,
            longitude=request.longitude
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying case: {str(e)}")

@router.get("/classify-case")
async def classify_case_get_endpoint(
    case_text: str = Query(..., description="The text of the legal case to classify"),
    latitude: Optional[float] = Query(None, description="Latitude of the user's location (optional)"),
    longitude: Optional[float] = Query(None, description="Longitude of the user's location (optional)")
) -> Dict[str, Any]:
    """
    🌟 GET endpoint for case classification.
    
    This endpoint provides the same functionality as the POST endpoint but accepts parameters as query parameters.
    
    Args:
        case_text: The text of the legal case to classify
        latitude: Optional latitude of the user's location
        longitude: Optional longitude of the user's location
        
    Returns:
        A dictionary containing the classification results and scroll alignment
    """
    try:
        result = classify_case(
            case_text=case_text,
            latitude=latitude,
            longitude=longitude
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying case: {str(e)}") 
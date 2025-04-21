from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
import json

# Import scroll components
from app.utils.scroll_time import get_scroll_time
from app.utils.scroll_classifier import scroll_aligned_classification, ScrollIntrusionAlert
from app.utils.scroll_judgment_writer import ScrollJudgmentWriter

# Import other components
from app.services.case_processor import CaseProcessor
from app.services.judgment_generator import JudgmentGenerator
from app.models.case import Case, CaseResponse
from app.models.judgment import Judgment, JudgmentResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
case_processor = CaseProcessor()
judgment_generator = JudgmentGenerator()
scroll_judgment_writer = ScrollJudgmentWriter()

@router.post("/process-case", response_model=CaseResponse)
async def process_case(
    request: Request,
    file: UploadFile = File(...),
    case_type: Optional[str] = Form(None)
):
    """
    Process a legal case document and classify it.
    This endpoint is protected by the ScrollGuard to ensure proper timing.
    """
    # Get scroll time from request state (injected by middleware)
    scroll_time = request.state.scroll_time
    
    # Check if case processing is allowed at this time
    from app.utils.scroll_guard import ScrollGuard
    scroll_guard = ScrollGuard()
    timing_check = scroll_guard.check_operation_timing("case_classification")
    
    if not timing_check["allow_operation"]:
        raise ScrollIntrusionAlert(timing_check["warning_message"])
    
    # Process the case
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode("utf-8")
        
        # Classify the case
        classification = case_processor.classify_case(text_content)
        
        # Apply scroll alignment to classification
        aligned_classification = scroll_aligned_classification(
            text_content, 
            classification, 
            scroll_time
        )
        
        # Create case response
        case_response = CaseResponse(
            case_id="case-" + str(hash(text_content))[:8],
            classification=aligned_classification,
            scroll_context=scroll_time
        )
        
        return case_response
    
    except Exception as e:
        logger.error(f"Error processing case: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing case: {str(e)}")

@router.post("/generate-judgment", response_model=JudgmentResponse)
async def generate_judgment(
    request: Request,
    case: Case
):
    """
    Generate a judgment for a legal case.
    This endpoint is protected by the ScrollGuard to ensure proper timing.
    """
    # Get scroll time from request state (injected by middleware)
    scroll_time = request.state.scroll_time
    
    # Check if judgment generation is allowed at this time
    from app.utils.scroll_guard import ScrollGuard
    scroll_guard = ScrollGuard()
    timing_check = scroll_guard.check_operation_timing("judgment_rendering")
    
    if not timing_check["allow_operation"]:
        raise ScrollIntrusionAlert(timing_check["warning_message"])
    
    try:
        # Generate base judgment
        base_judgment = judgment_generator.generate_judgment(
            case.text,
            case.classification
        )
        
        # Generate scroll-aligned judgment
        scroll_prompt = scroll_judgment_writer.generate_scroll_prompt(
            case.text,
            case.classification
        )
        
        # In a real implementation, you would use the prompt with your LLM
        # For now, we'll just use the base judgment
        scroll_judgment = scroll_judgment_writer.generate_judgment(
            case.text,
            case.classification,
            base_judgment
        )
        
        # Create judgment response
        judgment_response = JudgmentResponse(
            judgment_id="judgment-" + str(hash(case.text))[:8],
            judgment=scroll_judgment,
            scroll_context=scroll_time
        )
        
        return judgment_response
    
    except Exception as e:
        logger.error(f"Error generating judgment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating judgment: {str(e)}")

@router.get("/scroll-time")
async def get_current_scroll_time():
    """
    Get the current scroll time information.
    """
    scroll_time = get_scroll_time()
    return scroll_time 
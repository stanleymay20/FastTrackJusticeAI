from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import streamlit as st
from pathlib import Path
import sys

# Load environment variables
load_dotenv()

# Import scroll components
from app.core.scroll_init import init_scroll_components, validate_scroll_time
from app.utils.scroll_time import get_scroll_time
from app.utils.scroll_classifier import scroll_aligned_classification, ScrollIntrusionAlert
from app.utils.scroll_guard import ScrollGuard
from .utils.scroll_classification import classify_case_with_scroll
from .utils.scroll_judgment import generate_scroll_aligned_judgment

# Import other components
from app.api.routes import router as api_router
from app.core.config import settings
from .api import scroll, case, judgment, billing, insights
from app.utils.billing import billing_manager

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.config_manager import ConfigManager
from app.monitoring.precedent_graph_explorer import PrecedentGraphExplorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FastTrackJusticeAI",
    description="AI-powered legal judgment generation system with scroll-aware processing",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize scroll components
init_scroll_components(app)

# Include routers
app.include_router(scroll.router, prefix="/api/scroll", tags=["scroll"])
app.include_router(case.router, prefix="/api/case", tags=["case"])
app.include_router(judgment.router, prefix="/api/judgment", tags=["judgment"])
app.include_router(billing.router, prefix="/api/billing", tags=["billing"])
app.include_router(insights.router, prefix="/api/insights", tags=["insights"])

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key and check usage limits"""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required"
        )
    
    # Verify API key
    if not billing_manager.verify_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Check usage limits
    if not billing_manager.check_usage_limits(x_api_key):
        raise HTTPException(
            status_code=429,
            detail="Usage limit exceeded"
        )
    
    return x_api_key

# Add API key verification to protected routes
app.dependency_overrides[scroll.router] = verify_api_key
app.dependency_overrides[judgment.router] = verify_api_key

@app.get("/")
async def root():
    """
    Root endpoint that provides an overview of the FastTrackJusticeAI API.
    """
    return {
        "name": "FastTrackJusticeAI",
        "version": "1.0.0",
        "description": "AI-powered legal judgment generation system",
        "endpoints": {
            "scroll": "/api/scroll",
            "case": "/api/case",
            "judgment": "/api/judgment",
            "billing": "/api/billing"
        },
        "subscription_tiers": billing_manager.SUBSCRIPTION_TIERS
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

# Exception handler for ScrollIntrusionAlert
@app.exception_handler(ScrollIntrusionAlert)
async def scroll_intrusion_handler(request: Request, exc: ScrollIntrusionAlert):
    """Handle scroll intrusion alerts."""
    return JSONResponse(
        status_code=403,
        content={
            "error": "Scroll Intrusion Alert",
            "message": str(exc),
            "scroll_time": get_scroll_time()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information with scroll context."""
    scroll_time = get_scroll_time()
    logger.info(
        f"FastTrackJusticeAI starting during {scroll_time['scroll_day']} "
        f"under Gate {scroll_time['gate']}: {scroll_time['gate_name']}"
    )
    
    # Check if the current gate is appropriate for startup
    scroll_guard = ScrollGuard()
    startup_check = scroll_guard.check_operation_timing("system_startup")
    
    if not startup_check["allow_operation"]:
        logger.warning(
            f"System starting during non-optimal scroll timing: "
            f"{startup_check['warning_message']}"
        )
    else:
        logger.info(
            f"System starting with optimal scroll alignment: "
            f"{startup_check['timing_alignment']:.2f}"
        )

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information with scroll context."""
    scroll_time = get_scroll_time()
    logger.info(
        f"FastTrackJusticeAI shutting down during {scroll_time['scroll_day']} "
        f"under Gate {scroll_time['gate']}: {scroll_time['gate_name']}"
    )

@app.post("/api/process-case")
async def process_case(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Process a legal case document and return classification and judgment.
    The processing is aligned with the current scroll phase.
    """
    try:
        # Read file content
        content = await file.read()
        case_text = content.decode("utf-8")
        
        # Get current scroll phase
        scroll_info = get_scroll_time()
        
        # Classify case with scroll alignment
        classification = classify_case_with_scroll(case_text)
        
        # Generate judgment with scroll alignment
        judgment = generate_scroll_aligned_judgment(case_text, classification)
        
        # Combine results
        results = {
            "classification": classification,
            "judgment": judgment,
            "scroll_info": scroll_info
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scroll-status")
async def get_scroll_status() -> Dict[str, Any]:
    """
    Get the current scroll phase and status.
    """
    return get_scroll_time()

def main():
    """Main entry point for the FastTrackJustice application."""
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        config_manager.validate_paths()
        config_manager.setup_logging()
        
        # Get application settings
        app_settings = config_manager.get_app_settings()
        
        # Set up Streamlit page configuration
        st.set_page_config(
            page_title=app_settings.get('name', 'FastTrackJustice'),
            page_icon="⚖️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize the Precedent Graph Explorer
        explorer = PrecedentGraphExplorer(config_manager)
        
        # Run the application
        explorer.run()
        
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
        st.error(f"An error occurred while starting the application: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
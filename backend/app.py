from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv

# Import our modules
from .models.summarizer import LegalSummarizer
from .models.classifier import CaseClassifier
from .models.judgment_drafter import JudgmentDrafter
from .utils.data_loader import DataLoader
from .utils.evaluator import ModelEvaluator

# Load environment variables
load_dotenv()

app = FastAPI(
    title="FastTrackJusticeAI",
    description="An AI-powered legal case processing platform",
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

# Initialize models
summarizer = LegalSummarizer()
classifier = CaseClassifier()
judgment_drafter = JudgmentDrafter()
data_loader = DataLoader()
evaluator = ModelEvaluator()

class CaseResponse(BaseModel):
    summary: str
    classification: dict
    judgment_draft: str
    confidence_scores: dict

@app.post("/api/process-case", response_model=CaseResponse)
async def process_case(file: UploadFile = File(...)):
    """
    Process a legal case document and return summary, classification, and judgment draft.
    """
    try:
        # Read and validate file
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process the case
        text = data_loader.extract_text(content, file.filename)
        
        # Generate summary
        summary = summarizer.generate_summary(text)
        
        # Classify case
        classification = classifier.classify_case(text)
        
        # Generate judgment draft
        judgment = judgment_drafter.generate_judgment(text, summary, classification)
        
        # Calculate confidence scores
        confidence_scores = evaluator.calculate_confidence_scores(
            summary, classification, judgment
        )
        
        return CaseResponse(
            summary=summary,
            classification=classification,
            judgment_draft=judgment,
            confidence_scores=confidence_scores
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluate")
async def evaluate_models():
    """
    Return model performance metrics.
    """
    try:
        metrics = evaluator.get_model_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 
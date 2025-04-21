import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from typing import Dict, Any, List, Tuple
import os
import json
from datetime import datetime
from .scroll_time import get_scroll_time
from .scroll_logger import log_scroll_shift

# Define case categories
CASE_CATEGORIES = ["Criminal", "Civil", "Family", "Administrative"]

# Define phase-case weights for enhanced alignment
PHASE_CASE_WEIGHTS = {
    "dawn": {"Family": 1.2, "Civil": 1.0, "Criminal": 0.8, "Administrative": 0.9},
    "noon": {"Civil": 1.2, "Administrative": 1.1, "Criminal": 0.9, "Family": 0.8},
    "dusk": {"Criminal": 1.3, "Civil": 1.0, "Family": 0.9, "Administrative": 0.7},
    "night": {"Criminal": 1.1, "Administrative": 1.2, "Family": 0.7, "Civil": 0.8}
}

# Define phase keywords for semantic alignment
PHASE_KEYWORDS = {
    "dawn": ["beginning", "new", "start", "birth", "creation", "initiation", "emergence", "awakening"],
    "noon": ["balance", "justice", "equality", "harmony", "center", "middle", "peak", "zenith"],
    "dusk": ["end", "conclusion", "resolution", "final", "closing", "completion", "termination", "sunset"],
    "night": ["hidden", "mystery", "secrets", "darkness", "unknown", "obscure", "concealed", "veiled"]
}

# Divine titles for results
DIVINE_TITLES = {
    "Criminal": {
        "dawn": "The Awakening of Justice",
        "noon": "The Balance of Scales",
        "dusk": "The Twilight of Retribution",
        "night": "The Veiled Judgment"
    },
    "Civil": {
        "dawn": "The Dawn of Resolution",
        "noon": "The Harmony of Agreement",
        "dusk": "The Sunset of Dispute",
        "night": "The Hidden Accord"
    },
    "Family": {
        "dawn": "The Birth of Kinship",
        "noon": "The Center of Unity",
        "dusk": "The Closing of Bonds",
        "night": "The Mystery of Lineage"
    },
    "Administrative": {
        "dawn": "The Initiation of Order",
        "noon": "The Peak of Governance",
        "dusk": "The Completion of Process",
        "night": "The Obscure Regulation"
    }
}

# Load model and tokenizer
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "case_classifier")
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "case_classifier")

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Set model to evaluation mode
model.eval()

def calculate_scroll_alignment(text: str, current_phase: str, is_active: bool) -> float:
    """
    Calculate the semantic alignment between the case text and the current scroll phase.
    
    Args:
        text: The case text to analyze
        current_phase: The current scroll phase (dawn, noon, dusk, night)
        is_active: Whether the scroll is currently active
        
    Returns:
        A float between 0 and 1 representing the alignment score
    """
    # Get keywords for the current phase
    keywords = PHASE_KEYWORDS.get(current_phase, [])
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text.lower())
    
    # Count keyword matches
    matches = 0
    for keyword in keywords:
        keyword_tokens = tokenizer.tokenize(keyword.lower())
        for i in range(len(tokens) - len(keyword_tokens) + 1):
            if tokens[i:i+len(keyword_tokens)] == keyword_tokens:
                matches += 1
                break
    
    # Calculate base alignment score
    base_score = min(matches / (len(keywords) + 1), 1.0)
    
    # Apply active scroll bonus
    if is_active:
        base_score *= 1.2
    
    return min(base_score, 1.0)

def classify_case(case_text: str, latitude: float = None, longitude: float = None) -> Dict[str, Any]:
    """
    Classify a legal case and calculate its alignment with the current scroll phase.
    
    Args:
        case_text: The text of the legal case to classify
        latitude: Optional latitude for scroll phase calculation
        longitude: Optional longitude for scroll phase calculation
        
    Returns:
        A dictionary containing classification results and scroll alignment
    """
    # Get current scroll phase
    scroll_data = get_scroll_time(latitude, longitude)
    current_phase = scroll_data["phase"]
    is_active = scroll_data["is_active"]
    
    # Tokenize and prepare input
    inputs = tokenizer(case_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
    # Get predicted class and confidence
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[0][predicted_class].item()
    
    # Calculate scroll alignment
    scroll_alignment = calculate_scroll_alignment(case_text, current_phase, is_active)
    
    # Apply phase-case weights
    weighted = PHASE_CASE_WEIGHTS.get(current_phase, {}).get(CASE_CATEGORIES[predicted_class], 1.0)
    adjusted_score = min(scroll_alignment * weighted * (1.2 if is_active else 1.0), 1.0)
    
    # Get divine title
    divine_title = DIVINE_TITLES.get(CASE_CATEGORIES[predicted_class], {}).get(current_phase, "The Divine Judgment")
    
    # Prepare result
    result = {
        "category": CASE_CATEGORIES[predicted_class],
        "confidence": confidence,
        "scroll_phase": current_phase,
        "is_active": is_active,
        "scroll_alignment": adjusted_score,
        "divine_title": divine_title,
        "gate": scroll_data.get("gate"),
        "gate_name": scroll_data.get("gate_name"),
        "solar_hour": scroll_data.get("solar_hour"),
        "enano_pulse": scroll_data.get("enano_pulse")
    }
    
    # Log the classification
    log_case_classification(result, case_text, latitude, longitude)
    
    return result

def log_case_classification(result: Dict[str, Any], case_text: str, latitude: float = None, longitude: float = None):
    """
    Log a case classification to the case classification log.
    
    Args:
        result: The classification result
        case_text: The text of the legal case
        latitude: Optional latitude of the user's location
        longitude: Optional longitude of the user's location
    """
    # Define the log directory and file
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    log_file = os.path.join(log_dir, "case_classification.log")
    
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "category": result["category"],
        "confidence": result["confidence"],
        "scroll_phase": result["scroll_phase"],
        "is_active": result["is_active"],
        "scroll_alignment": result["scroll_alignment"],
        "divine_title": result["divine_title"],
        "gate": result.get("gate"),
        "gate_name": result.get("gate_name"),
        "solar_hour": result.get("solar_hour"),
        "enano_pulse": result.get("enano_pulse"),
        "case_text_preview": case_text[:200] + "..." if len(case_text) > 200 else case_text,
        "location": {
            "latitude": latitude,
            "longitude": longitude
        } if latitude is not None and longitude is not None else None
    }
    
    # Append to the log file
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n") 
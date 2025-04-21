from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .scroll_time import get_scroll_time

# Initialize model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

# Define case categories
CASE_CATEGORIES = {
    0: "Criminal",
    1: "Civil",
    2: "Family",
    3: "Administrative"
}

def classify_case_with_scroll(case_text: str) -> Dict[str, Any]:
    """
    Classify a legal case with consideration for the current scroll phase.
    
    Args:
        case_text (str): The text content of the legal case
        
    Returns:
        Dict[str, Any]: Classification results including:
            - category: The predicted case category
            - confidence: Confidence score for the prediction
            - scroll_alignment: How well the case aligns with current scroll phase
    """
    # Get current scroll phase
    scroll_info = get_scroll_time()
    
    # Tokenize and prepare input
    inputs = tokenizer(case_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Calculate scroll alignment
    scroll_alignment = calculate_scroll_alignment(
        case_text,
        scroll_info["phase"],
        scroll_info["is_active"]
    )
    
    return {
        "category": CASE_CATEGORIES[predicted_class],
        "confidence": confidence,
        "scroll_alignment": scroll_alignment,
        "scroll_phase": scroll_info["phase"]
    }

def calculate_scroll_alignment(case_text: str, current_phase: str, is_active: bool) -> float:
    """
    Calculate how well the case aligns with the current scroll phase.
    
    Args:
        case_text (str): The text content of the legal case
        current_phase (str): Current scroll phase (dawn, noon, dusk, night)
        is_active (bool): Whether the scroll is currently active
        
    Returns:
        float: Alignment score between 0 and 1
    """
    # Define phase-specific keywords
    phase_keywords = {
        "dawn": ["beginning", "start", "initial", "early", "morning"],
        "noon": ["middle", "central", "core", "main", "primary"],
        "dusk": ["end", "conclusion", "final", "closing", "evening"],
        "night": ["complex", "difficult", "challenging", "complicated", "night"]
    }
    
    # Count keyword matches
    keywords = phase_keywords[current_phase]
    matches = sum(1 for keyword in keywords if keyword.lower() in case_text.lower())
    
    # Calculate base alignment score
    base_score = matches / len(keywords)
    
    # Adjust score based on scroll activity
    if is_active:
        return min(base_score * 1.2, 1.0)  # Boost score during active phases
    return base_score 
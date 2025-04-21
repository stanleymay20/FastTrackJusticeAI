from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ScrollIntrusionAlert(Exception):
    """Exception raised when a case classification is misaligned with the current scroll context."""
    pass

def scroll_aligned_classification(
    text: str, 
    base_classification: Dict[str, float], 
    scroll_time: Dict[str, Any]
) -> Dict[str, float]:
    """
    Wraps the base classification with scroll alignment logic.
    Checks if the classification result contradicts the scroll context.
    
    Args:
        text: The text being classified
        base_classification: The original classification results
        scroll_time: The current scroll timing information
        
    Returns:
        The scroll-aligned classification results
        
    Raises:
        ScrollIntrusionAlert: If the classification is misaligned with the current gate
    """
    # Get the current gate information
    gate_name = scroll_time["gate_name"]
    gate_num = scroll_time["gate"]
    solar_hour = scroll_time["solar_hour"]
    
    # Define gate-specific rules
    gate_rules = {
        "Divine Justice": {
            "allowed_categories": ["criminal", "constitutional", "administrative"],
            "forbidden_categories": [],
            "preferred_hours": [5, 6, 7]  # Hours when justice is strongest
        },
        "Divine Architecture": {
            "allowed_categories": ["property", "contract", "intellectual_property"],
            "forbidden_categories": ["criminal"],
            "preferred_hours": [3, 4, 5]
        },
        "Sacred Harmony": {
            "allowed_categories": ["family", "divorce", "custody", "mediation"],
            "forbidden_categories": [],
            "preferred_hours": [2, 3, 4]
        },
        "Divine Compassion": {
            "allowed_categories": ["family", "immigration", "social_welfare"],
            "forbidden_categories": ["criminal"],
            "preferred_hours": [1, 2, 3]
        }
    }
    
    # Default rules for gates not specifically defined
    default_rules = {
        "allowed_categories": ["criminal", "civil", "family", "property", "contract", 
                              "constitutional", "administrative", "intellectual_property"],
        "forbidden_categories": [],
        "preferred_hours": list(range(1, 13))
    }
    
    # Get rules for current gate
    rules = gate_rules.get(gate_name, default_rules)
    
    # Check for forbidden categories
    for category in rules["forbidden_categories"]:
        if base_classification.get(category, 0) > 0.7:
            logger.warning(
                f"Scroll Intrusion: {category} case detected during {gate_name} gate. "
                f"This gate does not permit {category} cases."
            )
            raise ScrollIntrusionAlert(
                f"Scroll Intrusion: {category} case attempted during {gate_name} gate. "
                f"This gate does not permit {category} cases."
            )
    
    # Check if current hour is preferred for this gate
    hour_alignment = 1.0
    if solar_hour not in rules["preferred_hours"]:
        # Reduce confidence for non-preferred hours
        hour_alignment = 0.8
        logger.info(
            f"Non-preferred hour {solar_hour} for {gate_name} gate. "
            f"Reducing classification confidence."
        )
    
    # Apply hour alignment to classification scores
    aligned_classification = {
        category: score * hour_alignment 
        for category, score in base_classification.items()
    }
    
    # Add scroll context to the classification
    aligned_classification["scroll_gate"] = gate_num
    aligned_classification["scroll_gate_name"] = gate_name
    aligned_classification["scroll_hour_alignment"] = hour_alignment
    
    return aligned_classification 
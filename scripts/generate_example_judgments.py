#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate example judgments with varying scroll contexts for demonstration purposes.
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.judgment_drafter import JudgmentDrafter
from backend.app.utils.scroll_time import get_scroll_time

# Sample case texts
SAMPLE_CASES = [
    {
        "text": """
        The defendant is charged with second-degree assault. On the night of June 15, 2023,
        the defendant allegedly struck the victim with a baseball bat, causing serious injury.
        Witnesses testified that they saw the defendant approach the victim from behind and
        strike them without provocation. The defendant claims they acted in self-defense,
        but no evidence supports this claim.
        """,
        "summary": """
        This case involves a second-degree assault charge where the defendant allegedly struck
        the victim with a baseball bat, causing serious injury. The defendant claims self-defense,
        but witnesses testified that the attack was unprovoked.
        """,
        "classification": {
            "criminal": 0.85,
            "urgent": 0.65,
            "civil": 0.15,
            "family": 0.05,
            "confidence": {
                "primary_category": "criminal",
                "primary_confidence": 0.85,
                "secondary_categories": ["urgent"],
                "uncertainty_score": 0.15
            }
        },
        "case_id": "CASE-2023-001",
        "jurisdiction": "Supreme Court"
    },
    {
        "text": """
        The plaintiff alleges that the defendant breached a contract for the sale of a commercial property.
        According to the plaintiff, the parties entered into a written agreement on March 10, 2023, whereby
        the defendant agreed to sell the property for $1.5 million. The plaintiff paid a deposit of $150,000,
        but the defendant failed to complete the sale as agreed. The defendant contends that the contract
        was void due to mutual mistake regarding the property's zoning status.
        """,
        "summary": """
        This case involves a breach of contract claim for the sale of a commercial property. The plaintiff
        paid a deposit but the defendant failed to complete the sale, claiming the contract was void due
        to mutual mistake regarding zoning.
        """,
        "classification": {
            "civil": 0.92,
            "contract": 0.88,
            "property": 0.75,
            "criminal": 0.05,
            "family": 0.03,
            "confidence": {
                "primary_category": "civil",
                "primary_confidence": 0.92,
                "secondary_categories": ["contract", "property"],
                "uncertainty_score": 0.08
            }
        },
        "case_id": "CASE-2023-002",
        "jurisdiction": "District Court"
    },
    {
        "text": """
        The petitioner seeks custody of the minor child, alleging that the respondent has engaged in
        behavior that is harmful to the child's well-being. The petitioner claims that the respondent
        has a history of substance abuse and has been inconsistent with visitation schedules. The respondent
        denies these allegations and contends that the petitioner is attempting to alienate the child
        from them. Both parties have submitted evidence from mental health professionals supporting
        their respective positions.
        """,
        "summary": """
        This case involves a custody dispute where the petitioner alleges the respondent's behavior
        is harmful to the child. The respondent denies these allegations and claims the petitioner
        is attempting to alienate the child.
        """,
        "classification": {
            "family": 0.95,
            "custody": 0.90,
            "urgent": 0.45,
            "criminal": 0.05,
            "civil": 0.10,
            "confidence": {
                "primary_category": "family",
                "primary_confidence": 0.95,
                "secondary_categories": ["custody"],
                "uncertainty_score": 0.05
            }
        },
        "case_id": "CASE-2023-003",
        "jurisdiction": "Family Court"
    }
]

# Scroll phases to test
SCROLL_PHASES = ["dawn", "noon", "dusk", "night"]

# Languages to test
LANGUAGES = ["English", "Spanish", "French", "German"]

def generate_example_judgments():
    """Generate example judgments with varying scroll contexts."""
    # Initialize the drafter
    drafter = JudgmentDrafter()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate judgments for each case, scroll phase, and language
    for case in SAMPLE_CASES:
        case_id = case["case_id"]
        
        for phase in SCROLL_PHASES:
            # Mock the scroll time function to return the desired phase
            with patch("backend.models.judgment_drafter.get_scroll_time") as mock:
                mock.return_value = {
                    "phase": phase,
                    "is_active": phase in ["dawn", "dusk"],
                    "time_remaining": "2 hours",
                    "next_phase": SCROLL_PHASES[(SCROLL_PHASES.index(phase) + 1) % 4],
                    "gate": 3,
                    "gate_name": "Gate of Courage",
                    "scroll_day": 42,
                    "solar_hour": 6,
                    "enano_pulse": 67
                }
                
                for language in LANGUAGES:
                    # Generate the judgment
                    judgment = drafter.generate_judgment(
                        text=case["text"],
                        summary=case["summary"],
                        classification=case["classification"],
                        case_id=case_id,
                        jurisdiction=case["jurisdiction"],
                        language=language
                    )
                    
                    # Save the judgment to a file
                    filename = f"{case_id}_{phase}_{language.lower()}.txt"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(judgment)
                    
                    print(f"Generated judgment for {case_id} in {phase} phase in {language}: {filepath}")
    
    print(f"All example judgments generated in {output_dir}")

if __name__ == "__main__":
    from unittest.mock import patch
    generate_example_judgments() 
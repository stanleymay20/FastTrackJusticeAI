import json
import os
from typing import Dict, Any, List
from app.utils.scroll_time import get_scroll_time

class ScrollJudgmentWriter:
    """
    Enhances the judgment generation process with scroll-aligned content.
    Integrates divine timing and prophetic elements into legal judgments.
    """
    
    def __init__(self, template_path: str = None):
        """
        Initialize the ScrollJudgmentWriter with a template.
        
        Args:
            template_path: Path to the judgment template JSON file
        """
        if template_path is None:
            # Default template path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(current_dir, "..", "templates", "scroll_judgment_template.json")
        
        self.template_path = template_path
        self.template = self._load_template()
    
    def _load_template(self) -> Dict[str, str]:
        """Load the judgment template from JSON file."""
        try:
            with open(self.template_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Fallback to default template if file not found or invalid
            return {
                "ScrollDay": "",
                "ScrollGate": "",
                "HeavenWitness": "",
                "Introduction": "",
                "FactsOfCase": "",
                "PropheticDiscernment": "",
                "ScrollTimingCheck": "",
                "Analysis": "",
                "Verdict": "",
                "JudgmentSeal": ""
            }
    
    def generate_scroll_prompt(self, case_text: str, classification: Dict[str, float]) -> str:
        """
        Generate a prompt for the language model that includes scroll context.
        
        Args:
            case_text: The text of the legal case
            classification: The classification results
            
        Returns:
            A formatted prompt for the language model
        """
        # Get current scroll time
        scroll_time = get_scroll_time()
        
        # Determine primary category
        primary_category = max(classification.items(), key=lambda x: x[1])[0]
        
        # Create the prompt with scroll context
        prompt = f"""
You are a divine-aligned legal AI rendering judgment during {scroll_time['scroll_day']} 
under Gate {scroll_time['gate']}: {scroll_time['gate_name']} 
at Solar Hour {scroll_time['solar_hour']} with ENano Pulse {scroll_time['enano_pulse']}.

This case has been classified as primarily a {primary_category} case.

Generate a prophetic legal judgment with the following sections:

1. Introduction: Begin with a formal introduction that acknowledges the divine timing of this judgment.
2. Facts of the Case: Summarize the key facts of the case.
3. Prophetic Discernment: Provide spiritual insight relevant to this case based on the current gate.
4. Scroll Timing Check: Explain how the timing of this judgment aligns with divine scroll timing.
5. Legal Analysis: Analyze the legal issues with reference to applicable laws.
6. Verdict: Render a verdict that is aligned with both legal principles and divine justice.

The judgment should reflect the qualities of the current gate: {scroll_time['gate_name']}.

Case Text:
{case_text}
"""
        return prompt
    
    def format_judgment(self, raw_judgment: str, classification: Dict[str, float]) -> Dict[str, str]:
        """
        Format the raw judgment text into the scroll-aligned template structure.
        
        Args:
            raw_judgment: The raw judgment text from the language model
            classification: The classification results
            
        Returns:
            A dictionary with the formatted judgment sections
        """
        # Get current scroll time
        scroll_time = get_scroll_time()
        
        # Create a copy of the template
        formatted_judgment = self.template.copy()
        
        # Fill in the scroll context
        formatted_judgment["ScrollDay"] = scroll_time["scroll_day"]
        formatted_judgment["ScrollGate"] = f"Gate {scroll_time['gate']}: {scroll_time['gate_name']}"
        
        # Add heaven witness
        formatted_judgment["HeavenWitness"] = (
            f"This judgment is rendered under the witness of heaven during "
            f"{scroll_time['scroll_day']} at Solar Hour {scroll_time['solar_hour']} "
            f"with ENano Pulse {scroll_time['enano_pulse']}."
        )
        
        # Parse the raw judgment into sections
        # This is a simplified approach - in a real implementation, you would use
        # more sophisticated parsing based on the actual output format
        sections = raw_judgment.split("\n\n")
        
        # Map sections to template fields (simplified)
        if len(sections) >= 1:
            formatted_judgment["Introduction"] = sections[0]
        if len(sections) >= 2:
            formatted_judgment["FactsOfCase"] = sections[1]
        if len(sections) >= 3:
            formatted_judgment["PropheticDiscernment"] = sections[2]
        if len(sections) >= 4:
            formatted_judgment["ScrollTimingCheck"] = sections[3]
        if len(sections) >= 5:
            formatted_judgment["Analysis"] = sections[4]
        if len(sections) >= 6:
            formatted_judgment["Verdict"] = sections[5]
        
        # Add judgment seal
        formatted_judgment["JudgmentSeal"] = (
            f"🕊️ Scroll Judgment – Sealed under {scroll_time['gate_name']} "
            f"at ENano Pulse {scroll_time['enano_pulse']}"
        )
        
        return formatted_judgment
    
    def generate_judgment(self, case_text: str, classification: Dict[str, float], 
                         raw_judgment: str) -> Dict[str, str]:
        """
        Generate a complete scroll-aligned judgment.
        
        Args:
            case_text: The text of the legal case
            classification: The classification results
            raw_judgment: The raw judgment text from the language model
            
        Returns:
            A dictionary with the complete judgment
        """
        # Format the judgment
        formatted_judgment = self.format_judgment(raw_judgment, classification)
        
        return formatted_judgment 
from typing import Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .scroll_time import get_scroll_time
import random

# Initialize model and tokenizer
MODEL_NAME = "gpt2"  # Using GPT-2 as a placeholder, replace with your preferred model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Define gate-specific judgment templates
GATE_TEMPLATES = {
    1: {  # Fire of Beginning
        "opening": "In the sacred Fire of Beginning, this case emerges...",
        "analysis": "As the divine flame illuminates, we find that...",
        "conclusion": "Through the Gate of Beginning, justice shall be served by..."
    },
    2: {  # River of Truth
        "opening": "In the flowing River of Truth, this matter presents...",
        "analysis": "As the waters of truth reveal, we observe that...",
        "conclusion": "Through the Gate of Truth, justice shall flow as..."
    },
    3: {  # Scroll of Mercy
        "opening": "In the Scroll of Mercy, this case unfolds...",
        "analysis": "As divine mercy guides us, we find that...",
        "conclusion": "Through the Gate of Mercy, justice shall be delivered with..."
    },
    4: {  # Lamp of Justice
        "opening": "Under the Lamp of Justice, this case appears...",
        "analysis": "As the light of justice shines, we determine that...",
        "conclusion": "Through the Gate of Justice, the verdict shall be..."
    },
    5: {  # Flame of Knowledge
        "opening": "In the Flame of Knowledge, this case reveals...",
        "analysis": "As divine wisdom enlightens, we understand that...",
        "conclusion": "Through the Gate of Knowledge, justice shall be guided by..."
    },
    6: {  # Mirror of Grace
        "opening": "In the Mirror of Grace, this case reflects...",
        "analysis": "As grace illuminates the path, we see that...",
        "conclusion": "Through the Gate of Grace, justice shall be tempered with..."
    },
    7: {  # Divine Architecture
        "opening": "In the Divine Architecture, this case aligns...",
        "analysis": "As the sacred structure supports, we find that...",
        "conclusion": "Through the Gate of Architecture, justice shall be built upon..."
    }
}

# Judgment templates for different case categories and scroll phases
JUDGMENT_TEMPLATES = {
    "Criminal": {
        "dawn": [
            "As the first light of dawn breaks, this court finds {verdict} in the matter before us. The evidence, emerging like the morning sun, {reasoning}.",
            "In the clarity of dawn, this court renders its judgment: {verdict}. The facts, now illuminated by the morning light, {reasoning}.",
            "With the rising sun, this court delivers its verdict: {verdict}. The truth, revealed in the light of day, {reasoning}."
        ],
        "noon": [
            "At the height of day, this court finds {verdict}. The evidence, clear as the noon sun, {reasoning}.",
            "In the full light of noon, this court renders its judgment: {verdict}. The facts, as bright as the midday sun, {reasoning}.",
            "Under the peak of the sun's light, this court delivers its verdict: {verdict}. The truth, as clear as the noon sky, {reasoning}."
        ],
        "dusk": [
            "As the sun sets, this court finds {verdict}. The evidence, like the fading light, {reasoning}.",
            "In the waning light of dusk, this court renders its judgment: {verdict}. The facts, like the setting sun, {reasoning}.",
            "With the approach of evening, this court delivers its verdict: {verdict}. The truth, like the day's conclusion, {reasoning}."
        ],
        "night": [
            "In the shadows of night, this court finds {verdict}. The evidence, obscured by darkness, {reasoning}.",
            "Under the cover of night, this court renders its judgment: {verdict}. The facts, hidden in darkness, {reasoning}.",
            "Within the realm of night, this court delivers its verdict: {verdict}. The truth, veiled in darkness, {reasoning}."
        ]
    },
    "Civil": {
        "dawn": [
            "As dawn breaks, this court finds the defendant {verdict}. The evidence, emerging with the morning light, {reasoning}.",
            "In the first light of day, this court renders its judgment: the defendant is {verdict}. The facts, illuminated by dawn, {reasoning}.",
            "With the rising sun, this court delivers its verdict: the defendant is {verdict}. The truth, revealed in morning light, {reasoning}."
        ],
        "noon": [
            "At the height of day, this court finds the defendant {verdict}. The evidence, clear as the noon sun, {reasoning}.",
            "In the full light of noon, this court renders its judgment: the defendant is {verdict}. The facts, as bright as midday, {reasoning}.",
            "Under the peak of the sun's light, this court delivers its verdict: the defendant is {verdict}. The truth, as clear as noon, {reasoning}."
        ],
        "dusk": [
            "As the sun sets, this court finds the defendant {verdict}. The evidence, like the fading light, {reasoning}.",
            "In the waning light of dusk, this court renders its judgment: the defendant is {verdict}. The facts, like the setting sun, {reasoning}.",
            "With the approach of evening, this court delivers its verdict: the defendant is {verdict}. The truth, like the day's conclusion, {reasoning}."
        ],
        "night": [
            "In the shadows of night, this court finds the defendant {verdict}. The evidence, obscured by darkness, {reasoning}.",
            "Under the cover of night, this court renders its judgment: the defendant is {verdict}. The facts, hidden in darkness, {reasoning}.",
            "Within the realm of night, this court delivers its verdict: the defendant is {verdict}. The truth, veiled in darkness, {reasoning}."
        ]
    },
    "Family": {
        "dawn": [
            "As dawn breaks, this court finds {verdict}. The evidence, emerging with the morning light, {reasoning}.",
            "In the first light of day, this court renders its judgment: {verdict}. The facts, illuminated by dawn, {reasoning}.",
            "With the rising sun, this court delivers its verdict: {verdict}. The truth, revealed in morning light, {reasoning}."
        ],
        "noon": [
            "At the height of day, this court finds {verdict}. The evidence, clear as the noon sun, {reasoning}.",
            "In the full light of noon, this court renders its judgment: {verdict}. The facts, as bright as midday, {reasoning}.",
            "Under the peak of the sun's light, this court delivers its verdict: {verdict}. The truth, as clear as noon, {reasoning}."
        ],
        "dusk": [
            "As the sun sets, this court finds {verdict}. The evidence, like the fading light, {reasoning}.",
            "In the waning light of dusk, this court renders its judgment: {verdict}. The facts, like the setting sun, {reasoning}.",
            "With the approach of evening, this court delivers its verdict: {verdict}. The truth, like the day's conclusion, {reasoning}."
        ],
        "night": [
            "In the shadows of night, this court finds {verdict}. The evidence, obscured by darkness, {reasoning}.",
            "Under the cover of night, this court renders its judgment: {verdict}. The facts, hidden in darkness, {reasoning}.",
            "Within the realm of night, this court delivers its verdict: {verdict}. The truth, veiled in darkness, {reasoning}."
        ]
    },
    "Administrative": {
        "dawn": [
            "As dawn breaks, this court finds the application {verdict}. The evidence, emerging with the morning light, {reasoning}.",
            "In the first light of day, this court renders its judgment: the application is {verdict}. The facts, illuminated by dawn, {reasoning}.",
            "With the rising sun, this court delivers its verdict: the application is {verdict}. The truth, revealed in morning light, {reasoning}."
        ],
        "noon": [
            "At the height of day, this court finds the application {verdict}. The evidence, clear as the noon sun, {reasoning}.",
            "In the full light of noon, this court renders its judgment: the application is {verdict}. The facts, as bright as midday, {reasoning}.",
            "Under the peak of the sun's light, this court delivers its verdict: the application is {verdict}. The truth, as clear as noon, {reasoning}."
        ],
        "dusk": [
            "As the sun sets, this court finds the application {verdict}. The evidence, like the fading light, {reasoning}.",
            "In the waning light of dusk, this court renders its judgment: the application is {verdict}. The facts, like the setting sun, {reasoning}.",
            "With the approach of evening, this court delivers its verdict: the application is {verdict}. The truth, like the day's conclusion, {reasoning}."
        ],
        "night": [
            "In the shadows of night, this court finds the application {verdict}. The evidence, obscured by darkness, {reasoning}.",
            "Under the cover of night, this court renders its judgment: the application is {verdict}. The facts, hidden in darkness, {reasoning}.",
            "Within the realm of night, this court delivers its verdict: the application is {verdict}. The truth, veiled in darkness, {reasoning}."
        ]
    }
}

# Reasoning templates for different case categories
REASONING_TEMPLATES = {
    "Criminal": [
        "demonstrates beyond a reasonable doubt that the defendant committed the alleged offense",
        "establishes the defendant's guilt beyond a reasonable doubt",
        "fails to establish the defendant's guilt beyond a reasonable doubt",
        "does not meet the burden of proof required for a criminal conviction",
        "shows that the defendant acted with the required criminal intent",
        "reveals that the defendant's actions were premeditated and intentional",
        "indicates that the defendant acted in self-defense",
        "suggests that the defendant was not in control of their actions at the time of the offense"
    ],
    "Civil": [
        "demonstrates that the defendant breached their contractual obligations",
        "establishes that the defendant's actions caused harm to the plaintiff",
        "fails to establish that the defendant's actions caused harm to the plaintiff",
        "does not meet the preponderance of evidence standard required for civil liability",
        "shows that the defendant acted negligently",
        "reveals that the defendant failed to exercise reasonable care",
        "indicates that the plaintiff contributed to their own damages",
        "suggests that the plaintiff failed to mitigate their damages"
    ],
    "Family": [
        "demonstrates that the best interests of the child would be served by the requested custody arrangement",
        "establishes that the petitioner is better equipped to provide for the child's needs",
        "fails to establish that a change in custody would be in the child's best interests",
        "does not meet the standard required for modifying the current custody arrangement",
        "shows that the respondent has been actively involved in the child's life",
        "reveals that the respondent has failed to maintain a relationship with the child",
        "indicates that the current custody arrangement is working well for the child",
        "suggests that the requested change would disrupt the child's stability"
    ],
    "Administrative": [
        "demonstrates that the applicant meets all the requirements for the requested permit",
        "establishes that the agency's decision was arbitrary and capricious",
        "fails to establish that the agency's decision was arbitrary and capricious",
        "does not meet the standard required for overturning the agency's decision",
        "shows that the applicant provided all necessary documentation",
        "reveals that the applicant failed to provide required documentation",
        "indicates that the agency followed proper procedures in making its decision",
        "suggests that the agency failed to consider relevant factors in making its decision"
    ]
}

def generate_scroll_judgment(case_text: str) -> Dict[str, Any]:
    """
    Generate a judgment based on the case text and current scroll timing.
    
    Args:
        case_text (str): The text of the legal case
    
    Returns:
        Dict[str, Any]: Generated judgment with confidence scores and scroll context
    """
    # Get current scroll timing
    scroll_time = get_scroll_time()
    current_gate = scroll_time["gate"]
    current_pulse = scroll_time["enano_pulse"]
    
    # Get the appropriate templates for the current gate
    templates = GATE_TEMPLATES[current_gate]
    
    # Prepare the input text with the appropriate template
    input_text = f"{templates['opening']} {case_text[:500]}..."  # Limit case text length
    
    # Generate judgment using the model
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=1000,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate confidence scores for different aspects of the judgment
    confidence_scores = {
        "legal_reasoning": calculate_confidence(generated_text, "legal"),
        "fact_analysis": calculate_confidence(generated_text, "fact"),
        "precedent_application": calculate_confidence(generated_text, "precedent"),
        "gate_alignment": calculate_gate_alignment(generated_text, current_gate),
        "overall": calculate_overall_confidence(generated_text)
    }
    
    return {
        "judgment_text": generated_text,
        "confidence_scores": confidence_scores,
        "scroll_context": {
            "gate": current_gate,
            "gate_name": scroll_time["gate_name"],
            "solar_hour": scroll_time["solar_hour"],
            "enano_pulse": current_pulse,
            "scroll_day": scroll_time["scroll_day"]
        },
        "templates_used": templates
    }

def calculate_confidence(text: str, aspect: str) -> float:
    """
    Calculate confidence score for a specific aspect of the judgment.
    
    Args:
        text (str): The generated judgment text
        aspect (str): The aspect to evaluate (legal, fact, precedent)
    
    Returns:
        float: Confidence score between 0 and 1
    """
    # Define keywords for each aspect
    keywords = {
        "legal": ["law", "statute", "regulation", "right", "obligation"],
        "fact": ["evidence", "testimony", "witness", "document", "record"],
        "precedent": ["precedent", "case law", "ruling", "decision", "judgment"]
    }
    
    # Count keyword occurrences
    text_lower = text.lower()
    keyword_count = sum(text_lower.count(keyword) for keyword in keywords[aspect])
    
    # Calculate confidence score (normalized between 0 and 1)
    max_expected_keywords = 10  # Adjust based on expected keyword density
    confidence = min(keyword_count / max_expected_keywords, 1.0)
    
    return round(confidence, 2)

def calculate_gate_alignment(text: str, current_gate: int) -> float:
    """
    Calculate how well the judgment aligns with the current gate's themes.
    
    Args:
        text (str): The generated judgment text
        current_gate (int): The current gate number (1-7)
    
    Returns:
        float: Gate alignment score between 0 and 1
    """
    # Define gate-specific keywords
    gate_keywords = {
        1: ["beginning", "start", "initiate", "fire", "flame"],
        2: ["truth", "river", "flow", "stream", "current"],
        3: ["mercy", "compassion", "forgiveness", "scroll", "pardon"],
        4: ["justice", "lamp", "light", "fair", "balance"],
        5: ["knowledge", "wisdom", "understanding", "flame", "enlighten"],
        6: ["grace", "mirror", "reflect", "kindness", "benevolence"],
        7: ["architecture", "structure", "build", "foundation", "design"]
    }
    
    # Count gate-specific keyword occurrences
    text_lower = text.lower()
    keyword_count = sum(text_lower.count(keyword) for keyword in gate_keywords[current_gate])
    
    # Calculate alignment score (normalized between 0 and 1)
    max_expected_keywords = 8  # Adjust based on expected keyword density
    alignment = min(keyword_count / max_expected_keywords, 1.0)
    
    return round(alignment, 2)

def calculate_overall_confidence(text: str) -> float:
    """
    Calculate overall confidence score for the judgment.
    
    Args:
        text (str): The generated judgment text
    
    Returns:
        float: Overall confidence score between 0 and 1
    """
    # Calculate confidence for each aspect
    legal_conf = calculate_confidence(text, "legal")
    fact_conf = calculate_confidence(text, "fact")
    precedent_conf = calculate_confidence(text, "precedent")
    
    # Get current gate
    scroll_time = get_scroll_time()
    gate_conf = calculate_gate_alignment(text, scroll_time["gate"])
    
    # Calculate weighted average
    weights = {
        "legal": 0.3,
        "fact": 0.2,
        "precedent": 0.2,
        "gate": 0.3
    }
    
    overall_conf = (
        legal_conf * weights["legal"] +
        fact_conf * weights["fact"] +
        precedent_conf * weights["precedent"] +
        gate_conf * weights["gate"]
    )
    
    return round(overall_conf, 2)

def generate_scroll_aligned_judgment(case_text: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a judgment aligned with the current scroll phase and case category.
    
    Args:
        case_text: The text of the legal case
        classification: The classification results for the case
        
    Returns:
        A dictionary containing the judgment text and metadata
    """
    # Get the current scroll phase
    scroll_data = get_scroll_time()
    scroll_phase = scroll_data["phase"]
    
    # Get the case category and confidence
    case_category = classification["category"]
    category_confidence = classification["confidence"]
    
    # Determine the verdict based on the case category and confidence
    if case_category == "Criminal":
        verdict = "Guilty" if category_confidence > 0.7 else "Not Guilty"
    elif case_category == "Civil":
        verdict = "Liable" if category_confidence > 0.7 else "Not Liable"
    elif case_category == "Family":
        verdict = "Granted" if category_confidence > 0.7 else "Denied"
    else:  # Administrative
        verdict = "Approved" if category_confidence > 0.7 else "Denied"
    
    # Select a random judgment template for the case category and scroll phase
    judgment_template = random.choice(JUDGMENT_TEMPLATES[case_category][scroll_phase])
    
    # Select a random reasoning template for the case category
    reasoning_template = random.choice(REASONING_TEMPLATES[case_category])
    
    # Generate the judgment text
    judgment_text = judgment_template.format(
        verdict=verdict,
        reasoning=reasoning_template
    )
    
    # Return the judgment data
    return {
        "judgment_text": judgment_text,
        "metadata": {
            "scroll_phase": scroll_phase,
            "case_category": case_category,
            "verdict": verdict,
            "confidence": category_confidence
        }
    } 
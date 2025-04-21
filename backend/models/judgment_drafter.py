from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch
from typing import Dict, Optional, List
import os
from dotenv import load_dotenv
import re
import json
from datetime import datetime
import logging
from ..app.utils.scroll_time import get_scroll_time
from ..app.utils.legal_precedent import LegalPrecedentManager, Precedent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Legal terminology categories
LEGAL_TERMINOLOGY = {
    "jurisdiction": [
        "jurisdiction", "venue", "forum", "court", "tribunal", "authority", 
        "competence", "power", "sovereignty", "territorial", "extraterritorial"
    ],
    "procedure": [
        "procedure", "process", "filing", "motion", "pleading", "discovery", 
        "deposition", "testimony", "evidence", "hearing", "trial", "appeal", 
        "judgment", "verdict", "sentence", "remedy", "relief", "injunction"
    ],
    "substantive": [
        "liability", "negligence", "breach", "contract", "tort", "crime", 
        "constitutional", "statutory", "common law", "precedent", "doctrine", 
        "right", "duty", "obligation", "standard", "reasonable", "proximate"
    ],
    "conclusion": [
        "therefore", "consequently", "thus", "accordingly", "for the foregoing reasons", 
        "it is hereby ordered", "it is hereby adjudged", "judgment is entered", 
        "plaintiff prevails", "defendant prevails", "case dismissed", "case remanded"
    ]
}

# Scroll phase enrichment for judgment tone
SCROLL_PHASE_ENRICHMENT = {
    "dawn": {
        "tone": "awakening and clarity",
        "emphasis": "new beginnings and fresh perspectives",
        "gate_1": "bold decree with clarity of purpose",
        "gate_3": "courageous judgment with balanced wisdom",
        "gate_5": "harmonious resolution with divine insight",
        "gate_7": "merciful judgment with compassionate understanding"
    },
    "noon": {
        "tone": "balance and justice",
        "emphasis": "equilibrium and fairness",
        "gate_1": "authoritative decree with balanced wisdom",
        "gate_3": "courageous judgment with divine insight",
        "gate_5": "harmonious resolution with compassionate understanding",
        "gate_7": "merciful judgment with clarity of purpose"
    },
    "dusk": {
        "tone": "resolution and closure",
        "emphasis": "conclusion and finality",
        "gate_1": "bold decree with compassionate understanding",
        "gate_3": "courageous judgment with clarity of purpose",
        "gate_5": "harmonious resolution with balanced wisdom",
        "gate_7": "merciful judgment with divine insight"
    },
    "night": {
        "tone": "reflection and wisdom",
        "emphasis": "contemplation and deeper understanding",
        "gate_1": "bold decree with divine insight",
        "gate_3": "courageous judgment with compassionate understanding",
        "gate_5": "harmonious resolution with clarity of purpose",
        "gate_7": "merciful judgment with balanced wisdom"
    }
}

class JudgmentDrafter:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize model and tokenizer
        self.model_name = os.getenv("JUDGMENT_MODEL", "gpt2-medium")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set generation parameters
        self.max_length = 2048
        self.min_length = 512
        self.num_beams = 4
        self.length_penalty = 1.0
        self.temperature = 0.7
        
        # Initialize precedent manager
        self.precedent_manager = LegalPrecedentManager()
        
        # Initialize cache
        self.cache = {}
        
        # Create log directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
    def preprocess_input(self, text: str, summary: str, classification: Dict[str, float], 
                        case_id: Optional[str] = None, jurisdiction: Optional[str] = None,
                        language: Optional[str] = None) -> str:
        """
        Prepare input for judgment generation.
        """
        # Extract primary category
        primary_category = classification["confidence"]["primary_category"]
        
        # Get scroll phase information
        scroll_info = get_scroll_time()
        scroll_phase = scroll_info["phase"]
        scroll_is_active = scroll_info["scroll_is_active"]
        gate = scroll_info.get("gate", 1)
        
        # Find relevant precedents
        relevant_precedents = self.precedent_manager.find_relevant_precedents(
            text,
            jurisdiction=jurisdiction,
            category=primary_category,
            scroll_phase=scroll_phase
        )
        
        # Extract legal principles
        legal_principles = self.precedent_manager.extract_legal_principles(relevant_precedents)
        
        # Format citations
        citations = self.precedent_manager.format_citations(relevant_precedents)
        
        # Create structured prompt
        prompt = f"""
        Case ID: {case_id or 'N/A'}
        Jurisdiction: {jurisdiction or 'General'}
        Case Type: {primary_category}
        Scroll Phase: {scroll_phase} (Active: {scroll_is_active})
        Gate: {gate}
        
        Case Summary:
        {summary}
        
        Key Facts:
        {text[:500]}...
        
        Relevant Precedents:
        {chr(10).join(f"- {citation}" for citation in citations)}
        
        Legal Principles:
        Substantive:
        {chr(10).join(f"- {p}" for p in legal_principles["substantive"])}
        
        Procedural:
        {chr(10).join(f"- {p}" for p in legal_principles["procedural"])}
        
        Constitutional:
        {chr(10).join(f"- {p}" for p in legal_principles["constitutional"])}
        
        Scroll-Aligned:
        {chr(10).join(f"- {p}" for p in legal_principles["scroll_aligned"])}
        
        Generate a legal judgment that:
        1. Applies relevant precedents
        2. Follows established legal principles
        3. Maintains scroll phase alignment
        4. Provides clear reasoning and citations
        5. Concludes with a well-supported verdict
        """
        
        # Add language translation if needed
        if language and language.lower() != "english":
            prompt = f"Translate to {language}:\n" + prompt
        
        return prompt.strip()
        
    def generate_judgment(self, text: str, summary: str, classification: Dict[str, float],
                         case_id: Optional[str] = None, jurisdiction: Optional[str] = None,
                         language: Optional[str] = None) -> str:
        """
        Generate a structured legal judgment.
        """
        try:
            # Check cache first
            cache_key = hash((text[:100], summary, str(classification), case_id, jurisdiction, language))
            if cache_key in self.cache:
                logger.info("Using cached judgment")
                return self.cache[cache_key]
            
            # Prepare input
            prompt = self.preprocess_input(text, summary, classification, case_id, jurisdiction, language)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            # Generate judgment
            judgment_ids = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                temperature=self.temperature,
                early_stopping=True
            )
            
            # Decode and clean up judgment
            judgment = self.tokenizer.decode(judgment_ids[0], skip_special_tokens=True)
            
            # Post-process judgment
            judgment = self.postprocess_judgment(judgment)
            
            # Cache the result
            self.cache[cache_key] = judgment
            
            # Log the generation
            self.log_judgment_generation(text, summary, classification, judgment)
            
            return judgment
            
        except Exception as e:
            raise Exception(f"Error generating judgment: {str(e)}")
            
    def postprocess_judgment(self, judgment: str) -> str:
        """
        Post-process the generated judgment.
        """
        # Clean up whitespace
        judgment = re.sub(r'\s+', ' ', judgment).strip()
        
        # Ensure proper paragraph breaks
        judgment = re.sub(r'([.!?])\s+', r'\1\n\n', judgment)
        
        # Format citations
        judgment = re.sub(r'\[(\d{4})\]', r' [\1] ', judgment)
        
        return judgment
        
    def evaluate_judgment(self, judgment: str, reference: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate the quality of the generated judgment.
        """
        metrics = {
            "length": len(judgment.split()),
            "has_required_sections": self._check_required_sections(judgment),
            "legal_terminology": self._check_legal_terminology(judgment)
        }
        
        if reference:
            # Add ROUGE scores if reference is provided
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, judgment)
            metrics.update({
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            })
            
        return metrics
        
    def _check_required_sections(self, judgment: str) -> Dict[str, bool]:
        """
        Check if judgment contains all required sections.
        """
        required_sections = [
            "Introduction",
            "Facts of the Case",
            "Legal Issues",
            "Analysis and Reasoning",
            "Conclusion and Verdict"
        ]
        
        return {
            section: section in judgment
            for section in required_sections
        }
        
    def _check_legal_terminology(self, judgment: str) -> Dict[str, int]:
        """
        Check for presence of legal terminology.
        """
        counts = {}
        for category, terms in LEGAL_TERMINOLOGY.items():
            counts[category] = sum(1 for term in terms if term in judgment.lower())
        return counts
        
    def log_judgment_generation(self, text: str, summary: str, classification: Dict[str, float], judgment: str) -> None:
        """
        Log the judgment generation.
        """
        # Define the log file path
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        log_file = os.path.join(log_dir, "judgment_generations.log")
        
        # Create log entry
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "summary": summary,
            "classification": classification,
            "judgment_preview": judgment[:200] + "..." if len(judgment) > 200 else judgment,
            "legal_terminology": self._check_legal_terminology(judgment)
        }
        
        # Append to log file
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception as e:
            logger.error(f"Error logging judgment generation: {str(e)}")
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the model and tokenizer to disk.
        
        Args:
            output_dir: The directory to save the model to
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}") 
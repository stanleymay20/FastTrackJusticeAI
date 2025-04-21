import os
import re
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from .legal_precedent import LegalPrecedentManager
from .legal_knowledge_augmenter import LegalKnowledgeAugmenter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JudgmentDrafter:
    """
    Drafts legal judgments using AI, with support for scroll phase awareness,
    legal terminology, and precedent-based reasoning.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        cache_dir: str = "cache/judgments",
        precedents_file: str = "precedents.json",
        language: str = "en",
        use_rag: bool = True
    ):
        """
        Initialize the JudgmentDrafter.
        
        Args:
            model_name: Name of the language model to use
            cache_dir: Directory to store judgment cache
            precedents_file: Path to the precedents JSON file
            language: Language code for judgment generation
            use_rag: Whether to use RAG for knowledge augmentation
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.language = language
        self.use_rag = use_rag
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize precedent manager and knowledge augmenter if RAG is enabled
        if use_rag:
            self.precedent_manager = LegalPrecedentManager(precedents_file=precedents_file)
            self.knowledge_augmenter = LegalKnowledgeAugmenter(
                precedent_manager=self.precedent_manager,
                max_precedents=3,
                min_similarity_score=0.7
            )
        else:
            self.precedent_manager = None
            self.knowledge_augmenter = None
            
        # Legal terminology categories
        self.legal_terminology = {
            "procedural": [
                "jurisdiction", "venue", "standing", "mootness", "ripeness",
                "exhaustion", "waiver", "estoppel", "res judicata", "collateral estoppel"
            ],
            "substantive": [
                "liability", "negligence", "breach", "damages", "remedy",
                "injunction", "specific performance", "rescission", "reformation"
            ],
            "constitutional": [
                "due process", "equal protection", "free speech", "establishment clause",
                "takings clause", "commerce clause", "necessary and proper"
            ],
            "criminal": [
                "mens rea", "actus reus", "beyond reasonable doubt", "probable cause",
                "reasonable suspicion", "exclusionary rule", "miranda rights"
            ]
        }
        
        # Scroll phase enrichment for judgment tone
        self.scroll_phase_enrichment = {
            "dawn": {
                "tone": "hopeful and forward-looking",
                "emphasis": "new beginnings and fresh perspectives",
                "language": "illuminating, revealing, emerging"
            },
            "noon": {
                "tone": "clear and decisive",
                "emphasis": "clarity and directness",
                "language": "definitive, conclusive, established"
            },
            "dusk": [
                "tone": "reflective and contemplative",
                "emphasis": "weighing competing interests",
                "language": "balanced, considered, measured"
            ],
            "night": {
                "tone": "protective and cautious",
                "emphasis": "safeguarding rights and preventing harm",
                "language": "protective, vigilant, safeguarding"
            }
        }
        
    def preprocess_input(
        self,
        case_text: str,
        case_id: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        scroll_phase: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Preprocess the input case text and metadata.
        
        Args:
            case_text: The case description or facts
            case_id: Optional case identifier
            jurisdiction: Optional jurisdiction code
            scroll_phase: Optional scroll phase (dawn, noon, dusk, night)
            
        Returns:
            Dictionary containing preprocessed input
        """
        # Extract key information from case text
        case_summary = self._extract_case_summary(case_text)
        
        # Determine relevant legal terminology
        terminology = self._identify_relevant_terminology(case_text)
        
        # Get scroll phase enrichment if provided
        phase_enrichment = {}
        if scroll_phase and scroll_phase in self.scroll_phase_enrichment:
            phase_enrichment = self.scroll_phase_enrichment[scroll_phase]
            
        # Retrieve relevant precedents if RAG is enabled
        precedent_knowledge = ""
        if self.use_rag and self.knowledge_augmenter:
            precedent_knowledge = self.knowledge_augmenter.format_for_prompt(
                case_text,
                scroll_phase=scroll_phase,
                jurisdiction=jurisdiction
            )
            
        return {
            "case_text": case_text,
            "case_id": case_id,
            "jurisdiction": jurisdiction,
            "scroll_phase": scroll_phase,
            "case_summary": case_summary,
            "terminology": terminology,
            "phase_enrichment": phase_enrichment,
            "precedent_knowledge": precedent_knowledge
        }
        
    def generate_judgment(
        self,
        preprocessed_input: Dict[str, Any],
        language: Optional[str] = None
    ) -> str:
        """
        Generate a legal judgment based on preprocessed input.
        
        Args:
            preprocessed_input: Dictionary containing preprocessed case information
            language: Optional language override
            
        Returns:
            Generated judgment text
        """
        # Use provided language or default
        language = language or self.language
        
        # Check cache first
        cache_key = self._generate_cache_key(preprocessed_input)
        cached_judgment = self._get_from_cache(cache_key)
        if cached_judgment:
            logger.info(f"Using cached judgment for case {preprocessed_input.get('case_id', 'unknown')}")
            return cached_judgment
            
        # Build the prompt
        prompt = self._build_judgment_prompt(preprocessed_input, language)
        
        # Generate judgment using the model
        judgment = self._call_model(prompt)
        
        # Cache the result
        self._add_to_cache(cache_key, judgment)
        
        return judgment
        
    def postprocess_judgment(self, judgment: str) -> str:
        """
        Postprocess the generated judgment.
        
        Args:
            judgment: The raw judgment text
            
        Returns:
            Postprocessed judgment
        """
        # Format citations
        judgment = self._format_citations(judgment)
        
        # Add section headers if missing
        judgment = self._ensure_section_headers(judgment)
        
        # Clean up formatting
        judgment = self._clean_formatting(judgment)
        
        return judgment
        
    def evaluate_judgment(
        self,
        judgment: str,
        case_text: str,
        preprocessed_input: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate the quality of the generated judgment.
        
        Args:
            judgment: The generated judgment
            case_text: The original case text
            preprocessed_input: The preprocessed input used for generation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # This would use metrics like ROUGE, BERTScore, etc.
        # For now, return placeholder metrics
        return {
            "coherence": 0.85,
            "legal_accuracy": 0.9,
            "terminology_usage": 0.8,
            "scroll_alignment": 0.75 if preprocessed_input.get("scroll_phase") else 1.0
        }
        
    def _extract_case_summary(self, case_text: str) -> str:
        """Extract a concise summary from the case text."""
        # Simple implementation - first paragraph or first 200 characters
        paragraphs = case_text.split("\n\n")
        if paragraphs:
            return paragraphs[0]
        return case_text[:200] + "..."
        
    def _identify_relevant_terminology(self, case_text: str) -> List[str]:
        """Identify relevant legal terminology from the case text."""
        relevant_terms = []
        for category, terms in self.legal_terminology.items():
            for term in terms:
                if term.lower() in case_text.lower():
                    relevant_terms.append(term)
        return relevant_terms
        
    def _generate_cache_key(self, preprocessed_input: Dict[str, Any]) -> str:
        """Generate a cache key for the preprocessed input."""
        # Use case_id if available, otherwise hash the case text
        if preprocessed_input.get("case_id"):
            return f"{preprocessed_input['case_id']}_{preprocessed_input.get('scroll_phase', 'none')}"
        
        # Hash the case text and metadata
        import hashlib
        text_to_hash = f"{preprocessed_input['case_text']}_{preprocessed_input.get('scroll_phase', 'none')}_{preprocessed_input.get('jurisdiction', 'none')}"
        return hashlib.md5(text_to_hash.encode()).hexdigest()
        
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Retrieve a judgment from the cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("judgment")
            except Exception as e:
                logger.error(f"Error reading from cache: {str(e)}")
        return None
        
    def _add_to_cache(self, cache_key: str, judgment: str) -> None:
        """Add a judgment to the cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "judgment": judgment,
                    "timestamp": datetime.datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
            
    def _build_judgment_prompt(
        self,
        preprocessed_input: Dict[str, Any],
        language: str
    ) -> str:
        """Build the prompt for judgment generation."""
        # Start with the basic prompt
        prompt_parts = [
            "You are a legal judgment drafter. Generate a comprehensive legal judgment based on the following case:",
            "",
            f"CASE: {preprocessed_input['case_text']}",
            ""
        ]
        
        # Add jurisdiction if provided
        if preprocessed_input.get("jurisdiction"):
            prompt_parts.append(f"JURISDICTION: {preprocessed_input['jurisdiction']}")
            prompt_parts.append("")
            
        # Add scroll phase enrichment if provided
        if preprocessed_input.get("scroll_phase") and preprocessed_input.get("phase_enrichment"):
            phase = preprocessed_input["scroll_phase"]
            enrichment = preprocessed_input["phase_enrichment"]
            prompt_parts.append(f"SCROLL PHASE: {phase}")
            prompt_parts.append(f"PHASE TONE: {enrichment.get('tone', 'neutral')}")
            prompt_parts.append(f"PHASE EMPHASIS: {enrichment.get('emphasis', 'balanced')}")
            prompt_parts.append(f"PHASE LANGUAGE: {enrichment.get('language', 'standard')}")
            prompt_parts.append("")
            
        # Add relevant legal terminology
        if preprocessed_input.get("terminology"):
            prompt_parts.append("RELEVANT LEGAL TERMINOLOGY:")
            for term in preprocessed_input["terminology"]:
                prompt_parts.append(f"- {term}")
            prompt_parts.append("")
            
        # Add precedent knowledge if available
        if preprocessed_input.get("precedent_knowledge"):
            prompt_parts.append(preprocessed_input["precedent_knowledge"])
            prompt_parts.append("")
            
        # Add language instruction
        prompt_parts.append(f"LANGUAGE: Generate the judgment in {language}.")
        prompt_parts.append("")
        
        # Add structure guidance
        prompt_parts.append("STRUCTURE THE JUDGMENT AS FOLLOWS:")
        prompt_parts.append("1. Introduction and case summary")
        prompt_parts.append("2. Statement of facts")
        prompt_parts.append("3. Issues presented")
        prompt_parts.append("4. Analysis and reasoning")
        prompt_parts.append("5. Conclusion and judgment")
        prompt_parts.append("")
        
        # Add final instruction
        prompt_parts.append("GENERATE A COMPREHENSIVE LEGAL JUDGMENT:")
        
        return "\n".join(prompt_parts)
        
    def _call_model(self, prompt: str) -> str:
        """Call the language model to generate the judgment."""
        # This would use the actual model API
        # For now, return a placeholder
        return f"GENERATED JUDGMENT FOR PROMPT: {prompt[:100]}..."
        
    def _format_citations(self, judgment: str) -> str:
        """Format legal citations in the judgment."""
        # Simple citation formatting - would be more sophisticated in production
        return judgment
        
    def _ensure_section_headers(self, judgment: str) -> str:
        """Ensure the judgment has proper section headers."""
        # Check for common section headers
        headers = ["INTRODUCTION", "FACTS", "ISSUES", "ANALYSIS", "CONCLUSION"]
        for header in headers:
            if header not in judgment:
                # Add missing header
                judgment = judgment.replace(f"{header}:", f"\n\n{header}:\n")
        return judgment
        
    def _clean_formatting(self, judgment: str) -> str:
        """Clean up formatting issues in the judgment."""
        # Remove excessive whitespace
        judgment = re.sub(r'\n{3,}', '\n\n', judgment)
        # Ensure proper paragraph spacing
        judgment = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\n\2', judgment)
        return judgment 
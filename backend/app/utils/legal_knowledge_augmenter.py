import logging
from typing import List, Dict, Any, Optional
from .legal_precedent import LegalPrecedentManager

logger = logging.getLogger(__name__)

class LegalKnowledgeAugmenter:
    """
    Augments legal judgment generation with relevant precedent knowledge using RAG (Retrieval Augmented Generation).
    Integrates with LegalPrecedentManager to retrieve and format relevant precedents for injection into prompts.
    """
    
    def __init__(
        self,
        precedent_manager: LegalPrecedentManager,
        max_precedents: int = 3,
        min_similarity_score: float = 0.7,
        include_citations: bool = True,
        include_principles: bool = True
    ):
        """
        Initialize the LegalKnowledgeAugmenter.
        
        Args:
            precedent_manager: Instance of LegalPrecedentManager for precedent retrieval
            max_precedents: Maximum number of precedents to include in augmentation
            min_similarity_score: Minimum similarity score threshold for including precedents
            include_citations: Whether to include formatted citations
            include_principles: Whether to include extracted legal principles
        """
        self.precedent_manager = precedent_manager
        self.max_precedents = max_precedents
        self.min_similarity_score = min_similarity_score
        self.include_citations = include_citations
        self.include_principles = include_principles
        
    def augment_with_precedents(
        self,
        query: str,
        scroll_phase: Optional[str] = None,
        jurisdiction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve and format relevant precedents for injection into a judgment generation prompt.
        
        Args:
            query: The case description or legal question
            scroll_phase: Optional scroll phase for phase-aware retrieval
            jurisdiction: Optional jurisdiction filter
            
        Returns:
            Dictionary containing formatted precedent knowledge for prompt injection
        """
        # Retrieve relevant precedents
        precedents = self.precedent_manager.find_relevant_precedents(
            query,
            top_k=self.max_precedents
        )
        
        # Filter by similarity score and jurisdiction
        filtered_precedents = [
            p for p in precedents 
            if p['similarity_score'] >= self.min_similarity_score
            and (not jurisdiction or p.get('jurisdiction') == jurisdiction)
        ]
        
        if not filtered_precedents:
            logger.warning(f"No relevant precedents found for query: {query}")
            return {
                "precedent_text": "",
                "citations": [],
                "principles": []
            }
            
        # Format precedents for prompt injection
        precedent_texts = []
        citations = []
        principles = []
        
        for precedent in filtered_precedents:
            # Format precedent text
            text = f"Case: {precedent['title']}\n"
            text += f"Court: {precedent['court']} ({precedent['year']})\n"
            text += f"Summary: {precedent['summary']}\n"
            if 'holding' in precedent:
                text += f"Holding: {precedent['holding']}\n"
            if 'reasoning' in precedent:
                text += f"Reasoning: {precedent['reasoning']}\n"
            precedent_texts.append(text)
            
            # Add citation if enabled
            if self.include_citations:
                citation = self.precedent_manager.format_citation(precedent)
                citations.append(citation)
                
            # Extract principles if enabled
            if self.include_principles:
                extracted_principles = self.precedent_manager.extract_legal_principles(precedent)
                principles.extend(extracted_principles)
                
        # Combine all precedent texts
        combined_text = "\n\n".join(precedent_texts)
        
        return {
            "precedent_text": combined_text,
            "citations": citations,
            "principles": list(set(principles))  # Remove duplicates
        }
        
    def format_for_prompt(
        self,
        query: str,
        scroll_phase: Optional[str] = None,
        jurisdiction: Optional[str] = None
    ) -> str:
        """
        Format retrieved precedents into a prompt-ready string.
        
        Args:
            query: The case description or legal question
            scroll_phase: Optional scroll phase for phase-aware retrieval
            jurisdiction: Optional jurisdiction filter
            
        Returns:
            Formatted string ready for injection into judgment generation prompt
        """
        augmentation = self.augment_with_precedents(
            query,
            scroll_phase,
            jurisdiction
        )
        
        if not augmentation["precedent_text"]:
            return ""
            
        # Build the formatted prompt section
        sections = ["RELEVANT LEGAL PRECEDENTS:"]
        sections.append(augmentation["precedent_text"])
        
        if augmentation["citations"]:
            sections.append("\nLEGAL CITATIONS:")
            sections.extend([f"- {citation}" for citation in augmentation["citations"]])
            
        if augmentation["principles"]:
            sections.append("\nKEY LEGAL PRINCIPLES:")
            sections.extend([f"- {principle}" for principle in augmentation["principles"]])
            
        return "\n\n".join(sections)
        
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge augmentation process.
        
        Returns:
            Dictionary containing augmentation statistics
        """
        return {
            "total_precedents": len(self.precedent_manager.precedents),
            "max_precedents": self.max_precedents,
            "min_similarity_score": self.min_similarity_score,
            "include_citations": self.include_citations,
            "include_principles": self.include_principles
        } 
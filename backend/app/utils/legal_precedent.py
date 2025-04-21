from typing import Dict, List, Optional, Tuple, Any
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Precedent:
    """Represents a legal precedent case."""
    case_id: str
    title: str
    court: str
    date: str
    summary: str
    key_principles: List[str]
    citations: List[str]
    jurisdiction: str
    category: str
    scroll_phase: Optional[str] = None

class LegalPrecedentManager:
    """
    Manages legal precedents with BERT-based semantic search capabilities.
    """
    
    def __init__(self, precedents_file: str = "precedents.json", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the LegalPrecedentManager.
        
        Args:
            precedents_file: Path to the JSON file containing precedents
            model_name: Name of the sentence-transformer model to use
        """
        self.precedents_file = precedents_file
        self.precedents: List[Dict[str, Any]] = []
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.load_precedents()
        
    def load_precedents(self) -> None:
        """Load precedents from the JSON file and initialize the FAISS index."""
        try:
            if os.path.exists(self.precedents_file):
                with open(self.precedents_file, 'r', encoding='utf-8') as f:
                    self.precedents = json.load(f)
                logger.info(f"Loaded {len(self.precedents)} precedents from {self.precedents_file}")
                self._initialize_index()
            else:
                logger.warning(f"Precedents file {self.precedents_file} not found. Starting with empty database.")
        except Exception as e:
            logger.error(f"Error loading precedents: {str(e)}")
            self.precedents = []
            
    def _initialize_index(self) -> None:
        """Initialize the FAISS index with existing precedents."""
        if not self.precedents:
            return
            
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Generate embeddings for all precedents
        texts = [f"{p['title']} {p['summary']} {p['facts']} {p['holding']}" for p in self.precedents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings
        embeddings = normalize(embeddings)
        
        # Add to index
        self.index.add(np.array(embeddings).astype('float32'))
        logger.info(f"Initialized FAISS index with {len(self.precedents)} precedents")
        
    def find_relevant_precedents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find relevant precedents using BERT-based semantic search.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of relevant precedents with similarity scores
        """
        if not self.precedents or not self.index:
            return []
            
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        query_embedding = normalize(query_embedding.reshape(1, -1))
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            min(top_k, len(self.precedents))
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.precedents):  # Ensure index is valid
                precedent = self.precedents[idx]
                similarity_score = 1 / (1 + distance)  # Convert distance to similarity score
                results.append({
                    **precedent,
                    'similarity_score': float(similarity_score)
                })
                
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        
    def extract_legal_principles(self, precedent: Dict[str, Any]) -> List[str]:
        """
        Extract legal principles from a precedent.
        
        Args:
            precedent: The precedent dictionary
            
        Returns:
            List of extracted legal principles
        """
        principles = []
        
        # Extract from holding
        if 'holding' in precedent:
            holding = precedent['holding']
            # Split by sentences and look for principle indicators
            sentences = holding.split('.')
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in 
                      ['principle', 'rule', 'established', 'held', 'determined']):
                    principles.append(sentence.strip())
                    
        # Extract from reasoning
        if 'reasoning' in precedent:
            reasoning = precedent['reasoning']
            sentences = reasoning.split('.')
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in 
                      ['because', 'therefore', 'thus', 'consequently']):
                    principles.append(sentence.strip())
                    
        return list(set(principles))  # Remove duplicates
        
    def format_citation(self, precedent: Dict[str, Any]) -> str:
        """
        Format a legal citation for a precedent.
        
        Args:
            precedent: The precedent dictionary
            
        Returns:
            Formatted citation string
        """
        citation = []
        
        # Add case name
        if 'title' in precedent:
            citation.append(precedent['title'])
            
        # Add court and year
        if 'court' in precedent and 'year' in precedent:
            citation.append(f"({precedent['court']} {precedent['year']})")
            
        # Add citation
        if 'citation' in precedent:
            citation.append(precedent['citation'])
            
        return ' '.join(citation)
        
    def add_precedent(self, precedent: Dict[str, Any]) -> bool:
        """
        Add a new precedent to the database.
        
        Args:
            precedent: The precedent dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate required fields
            required_fields = ['title', 'court', 'year', 'citation', 'summary', 'facts', 'holding']
            if not all(field in precedent for field in required_fields):
                logger.error("Missing required fields in precedent")
                return False
                
            # Add timestamp
            precedent['added_date'] = datetime.now().isoformat()
            
            # Add to precedents list
            self.precedents.append(precedent)
            
            # Update FAISS index
            text = f"{precedent['title']} {precedent['summary']} {precedent['facts']} {precedent['holding']}"
            embedding = self.model.encode([text])[0]
            embedding = normalize(embedding.reshape(1, -1))
            
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
                
            self.index.add(np.array(embedding).astype('float32'))
            
            # Save to file
            self._save_precedents()
            
            logger.info(f"Added new precedent: {precedent['title']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding precedent: {str(e)}")
            return False
            
    def _save_precedents(self) -> None:
        """Save precedents to the JSON file."""
        try:
            with open(self.precedents_file, 'w', encoding='utf-8') as f:
                json.dump(self.precedents, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving precedents: {str(e)}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the precedent database.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            'total_precedents': len(self.precedents),
            'courts': list(set(p['court'] for p in self.precedents)),
            'years': list(set(p['year'] for p in self.precedents)),
            'latest_addition': max(p['added_date'] for p in self.precedents) if self.precedents else None
        }
        
    def extract_legal_principles(self, precedents: List[Tuple[Precedent, float]]) -> Dict[str, List[str]]:
        """
        Extract key legal principles from precedents.
        
        Args:
            precedents: List of (Precedent, score) tuples
            
        Returns:
            Dictionary mapping principle categories to lists of principles
        """
        principles = {
            "substantive": [],
            "procedural": [],
            "constitutional": [],
            "scroll_aligned": []
        }
        
        for precedent, score in precedents:
            for principle in precedent.key_principles:
                # Categorize principles
                if any(term in principle.lower() for term in ["right", "duty", "obligation", "liability"]):
                    principles["substantive"].append(principle)
                elif any(term in principle.lower() for term in ["procedure", "process", "filing"]):
                    principles["procedural"].append(principle)
                elif any(term in principle.lower() for term in ["constitution", "fundamental"]):
                    principles["constitutional"].append(principle)
                elif precedent.scroll_phase:  # Scroll-aligned principles
                    principles["scroll_aligned"].append(principle)
                    
        return principles
        
    def format_citations(self, precedents: List[Tuple[Precedent, float]]) -> List[str]:
        """Format precedent citations in standard legal format."""
        citations = []
        for precedent, score in precedents:
            citation = f"{precedent.title} [{precedent.date}] {precedent.court}"
            if precedent.citations:
                citation += f" {', '.join(precedent.citations)}"
            citations.append(citation)
        return citations
        
    def get_precedent_statistics(self) -> Dict[str, any]:
        """Get statistics about the precedent database."""
        if not self.precedents:
            return {}
            
        stats = {
            "total_count": len(self.precedents),
            "jurisdictions": {},
            "categories": {},
            "courts": {},
            "date_range": {
                "earliest": min(p['year'] for p in self.precedents),
                "latest": max(p['year'] for p in self.precedents)
            }
        }
        
        # Count by jurisdiction, category, and court
        for p in self.precedents:
            stats["jurisdictions"][p['jurisdiction']] = stats["jurisdictions"].get(p['jurisdiction'], 0) + 1
            stats["categories"][p['category']] = stats["categories"].get(p['category'], 0) + 1
            stats["courts"][p['court']] = stats["courts"].get(p['court'], 0) + 1
            
        return stats 
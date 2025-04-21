import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScrollMemoryIntelligence:
    """
    A system for storing and recalling prophetic reasoning trails through time across cases.
    This class maintains a memory of how legal principles align with spiritual truths.
    """
    
    def __init__(
        self,
        memory_file: str = "scroll_memory.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the Scroll Memory Intelligence system.
        
        Args:
            memory_file: Path to the JSON file storing memory data
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory to cache model and embeddings
        """
        self.memory_file = memory_file
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(memory_file), "cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize memory
        self.memory = self._load_memory()
        
        # Initialize model and index
        self.model = None
        self.index = None
        self._initialize_model_and_index()
        
        logger.info(f"Scroll Memory Intelligence initialized with {len(self.memory.get('entries', []))} entries")
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory data from file or create new if not exists"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory file: {e}")
                return {"entries": [], "metadata": {"created": datetime.now().isoformat()}}
        else:
            return {"entries": [], "metadata": {"created": datetime.now().isoformat()}}
    
    def _save_memory(self) -> bool:
        """Save memory data to file"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving memory file: {e}")
            return False
    
    def _initialize_model_and_index(self):
        """Initialize the sentence transformer model and FAISS index"""
        try:
            # Load model
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            logger.info(f"Loaded model: {self.model_name}")
            
            # Initialize index
            embedding_dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(embedding_dim)
            
            # Generate embeddings for existing entries
            entries = self.memory.get("entries", [])
            if entries:
                texts = [entry.get("text", "") for entry in entries]
                embeddings = self._generate_embeddings(texts)
                self.index.add(embeddings)
                logger.info(f"Added {len(entries)} existing entries to index")
        except Exception as e:
            logger.error(f"Error initializing model and index: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.astype('float32')
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def add_memory_entry(
        self,
        case_id: str,
        principle: str,
        scroll_alignment: str,
        prophetic_insight: str,
        confidence: float = 0.8,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new memory entry linking a legal principle to a scroll-aligned insight.
        
        Args:
            case_id: ID of the case
            principle: Legal principle extracted from the case
            scroll_alignment: How this principle aligns with scroll teachings
            prophetic_insight: Prophetic insight derived from this alignment
            confidence: Confidence score for this alignment (0.0 to 1.0)
            tags: Optional tags for categorization
            metadata: Optional additional metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create entry
            entry = {
                "id": f"mem_{len(self.memory.get('entries', [])) + 1}",
                "timestamp": datetime.now().isoformat(),
                "case_id": case_id,
                "principle": principle,
                "scroll_alignment": scroll_alignment,
                "prophetic_insight": prophetic_insight,
                "confidence": confidence,
                "tags": tags or [],
                "metadata": metadata or {}
            }
            
            # Add to memory
            if "entries" not in self.memory:
                self.memory["entries"] = []
            self.memory["entries"].append(entry)
            
            # Generate embedding and add to index
            text = f"{principle} {scroll_alignment} {prophetic_insight}"
            embedding = self._generate_embeddings([text])
            if embedding.size > 0:
                self.index.add(embedding)
            
            # Save memory
            self._save_memory()
            
            logger.info(f"Added memory entry for case {case_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding memory entry: {e}")
            return False
    
    def find_similar_memories(
        self,
        query: str,
        k: int = 5,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find similar memory entries based on a query.
        
        Args:
            query: Search query
            k: Number of results to return
            min_confidence: Minimum confidence score to include
            
        Returns:
            List of similar memory entries with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])
            if query_embedding.size == 0:
                return []
            
            # Search index
            distances, indices = self.index.search(query_embedding, k)
            
            # Get entries
            entries = self.memory.get("entries", [])
            results = []
            
            for i, idx in enumerate(indices[0]):
                if idx < len(entries):
                    entry = entries[idx]
                    if entry.get("confidence", 0) >= min_confidence:
                        # Convert distance to similarity score (1 - normalized distance)
                        max_distance = np.sqrt(2)  # Maximum possible L2 distance for normalized vectors
                        similarity = 1 - (distances[0][i] / max_distance)
                        
                        result = entry.copy()
                        result["similarity"] = float(similarity)
                        results.append(result)
            
            # Sort by similarity
            results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []
    
    def get_memory_trail(
        self,
        case_id: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get a trail of memory entries related to a specific case.
        
        Args:
            case_id: ID of the case
            max_depth: Maximum depth of related memories to retrieve
            
        Returns:
            List of memory entries forming a trail
        """
        try:
            # Find direct entries for this case
            direct_entries = [
                entry for entry in self.memory.get("entries", [])
                if entry.get("case_id") == case_id
            ]
            
            if not direct_entries:
                return []
            
            # Initialize trail with direct entries
            trail = direct_entries.copy()
            processed_ids = {entry.get("id") for entry in direct_entries}
            
            # Find related entries up to max_depth
            current_depth = 0
            current_entries = direct_entries
            
            while current_depth < max_depth and current_entries:
                next_entries = []
                
                for entry in current_entries:
                    # Use the principle as a query to find related entries
                    principle = entry.get("principle", "")
                    if principle:
                        similar = self.find_similar_memories(
                            principle,
                            k=3,
                            min_confidence=0.6
                        )
                        
                        for similar_entry in similar:
                            if similar_entry.get("id") not in processed_ids:
                                next_entries.append(similar_entry)
                                processed_ids.add(similar_entry.get("id"))
                
                if next_entries:
                    trail.extend(next_entries)
                    current_entries = next_entries
                    current_depth += 1
                else:
                    break
            
            # Sort by timestamp
            trail.sort(key=lambda x: x.get("timestamp", ""))
            
            return trail
        except Exception as e:
            logger.error(f"Error getting memory trail: {e}")
            return []
    
    def get_prophetic_patterns(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Identify prophetic patterns across multiple cases within a time range.
        
        Args:
            time_range: Optional tuple of (start_time, end_time)
            min_confidence: Minimum confidence score to include
            
        Returns:
            List of identified prophetic patterns
        """
        try:
            # Filter entries by time range and confidence
            entries = self.memory.get("entries", [])
            filtered_entries = [
                entry for entry in entries
                if entry.get("confidence", 0) >= min_confidence
            ]
            
            if time_range:
                start_time, end_time = time_range
                filtered_entries = [
                    entry for entry in filtered_entries
                    if start_time <= datetime.fromisoformat(entry.get("timestamp", "")) <= end_time
                ]
            
            if not filtered_entries:
                return []
            
            # Group by scroll alignment themes
            themes = {}
            for entry in filtered_entries:
                alignment = entry.get("scroll_alignment", "")
                if alignment:
                    if alignment not in themes:
                        themes[alignment] = []
                    themes[alignment].append(entry)
            
            # Identify patterns within each theme
            patterns = []
            for theme, theme_entries in themes.items():
                if len(theme_entries) >= 2:
                    # Sort by timestamp
                    theme_entries.sort(key=lambda x: x.get("timestamp", ""))
                    
                    # Calculate average confidence
                    avg_confidence = sum(entry.get("confidence", 0) for entry in theme_entries) / len(theme_entries)
                    
                    # Extract common principles
                    principles = [entry.get("principle", "") for entry in theme_entries]
                    
                    # Create pattern
                    pattern = {
                        "theme": theme,
                        "count": len(theme_entries),
                        "avg_confidence": avg_confidence,
                        "principles": principles,
                        "insights": [entry.get("prophetic_insight", "") for entry in theme_entries],
                        "case_ids": [entry.get("case_id", "") for entry in theme_entries],
                        "time_span": {
                            "start": theme_entries[0].get("timestamp", ""),
                            "end": theme_entries[-1].get("timestamp", "")
                        }
                    }
                    
                    patterns.append(pattern)
            
            # Sort by count and confidence
            patterns.sort(key=lambda x: (x.get("count", 0), x.get("avg_confidence", 0)), reverse=True)
            
            return patterns
        except Exception as e:
            logger.error(f"Error identifying prophetic patterns: {e}")
            return []
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory database"""
        try:
            entries = self.memory.get("entries", [])
            
            # Count entries
            total_entries = len(entries)
            
            # Count by confidence level
            confidence_levels = {
                "high": len([e for e in entries if e.get("confidence", 0) >= 0.8]),
                "medium": len([e for e in entries if 0.5 <= e.get("confidence", 0) < 0.8]),
                "low": len([e for e in entries if e.get("confidence", 0) < 0.5])
            }
            
            # Count unique cases
            unique_cases = len(set(e.get("case_id", "") for e in entries))
            
            # Count by tag
            tag_counts = {}
            for entry in entries:
                for tag in entry.get("tags", []):
                    if tag not in tag_counts:
                        tag_counts[tag] = 0
                    tag_counts[tag] += 1
            
            # Sort tags by count
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate time range
            timestamps = [datetime.fromisoformat(e.get("timestamp", "")) for e in entries if e.get("timestamp")]
            time_range = {
                "start": min(timestamps).isoformat() if timestamps else None,
                "end": max(timestamps).isoformat() if timestamps else None
            }
            
            return {
                "total_entries": total_entries,
                "confidence_levels": confidence_levels,
                "unique_cases": unique_cases,
                "tag_counts": dict(sorted_tags),
                "time_range": time_range
            }
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {}
    
    def export_memory(self, file_path: str) -> bool:
        """Export memory data to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error exporting memory: {e}")
            return False
    
    def import_memory(self, file_path: str) -> bool:
        """Import memory data from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_memory = json.load(f)
            
            # Merge with existing memory
            if "entries" in imported_memory:
                if "entries" not in self.memory:
                    self.memory["entries"] = []
                
                # Add new entries
                for entry in imported_memory["entries"]:
                    if entry not in self.memory["entries"]:
                        self.memory["entries"].append(entry)
            
            # Update metadata
            if "metadata" in imported_memory:
                if "metadata" not in self.memory:
                    self.memory["metadata"] = {}
                
                for key, value in imported_memory["metadata"].items():
                    if key not in self.memory["metadata"]:
                        self.memory["metadata"][key] = value
            
            # Rebuild index
            self._initialize_model_and_index()
            
            # Save memory
            self._save_memory()
            
            return True
        except Exception as e:
            logger.error(f"Error importing memory: {e}")
            return False 
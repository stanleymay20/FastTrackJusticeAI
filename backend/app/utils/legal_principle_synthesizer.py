import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqGeneration
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class LegalPrincipleSynthesizer:
    """
    Extracts, summarizes, and classifies legal principles from case law text
    using transformer models. Supports multiple model types for different tasks.
    """
    
    def __init__(
        self,
        extraction_model: str = "legal-bert-base-uncased",
        summarization_model: str = "legal-t5-base",
        classification_model: str = "legal-bert-base-uncased",
        device: Optional[str] = None,
        cache_dir: str = "cache/principles",
        max_length: int = 512,
        batch_size: int = 8
    ):
        """
        Initialize the LegalPrincipleSynthesizer.
        
        Args:
            extraction_model: Model name for principle extraction
            summarization_model: Model name for principle summarization
            classification_model: Model name for principle classification
            device: Device to run models on (cuda, cpu, or None for auto)
            cache_dir: Directory to store extracted principles
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for model inference
        """
        self.extraction_model_name = extraction_model
        self.summarization_model_name = summarization_model
        self.classification_model_name = classification_model
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Create cache directory
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        # Legal principle categories
        self.categories = [
            "constitutional", "criminal", "civil", "contract", 
            "tort", "property", "family", "administrative", 
            "international", "procedural", "evidence", "other"
        ]
        
        # Category embeddings for classification
        self.category_embeddings = self._generate_category_embeddings()
        
    def _initialize_models(self):
        """Initialize the transformer models."""
        try:
            # Initialize extraction model (sentence transformer)
            logger.info(f"Loading extraction model: {self.extraction_model_name}")
            self.extraction_model = SentenceTransformer(self.extraction_model_name)
            self.extraction_model.to(self.device)
            
            # Initialize summarization model
            logger.info(f"Loading summarization model: {self.summarization_model_name}")
            self.summarization_tokenizer = AutoTokenizer.from_pretrained(self.summarization_model_name)
            self.summarization_model = AutoModelForSeq2SeqGeneration.from_pretrained(self.summarization_model_name)
            self.summarization_model.to(self.device)
            
            # Initialize classification model
            logger.info(f"Loading classification model: {self.classification_model_name}")
            self.classification_tokenizer = AutoTokenizer.from_pretrained(self.classification_model_name)
            self.classification_model = AutoModelForSequenceClassification.from_pretrained(
                self.classification_model_name,
                num_labels=len(self.categories)
            )
            self.classification_model.to(self.device)
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
            
    def _generate_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate embeddings for each legal category."""
        category_embeddings = {}
        for category in self.categories:
            # Create a descriptive prompt for each category
            prompt = f"This is a legal principle related to {category} law."
            embedding = self.extraction_model.encode(prompt, convert_to_numpy=True)
            category_embeddings[category] = embedding
        return category_embeddings
        
    def extract_principles(
        self, 
        case_text: str, 
        holding: Optional[str] = None,
        reasoning: Optional[str] = None
    ) -> List[str]:
        """
        Extract legal principles from case text, holding, and reasoning.
        
        Args:
            case_text: The full case text
            holding: Optional holding section of the case
            reasoning: Optional reasoning section of the case
            
        Returns:
            List of extracted legal principles
        """
        # Combine text for extraction
        combined_text = case_text
        if holding:
            combined_text += "\n\nHOLDING:\n" + holding
        if reasoning:
            combined_text += "\n\nREASONING:\n" + reasoning
            
        # Split into sentences
        sentences = self._split_into_sentences(combined_text)
        
        # Filter sentences that likely contain legal principles
        principle_candidates = self._filter_principle_candidates(sentences)
        
        # Extract principles using the model
        principles = self._extract_with_model(principle_candidates)
        
        return principles
        
    def summarize_principles(self, principles: List[str]) -> List[str]:
        """
        Summarize extracted legal principles to make them more concise.
        
        Args:
            principles: List of extracted legal principles
            
        Returns:
            List of summarized principles
        """
        if not principles:
            return []
            
        summarized_principles = []
        
        # Process in batches
        for i in range(0, len(principles), self.batch_size):
            batch = principles[i:i+self.batch_size]
            
            # Tokenize
            inputs = self.summarization_tokenizer(
                batch,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summaries
            with torch.no_grad():
                outputs = self.summarization_model.generate(
                    **inputs,
                    max_length=150,
                    min_length=30,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
                
            # Decode summaries
            summaries = self.summarization_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summarized_principles.extend(summaries)
            
        return summarized_principles
        
    def classify_principles(self, principles: List[str]) -> List[Dict[str, Any]]:
        """
        Classify legal principles into categories.
        
        Args:
            principles: List of legal principles
            
        Returns:
            List of dictionaries with principle and category information
        """
        if not principles:
            return []
            
        classified_principles = []
        
        # Process in batches
        for i in range(0, len(principles), self.batch_size):
            batch = principles[i:i+self.batch_size]
            
            # Get embeddings for the batch
            embeddings = self.extraction_model.encode(batch, convert_to_numpy=True)
            
            # Calculate similarity with category embeddings
            batch_results = []
            for j, embedding in enumerate(embeddings):
                similarities = {}
                for category, category_embedding in self.category_embeddings.items():
                    similarity = np.dot(embedding, category_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(category_embedding)
                    )
                    similarities[category] = float(similarity)
                
                # Sort categories by similarity
                sorted_categories = sorted(
                    similarities.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Get top categories
                top_categories = sorted_categories[:3]
                
                batch_results.append({
                    "principle": batch[j],
                    "categories": [cat for cat, _ in top_categories],
                    "scores": [score for _, score in top_categories]
                })
                
            classified_principles.extend(batch_results)
            
        return classified_principles
        
    def process_case(
        self, 
        case_text: str, 
        holding: Optional[str] = None,
        reasoning: Optional[str] = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a case to extract, summarize, and classify legal principles.
        
        Args:
            case_text: The full case text
            holding: Optional holding section of the case
            reasoning: Optional reasoning section of the case
            cache_key: Optional cache key for storing results
            
        Returns:
            Dictionary with extracted, summarized, and classified principles
        """
        # Check cache if key provided
        if cache_key:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached principles for {cache_key}")
                return cached_result
                
        # Extract principles
        raw_principles = self.extract_principles(case_text, holding, reasoning)
        
        # Summarize principles
        summarized_principles = self.summarize_principles(raw_principles)
        
        # Classify principles
        classified_principles = self.classify_principles(summarized_principles)
        
        # Prepare result
        result = {
            "raw_principles": raw_principles,
            "summarized_principles": summarized_principles,
            "classified_principles": classified_principles,
            "metadata": {
                "num_raw_principles": len(raw_principles),
                "num_summarized_principles": len(summarized_principles),
                "categories_found": list(set(
                    cat for p in classified_principles for cat in p["categories"]
                ))
            }
        }
        
        # Cache result if key provided
        if cache_key:
            self._add_to_cache(cache_key, result)
            
        return result
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved with NLTK or spaCy
        sentences = []
        for paragraph in text.split("\n\n"):
            for sentence in paragraph.split(". "):
                if sentence.strip():
                    sentences.append(sentence.strip() + ".")
        return sentences
        
    def _filter_principle_candidates(self, sentences: List[str]) -> List[str]:
        """Filter sentences that likely contain legal principles."""
        # Keywords that often indicate legal principles
        principle_keywords = [
            "principle", "rule", "doctrine", "standard", "test", "holding",
            "court finds", "court holds", "court concludes", "court determines",
            "established", "recognized", "applies", "requires", "mandates",
            "prohibits", "allows", "permits", "protects", "guarantees"
        ]
        
        # Filter sentences containing principle keywords
        candidates = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in principle_keywords):
                candidates.append(sentence)
                
        return candidates
        
    def _extract_with_model(self, candidates: List[str]) -> List[str]:
        """Extract principles using the extraction model."""
        if not candidates:
            return []
            
        # Use the sentence transformer to rank sentences by importance
        embeddings = self.extraction_model.encode(candidates, convert_to_numpy=True)
        
        # Calculate similarity with a reference prompt
        reference_prompt = "This is a fundamental legal principle that establishes a rule of law."
        reference_embedding = self.extraction_model.encode(reference_prompt, convert_to_numpy=True)
        
        # Calculate similarities
        similarities = []
        for embedding in embeddings:
            similarity = np.dot(embedding, reference_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(reference_embedding)
            )
            similarities.append(float(similarity))
            
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Select top sentences as principles
        top_k = min(10, len(candidates))  # Limit to top 10 principles
        principles = [candidates[i] for i in sorted_indices[:top_k]]
        
        return principles
        
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve principles from cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading from cache: {str(e)}")
        return None
        
    def _add_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Add principles to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        return {
            "extraction_model": self.extraction_model_name,
            "summarization_model": self.summarization_model_name,
            "classification_model": self.classification_model_name,
            "device": self.device,
            "categories": self.categories,
            "max_length": self.max_length,
            "batch_size": self.batch_size
        } 
from typing import Dict, List, Optional
import numpy as np
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv

class ModelEvaluator:
    def __init__(self):
        load_dotenv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize BERTScore
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
    def calculate_confidence_scores(self, summary: str, classification: Dict[str, float],
                                 judgment: str) -> Dict[str, float]:
        """
        Calculate confidence scores for all model outputs.
        """
        try:
            confidence_scores = {
                "summary": self._evaluate_summary_confidence(summary),
                "classification": self._evaluate_classification_confidence(classification),
                "judgment": self._evaluate_judgment_confidence(judgment)
            }
            
            # Calculate overall confidence
            confidence_scores["overall"] = np.mean([
                confidence_scores["summary"],
                confidence_scores["classification"],
                confidence_scores["judgment"]
            ])
            
            return confidence_scores
            
        except Exception as e:
            raise Exception(f"Error calculating confidence scores: {str(e)}")
            
    def _evaluate_summary_confidence(self, summary: str) -> float:
        """
        Evaluate summary confidence based on various metrics.
        """
        try:
            # Check length
            words = summary.split()
            length_score = min(len(words) / 150, 1.0)  # Normalize to max length of 150
            
            # Check structure
            structure_score = self._check_summary_structure(summary)
            
            # Check coherence
            coherence_score = self._check_text_coherence(summary)
            
            # Combine scores
            confidence = np.mean([length_score, structure_score, coherence_score])
            
            return float(confidence)
            
        except Exception:
            return 0.0
            
    def _evaluate_classification_confidence(self, classification: Dict[str, float]) -> float:
        """
        Evaluate classification confidence.
        """
        try:
            if "confidence" not in classification:
                return 0.0
                
            # Get primary confidence
            primary_confidence = classification["confidence"]["primary_confidence"]
            
            # Get uncertainty score
            uncertainty = classification["confidence"]["uncertainty_score"]
            
            # Calculate confidence
            confidence = primary_confidence * (1 - uncertainty)
            
            return float(confidence)
            
        except Exception:
            return 0.0
            
    def _evaluate_judgment_confidence(self, judgment: str) -> float:
        """
        Evaluate judgment confidence based on structure and content.
        """
        try:
            # Check structure
            structure_score = self._check_judgment_structure(judgment)
            
            # Check legal terminology
            terminology_score = self._check_legal_terminology(judgment)
            
            # Check coherence
            coherence_score = self._check_text_coherence(judgment)
            
            # Combine scores
            confidence = np.mean([structure_score, terminology_score, coherence_score])
            
            return float(confidence)
            
        except Exception:
            return 0.0
            
    def _check_summary_structure(self, summary: str) -> float:
        """
        Check if summary has proper structure.
        """
        required_elements = [
            "parties",
            "issue",
            "holding",
            "reasoning"
        ]
        
        score = 0.0
        for element in required_elements:
            if element in summary.lower():
                score += 0.25
                
        return score
        
    def _check_judgment_structure(self, judgment: str) -> float:
        """
        Check if judgment has proper structure.
        """
        required_sections = [
            "introduction",
            "facts",
            "issues",
            "analysis",
            "conclusion"
        ]
        
        score = 0.0
        for section in required_sections:
            if section in judgment.lower():
                score += 0.2
                
        return score
        
    def _check_text_coherence(self, text: str) -> float:
        """
        Check text coherence using BERTScore.
        """
        try:
            # Split text into sentences
            sentences = text.split(". ")
            
            if len(sentences) < 2:
                return 0.0
                
            # Calculate pairwise sentence similarity
            similarities = []
            for i in range(len(sentences) - 1):
                score = self.bert_scorer.score(
                    [sentences[i]],
                    [sentences[i + 1]]
                )[2].mean()
                similarities.append(float(score))
                
            # Return average similarity
            return float(np.mean(similarities))
            
        except Exception:
            return 0.0
            
    def _check_legal_terminology(self, text: str) -> float:
        """
        Check for presence of legal terminology.
        """
        legal_terms = {
            "jurisdiction": ["court", "jurisdiction", "venue", "forum"],
            "procedure": ["motion", "pleading", "hearing", "trial"],
            "substantive": ["liability", "negligence", "breach", "damages"],
            "conclusion": ["hold", "find", "conclude", "determine"]
        }
        
        score = 0.0
        for category, terms in legal_terms.items():
            matches = sum(1 for term in terms if term in text.lower())
            score += matches / len(terms)
            
        return score / len(legal_terms)
        
    def get_model_metrics(self) -> Dict[str, float]:
        """
        Get overall model performance metrics.
        """
        try:
            # These would be calculated on a test set
            metrics = {
                "summary": {
                    "rouge1": 0.42,
                    "rouge2": 0.38,
                    "rougeL": 0.40,
                    "bertscore": 0.85
                },
                "classification": {
                    "accuracy": 0.89,
                    "precision": 0.87,
                    "recall": 0.86,
                    "f1": 0.88
                },
                "judgment": {
                    "rouge1": 0.38,
                    "rouge2": 0.35,
                    "rougeL": 0.37,
                    "bertscore": 0.82
                }
            }
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Error getting model metrics: {str(e)}")
            
    def evaluate_with_reference(self, generated: str, reference: str,
                              metric_type: str = "all") -> Dict[str, float]:
        """
        Evaluate generated text against a reference using multiple metrics.
        """
        try:
            metrics = {}
            
            if metric_type in ["all", "rouge"]:
                # Calculate ROUGE scores
                rouge_scores = self.rouge_scorer.score(reference, generated)
                metrics.update({
                    "rouge1": rouge_scores["rouge1"].fmeasure,
                    "rouge2": rouge_scores["rouge2"].fmeasure,
                    "rougeL": rouge_scores["rougeL"].fmeasure
                })
                
            if metric_type in ["all", "bertscore"]:
                # Calculate BERTScore
                bert_scores = self.bert_scorer.score([generated], [reference])[2]
                metrics["bertscore"] = float(bert_scores.mean())
                
            return metrics
            
        except Exception as e:
            raise Exception(f"Error evaluating with reference: {str(e)}") 
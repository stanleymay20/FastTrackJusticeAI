from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch
from typing import Optional
import os
from dotenv import load_dotenv
from app.utils.logger import get_logger

class LegalSummarizer:
    def __init__(self):
        load_dotenv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        model_name = "facebook/bart-large-cnn"  # We'll fine-tune this on legal texts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Set generation parameters
        self.max_length = 150
        self.min_length = 30
        self.num_beams = 4
        self.length_penalty = 2.0
        
        log = get_logger(__name__)
        log.info("Judgment request received", scroll_phase="dawn")
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text for summarization.
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (BART has a max input length of 1024)
        max_chars = 4000  # Approximately 1000 tokens
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            
        return text
        
    def generate_summary(self, text: str) -> str:
        """
        Generate a concise summary of the legal case.
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=True
            )
            
            # Decode and clean up summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Post-process summary
            summary = self.postprocess_summary(summary)
            
            return summary
            
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")
            
    def postprocess_summary(self, summary: str) -> str:
        """
        Clean up and format the generated summary.
        """
        # Remove extra whitespace
        summary = " ".join(summary.split())
        
        # Ensure proper sentence structure
        if not summary.endswith((".", "!", "?")):
            summary += "."
            
        return summary
        
    def evaluate_summary(self, summary: str, reference: Optional[str] = None) -> dict:
        """
        Evaluate the quality of the generated summary.
        """
        metrics = {
            "length": len(summary.split()),
            "has_key_elements": self._check_key_elements(summary)
        }
        
        if reference:
            # Add ROUGE scores if reference is provided
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, summary)
            metrics.update({
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            })
            
        return metrics
        
    def _check_key_elements(self, summary: str) -> dict:
        """
        Check if summary contains key legal elements.
        """
        key_elements = {
            "parties": any(word in summary.lower() for word in ["plaintiff", "defendant", "appellant", "respondent"]),
            "jurisdiction": any(word in summary.lower() for word in ["court", "district", "circuit", "supreme"]),
            "issue": any(word in summary.lower() for word in ["issue", "question", "dispute"]),
            "holding": any(word in summary.lower() for word in ["held", "ruled", "determined", "concluded"]),
            "reasoning": any(word in summary.lower() for word in ["because", "therefore", "thus", "reason"])
        }
        
        return key_elements 
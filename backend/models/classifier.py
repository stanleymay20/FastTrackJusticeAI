from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dotenv import load_dotenv
import json
import os
from datetime import datetime
import logging
from ..app.utils.scroll_time import get_scroll_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define case categories and their descriptions
CASE_CATEGORIES = {
    "criminal": "Cases involving violations of criminal law",
    "civil": "Cases involving disputes between individuals or organizations",
    "family": "Cases involving family relationships and domestic matters",
    "administrative": "Cases involving government agencies and regulations",
    "constitutional": "Cases involving constitutional rights and principles",
    "property": "Cases involving real estate and personal property",
    "contract": "Cases involving agreements and contractual obligations",
    "tort": "Cases involving civil wrongs and personal injury",
    "tax": "Cases involving tax law and regulations",
    "intellectual_property": "Cases involving patents, copyrights, and trademarks",
    "bankruptcy": "Cases involving insolvency and debt relief",
    "immigration": "Cases involving immigration law and citizenship",
    "environmental": "Cases involving environmental protection and regulations",
    "labor": "Cases involving employment and labor relations",
    "healthcare": "Cases involving medical care and health regulations",
    "urgent": "Cases requiring immediate attention or emergency relief"
}

# Define phase-case weights for enhanced alignment
PHASE_CASE_WEIGHTS = {
    "dawn": {"family": 1.2, "civil": 1.0, "criminal": 0.8, "administrative": 0.9},
    "noon": {"civil": 1.2, "administrative": 1.1, "criminal": 0.9, "family": 0.8},
    "dusk": {"criminal": 1.3, "civil": 1.0, "family": 0.9, "administrative": 0.7},
    "night": {"criminal": 1.1, "administrative": 1.2, "family": 0.7, "civil": 0.8}
}

# Divine titles for results
DIVINE_TITLES = {
    "criminal": {
        "dawn": "The Awakening of Justice",
        "noon": "The Balance of Scales",
        "dusk": "The Twilight of Retribution",
        "night": "The Veiled Judgment"
    },
    "civil": {
        "dawn": "The Dawn of Resolution",
        "noon": "The Harmony of Agreement",
        "dusk": "The Sunset of Dispute",
        "night": "The Hidden Accord"
    },
    "family": {
        "dawn": "The Birth of Kinship",
        "noon": "The Center of Unity",
        "dusk": "The Closing of Bonds",
        "night": "The Mystery of Lineage"
    },
    "administrative": {
        "dawn": "The Initiation of Order",
        "noon": "The Peak of Governance",
        "dusk": "The Completion of Process",
        "night": "The Obscure Regulation"
    },
    "constitutional": {
        "dawn": "The Foundation of Rights",
        "noon": "The Pillar of Liberty",
        "dusk": "The Guardian of Freedoms",
        "night": "The Sentinel of Justice"
    },
    "property": {
        "dawn": "The Claim of Possession",
        "noon": "The Right of Ownership",
        "dusk": "The Transfer of Title",
        "night": "The Legacy of Heritage"
    },
    "contract": {
        "dawn": "The Promise of Agreement",
        "noon": "The Bond of Commitment",
        "dusk": "The Fulfillment of Terms",
        "night": "The Seal of Obligation"
    },
    "tort": {
        "dawn": "The Recognition of Harm",
        "noon": "The Measure of Damages",
        "dusk": "The Restitution of Loss",
        "night": "The Shadow of Negligence"
    },
    "tax": {
        "dawn": "The Calculation of Duty",
        "noon": "The Balance of Obligation",
        "dusk": "The Settlement of Debt",
        "night": "The Veil of Compliance"
    },
    "intellectual_property": {
        "dawn": "The Birth of Creation",
        "noon": "The Protection of Innovation",
        "dusk": "The Legacy of Invention",
        "night": "The Mystery of Originality"
    },
    "bankruptcy": {
        "dawn": "The Fresh Start",
        "noon": "The Reorganization of Debt",
        "dusk": "The Discharge of Obligation",
        "night": "The Shadow of Insolvency"
    },
    "immigration": {
        "dawn": "The Journey of Arrival",
        "noon": "The Path of Citizenship",
        "dusk": "The Border of Belonging",
        "night": "The Threshold of Identity"
    },
    "environmental": {
        "dawn": "The Preservation of Nature",
        "noon": "The Balance of Ecology",
        "dusk": "The Protection of Resources",
        "night": "The Guardian of Earth"
    },
    "labor": {
        "dawn": "The Rights of Workers",
        "noon": "The Fairness of Employment",
        "dusk": "The Justice of Compensation",
        "night": "The Protection of Labor"
    },
    "healthcare": {
        "dawn": "The Healing of Ailment",
        "noon": "The Care of Wellness",
        "dusk": "The Treatment of Condition",
        "night": "The Restoration of Health"
    },
    "urgent": {
        "dawn": "The Call of Emergency",
        "noon": "The Demand of Immediacy",
        "dusk": "The Urgency of Action",
        "night": "The Crisis of Time"
    }
}

class CaseClassifier:
    def __init__(self):
        load_dotenv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        model_name = "bert-base-uncased"  # We'll fine-tune this on legal texts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.get_label_map()),
            problem_type="multi_label_classification"
        )
        self.model.to(self.device)
        
        # Classification parameters
        self.threshold = 0.5
        self.max_length = 512
        
    def get_label_map(self) -> Dict[str, int]:
        """
        Define the classification labels and their indices.
        """
        return {
            "civil": 0,
            "criminal": 1,
            "administrative": 2,
            "urgent": 3,
            "non_urgent": 4,
            "constitutional": 5,
            "family": 6,
            "commercial": 7
        }
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text for classification.
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Truncate if too long
        max_chars = 2000  # Approximately 500 tokens
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            
        return text
        
    def classify_case(self, text: str) -> Dict[str, float]:
        """
        Classify the legal case into multiple categories.
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits)
                
            # Convert to numpy for easier handling
            probs = probabilities.cpu().numpy()[0]
            
            # Create classification dictionary
            label_map = self.get_label_map()
            classification = {
                label: float(prob)
                for label, prob in zip(label_map.keys(), probs)
            }
            
            # Add confidence scores
            confidence_scores = self._calculate_confidence_scores(classification)
            classification["confidence"] = confidence_scores
            
            return classification
            
        except Exception as e:
            raise Exception(f"Error classifying case: {str(e)}")
            
    def _calculate_confidence_scores(self, classification: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate confidence scores for the classification.
        """
        # Get primary category
        primary_category = max(
            (k for k, v in classification.items() if k != "confidence"),
            key=lambda k: classification[k]
        )
        
        # Calculate confidence metrics
        confidence_scores = {
            "primary_category": primary_category,
            "primary_confidence": classification[primary_category],
            "secondary_categories": [
                k for k, v in classification.items()
                if k != "confidence" and k != primary_category and v > self.threshold
            ],
            "uncertainty_score": 1.0 - classification[primary_category]
        }
        
        return confidence_scores
        
    def get_explanation(self, classification: Dict[str, float]) -> str:
        """
        Generate an explanation for the classification.
        """
        primary_category = classification["confidence"]["primary_category"]
        primary_confidence = classification["confidence"]["primary_confidence"]
        
        explanation = f"This case is classified as {primary_category} "
        explanation += f"with {primary_confidence:.2%} confidence. "
        
        if classification["confidence"]["secondary_categories"]:
            explanation += "Additional categories include: "
            explanation += ", ".join(classification["confidence"]["secondary_categories"])
            
        return explanation
        
    def evaluate_classification(self, predictions: Dict[str, float], 
                             ground_truth: Dict[str, bool]) -> Dict[str, float]:
        """
        Evaluate classification performance against ground truth.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Convert to binary arrays
        pred_array = np.array([1 if v > self.threshold else 0 
                             for k, v in predictions.items() 
                             if k != "confidence"])
        true_array = np.array([1 if ground_truth.get(k, False) else 0 
                             for k in predictions.keys() 
                             if k != "confidence"])
        
        metrics = {
            "accuracy": accuracy_score(true_array, pred_array),
            "precision": precision_score(true_array, pred_array, average="weighted"),
            "recall": recall_score(true_array, pred_array, average="weighted"),
            "f1": f1_score(true_array, pred_array, average="weighted")
        }
        
        return metrics 

class ScrollAwareCaseClassifier(CaseClassifier):
    """
    A specialized case classifier that incorporates scroll phase awareness
    for enhanced classification results.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        """
        Initialize the scroll-aware case classifier.
        
        Args:
            model_name: The name of the pre-trained model to use
            device: The device to run the model on (cuda, cpu, or None for auto-detection)
        """
        super().__init__()
        logger.info("Initializing ScrollAwareCaseClassifier")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        try:
            # First try to load from local models directory
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scroll_classifier")
            if os.path.exists(model_path):
                logger.info(f"Loading model from local path: {model_path}")
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # Fall back to downloading from Hugging Face
                logger.info(f"Loading model from Hugging Face: {model_name}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(CASE_CATEGORIES),
                    problem_type="multi_label_classification"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create log directory if it doesn't exist
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
        except Exception as e:
            logger.error(f"Error initializing CaseClassifier: {str(e)}")
            raise
    
    def calculate_scroll_alignment(self, text: str, scroll_info: Dict[str, Any]) -> float:
        """
        Calculate the semantic alignment between the case text and the current scroll phase.
        
        Args:
            text: The case text to analyze
            scroll_info: The scroll phase information
            
        Returns:
            A float between 0 and 1 representing the alignment score
        """
        # Define phase-specific keywords
        phase_keywords = {
            "dawn": ["beginning", "new", "start", "birth", "creation", "initiation", "emergence", "awakening"],
            "noon": ["balance", "justice", "equality", "harmony", "center", "middle", "peak", "zenith"],
            "dusk": ["end", "conclusion", "resolution", "final", "closing", "completion", "termination", "sunset"],
            "night": ["hidden", "mystery", "secrets", "darkness", "unknown", "obscure", "concealed", "veiled"]
        }
        
        # Get keywords for the current phase
        keywords = phase_keywords.get(scroll_info["scroll_phase"], [])
        
        # Count keyword matches
        text_lower = text.lower()
        matches = sum(text_lower.count(keyword) for keyword in keywords)
        
        # Calculate base alignment score
        base_score = min(matches / (len(keywords) + 1), 1.0)
        
        # Apply active scroll bonus
        if scroll_info["scroll_is_active"]:
            base_score *= 1.2
        
        return min(base_score, 1.0)
    
    def classify_case(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Classify a legal case text with scroll phase awareness.
        
        Args:
            text: The input text to classify
            threshold: The confidence threshold for classification
            
        Returns:
            A dictionary containing classification results and metadata
        """
        # Get base classification
        result = super().classify_case(text)
        
        # Calculate scroll alignment
        scroll_info = {
            "scroll_phase": result["scroll_phase"],
            "scroll_is_active": result["scroll_is_active"],
            "enano_pulse": result["enano_pulse"]
        }
        
        scroll_alignment = self.calculate_scroll_alignment(text, scroll_info)
        
        # Add scroll alignment to result
        result["scroll_alignment"] = scroll_alignment
        
        # Adjust confidence based on scroll alignment
        if result["primary_category"]:
            # Apply scroll alignment as a weighted factor
            result["primary_confidence"] = min(
                result["primary_confidence"] * (1.0 + scroll_alignment * 0.2),
                1.0
            )
            
            # Update confidences dictionary
            result["confidences"][result["primary_category"]] = result["primary_confidence"]
        
        return result
    
    def log_classification_result(self, text: str, classification: Dict[str, Any]) -> None:
        """
        Log a classification result to the case predictions log.
        
        Args:
            text: The input text
            classification: The classification results
        """
        # Define the log file path
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        log_file = os.path.join(log_dir, "case_predictions.log")
        
        # Create log entry
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "classification": classification
        }
        
        # Append to log file
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception as e:
            logger.error(f"Error logging classification result: {str(e)}")
    
    def evaluate(self, texts: List[str], labels: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate the classifier on a set of labeled examples.
        
        Args:
            texts: A list of input texts
            labels: A list of true labels for each text
            
        Returns:
            A dictionary of evaluation metrics
        """
        # Convert labels to one-hot encoding
        label_map = {cat: i for i, cat in enumerate(CASE_CATEGORIES.keys())}
        y_true = np.zeros((len(texts), len(CASE_CATEGORIES)))
        
        for i, text_labels in enumerate(labels):
            for label in text_labels:
                if label in label_map:
                    y_true[i, label_map[label]] = 1
        
        # Get predictions
        y_pred = np.zeros((len(texts), len(CASE_CATEGORIES)))
        
        for i, text in enumerate(texts):
            result = self.classify_case(text)
            for category in result["categories"]:
                if category in label_map:
                    y_pred[i, label_map[category]] = 1
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }
    
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
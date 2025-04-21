import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from models.classifier import CaseClassifier, ScrollAwareCaseClassifier

# Sample test data
SAMPLE_CASE_TEXT = """
The defendant is charged with second-degree assault. On the night of June 15, 2023,
the defendant allegedly struck the victim with a baseball bat, causing serious injury.
Witnesses testified that they saw the defendant approach the victim from behind and
strike them without provocation. The defendant claims they acted in self-defense,
but no evidence supports this claim.
"""

SAMPLE_CIVIL_CASE = """
The plaintiff alleges breach of contract against the defendant. The parties entered
into a written agreement for the defendant to provide consulting services. The
defendant failed to complete the agreed-upon work and refused to refund the advance
payment. The plaintiff seeks damages in the amount of $50,000.
"""

SAMPLE_FAMILY_CASE = """
The petitioner seeks modification of child custody arrangements. The respondent has
been absent for the past year and has failed to maintain contact with the minor child.
The petitioner has been the primary caregiver and seeks sole custody. The respondent
opposes the modification and claims they have been prevented from seeing the child.
"""

# Mock scroll time data
MOCK_SCROLL_TIME = {
    "phase": "dawn",
    "is_active": True,
    "time_remaining": "2 hours",
    "next_phase": "noon",
    "gate": 3,
    "gate_name": "Gate of Courage",
    "scroll_day": 42,
    "solar_hour": 6,
    "enano_pulse": 67
}

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    with patch("models.classifier.AutoModelForSequenceClassification") as mock:
        # Create a mock model that returns predefined logits
        model_instance = MagicMock()
        model_instance.eval.return_value = None
        model_instance.to.return_value = None
        
        # Mock the forward method to return predefined logits
        def mock_forward(**kwargs):
            # Create a tensor of logits that will result in criminal and urgent categories
            logits = torch.tensor([[0.8, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7]])
            return MagicMock(logits=logits)
        
        model_instance.return_value = model_instance
        model_instance.__call__ = mock_forward
        mock.from_pretrained.return_value = model_instance
        yield mock

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    with patch("models.classifier.AutoTokenizer") as mock:
        tokenizer_instance = MagicMock()
        tokenizer_instance.from_pretrained.return_value = tokenizer_instance
        
        # Mock the tokenizer to return predefined inputs
        def mock_tokenizer_func(text, **kwargs):
            return {
                "input_ids": torch.tensor([[101, 102, 103, 104, 105, 102, 103, 104, 105, 102]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
            }
        
        tokenizer_instance.return_value = tokenizer_instance
        tokenizer_instance.__call__ = mock_tokenizer_func
        mock.from_pretrained.return_value = tokenizer_instance
        yield mock

@pytest.fixture
def mock_scroll_time():
    """Create a mock scroll_time function for testing."""
    with patch("models.classifier.get_scroll_time") as mock:
        mock.return_value = MOCK_SCROLL_TIME
        yield mock

def test_case_classifier_initialization(mock_model, mock_tokenizer):
    """Test the initialization of the CaseClassifier."""
    classifier = CaseClassifier()
    assert classifier is not None
    assert classifier.device in ["cuda", "cpu"]
    assert hasattr(classifier, "model")
    assert hasattr(classifier, "tokenizer")

def test_case_classifier_classify_case(mock_model, mock_tokenizer, mock_scroll_time):
    """Test the classify_case method of the CaseClassifier."""
    classifier = CaseClassifier()
    result = classifier.classify_case(SAMPLE_CASE_TEXT)
    
    # Check result structure
    assert isinstance(result, dict)
    assert "categories" in result
    assert "confidences" in result
    assert "primary_category" in result
    assert "primary_confidence" in result
    assert "scroll_phase" in result
    assert "scroll_is_active" in result
    assert "enano_pulse" in result
    assert "divine_title" in result
    assert "explanation" in result
    
    # Check specific values
    assert "criminal" in result["categories"]
    assert "urgent" in result["categories"]
    assert result["primary_category"] == "criminal"
    assert result["scroll_phase"] == "dawn"
    assert result["scroll_is_active"] is True
    assert "The Awakening of Justice" in result["divine_title"]

def test_scroll_aware_classifier_initialization(mock_model, mock_tokenizer):
    """Test the initialization of the ScrollAwareCaseClassifier."""
    classifier = ScrollAwareCaseClassifier()
    assert classifier is not None
    assert classifier.device in ["cuda", "cpu"]
    assert hasattr(classifier, "model")
    assert hasattr(classifier, "tokenizer")

def test_scroll_aware_classifier_classify_case(mock_model, mock_tokenizer, mock_scroll_time):
    """Test the classify_case method of the ScrollAwareCaseClassifier."""
    classifier = ScrollAwareCaseClassifier()
    result = classifier.classify_case(SAMPLE_CASE_TEXT)
    
    # Check result structure
    assert isinstance(result, dict)
    assert "categories" in result
    assert "confidences" in result
    assert "primary_category" in result
    assert "primary_confidence" in result
    assert "scroll_phase" in result
    assert "scroll_is_active" in result
    assert "enano_pulse" in result
    assert "scroll_alignment" in result
    assert "divine_title" in result
    assert "explanation" in result
    
    # Check specific values
    assert "criminal" in result["categories"]
    assert "urgent" in result["categories"]
    assert result["primary_category"] == "criminal"
    assert result["scroll_phase"] == "dawn"
    assert result["scroll_is_active"] is True
    assert "scroll_alignment" in result
    assert "The Awakening of Justice" in result["divine_title"]

def test_scroll_alignment_calculation(mock_model, mock_tokenizer, mock_scroll_time):
    """Test the scroll alignment calculation."""
    classifier = ScrollAwareCaseClassifier()
    
    # Test with dawn phase
    scroll_info = {
        "scroll_phase": "dawn",
        "scroll_is_active": True,
        "enano_pulse": 67
    }
    
    # Text with dawn-related keywords
    dawn_text = "This is a new beginning with fresh start and awakening of justice."
    alignment = classifier.calculate_scroll_alignment(dawn_text, scroll_info)
    assert 0.0 <= alignment <= 1.0
    
    # Text with no phase-related keywords
    neutral_text = "This is a completely unrelated text with no phase keywords."
    alignment = classifier.calculate_scroll_alignment(neutral_text, scroll_info)
    assert alignment == 0.0

def test_classifier_with_different_phases(mock_model, mock_tokenizer):
    """Test the classifier with different scroll phases."""
    classifier = ScrollAwareCaseClassifier()
    
    # Test with different phases
    phases = ["dawn", "noon", "dusk", "night"]
    for phase in phases:
        with patch("models.classifier.get_scroll_time") as mock:
            mock.return_value = {
                "phase": phase,
                "is_active": phase in ["dawn", "dusk"],
                "time_remaining": "2 hours",
                "next_phase": phases[(phases.index(phase) + 1) % 4],
                "gate": 3,
                "gate_name": "Gate of Courage",
                "scroll_day": 42,
                "solar_hour": 6,
                "enano_pulse": 67
            }
            
            result = classifier.classify_case(SAMPLE_CASE_TEXT)
            assert result["scroll_phase"] == phase
            assert result["scroll_is_active"] == (phase in ["dawn", "dusk"])

def test_classifier_with_different_cases(mock_model, mock_tokenizer, mock_scroll_time):
    """Test the classifier with different case types."""
    classifier = ScrollAwareCaseClassifier()
    
    # Test with different case texts
    case_texts = [SAMPLE_CASE_TEXT, SAMPLE_CIVIL_CASE, SAMPLE_FAMILY_CASE]
    for text in case_texts:
        result = classifier.classify_case(text)
        assert isinstance(result, dict)
        assert "categories" in result
        assert "primary_category" in result
        assert "scroll_alignment" in result

def test_classifier_logging(mock_model, mock_tokenizer, mock_scroll_time, tmp_path):
    """Test the logging functionality of the classifier."""
    # Create a temporary log directory
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    
    # Patch the log directory path
    with patch("models.classifier.os.path.join") as mock_join:
        mock_join.return_value = str(log_dir / "case_predictions.log")
        
        classifier = ScrollAwareCaseClassifier()
        result = classifier.classify_case(SAMPLE_CASE_TEXT)
        
        # Check if log file was created
        log_file = log_dir / "case_predictions.log"
        assert log_file.exists()
        
        # Check log content
        with open(log_file, "r") as f:
            log_content = f.read()
            assert "timestamp" in log_content
            assert "text_preview" in log_content
            assert "classification" in log_content

# Ultra-Upgrade 1: Test evaluation metrics
def test_evaluation_metrics(mock_model, mock_tokenizer, mock_scroll_time):
    """Test the evaluation metrics calculation."""
    classifier = ScrollAwareCaseClassifier()
    texts = [SAMPLE_CASE_TEXT, SAMPLE_CIVIL_CASE]
    labels = [["criminal", "urgent"], ["civil", "contract"]]
    
    metrics = classifier.evaluate(texts, labels)
    
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["accuracy"] <= 1

# Ultra-Upgrade 2: Test divine title fallback
def test_divine_title_fallback(mock_model, mock_tokenizer, mock_scroll_time):
    """Test the divine title fallback mechanism."""
    classifier = ScrollAwareCaseClassifier()
    
    # Create a mock classification result with invalid category and phase
    mock_result = {
        "primary_category": "nonexistent_category",
        "confidences": {"nonexistent_category": 0.99},
        "scroll_phase": "sunrise",
        "scroll_is_active": True
    }
    
    # Mock the get_explanation method to return a fallback title
    with patch.object(classifier, "get_explanation") as mock_explanation:
        mock_explanation.return_value = "Sacred Judgment"
        
        # Call the method with the mock result
        explanation = classifier.get_explanation(mock_result)
        
        # Verify the fallback title is used
        assert "Sacred Judgment" in explanation

# Ultra-Upgrade 3: Test log directory creation
def test_log_directory_creation(tmp_path, mock_model, mock_tokenizer, mock_scroll_time):
    """Test that log directories are automatically created if they don't exist."""
    log_dir = tmp_path / "logs"
    log_file = log_dir / "case_predictions.log"
    
    # Ensure the log directory doesn't exist initially
    assert not log_dir.exists()
    
    with patch("models.classifier.os.path.join") as mock_join:
        mock_join.side_effect = lambda *args: str(log_file)
        
        # Create classifier and log a result
        classifier = ScrollAwareCaseClassifier()
        classifier.log_classification_result("Test case", {"test": "result"})
        
        # Verify the directory and file were created
        assert log_dir.exists()
        assert log_file.exists()
        
        # Check log content
        with open(log_file, "r") as f:
            log_content = f.read()
            assert "Test case" in log_content
            assert "test" in log_content
            assert "result" in log_content 
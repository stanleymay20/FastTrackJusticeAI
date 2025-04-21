import os
import json
import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from backend.app.utils.legal_principle_synthesizer import LegalPrincipleSynthesizer

# Sample case text for testing
SAMPLE_CASE_TEXT = """
In the case of Smith v. Jones, the plaintiff alleges that the defendant breached a contract
by failing to deliver goods as specified in the agreement. The court must determine whether
the defendant's actions constituted a material breach of the contract.

The court finds that the defendant's failure to deliver the goods within the specified time
frame constitutes a material breach of the contract. The court holds that time was of the
essence in this agreement, and the delay caused significant financial harm to the plaintiff.

The court applies the principle that a material breach of contract excuses the non-breaching
party from further performance and allows for recovery of damages. The court recognizes that
the defendant's actions were not excused by any force majeure clause or other provision in
the contract.
"""

SAMPLE_HOLDING = """
The court holds that the defendant's failure to deliver goods within the specified time
frame constitutes a material breach of the contract, excusing the plaintiff from further
performance and entitling the plaintiff to damages.
"""

SAMPLE_REASONING = """
The court reasons that time was of the essence in this agreement, as evidenced by the
explicit language in the contract and the plaintiff's need for timely delivery to fulfill
obligations to third parties. The court applies the established principle that a material
breach of contract excuses the non-breaching party from further performance and allows for
recovery of damages. The court finds that the defendant's actions were not excused by any
force majeure clause or other provision in the contract.
"""

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_extraction_model():
    """Create a mock extraction model"""
    with patch("backend.app.utils.legal_principle_synthesizer.SentenceTransformer") as mock:
        model_instance = MagicMock()
        
        # Mock the encode method to return predefined embeddings
        def mock_encode(texts, **kwargs):
            # Create random embeddings for testing
            if isinstance(texts, str):
                return np.random.rand(384)  # Typical BERT embedding size
            else:
                return np.array([np.random.rand(384) for _ in texts])
                
        model_instance.encode = mock_encode
        model_instance.to.return_value = None
        mock.return_value = model_instance
        yield mock

@pytest.fixture
def mock_summarization_model():
    """Create a mock summarization model"""
    with patch("backend.app.utils.legal_principle_synthesizer.AutoModelForSeq2SeqGeneration") as mock:
        model_instance = MagicMock()
        
        # Mock the generate method to return predefined outputs
        def mock_generate(**kwargs):
            # Create a tensor of token IDs that will decode to a sample summary
            import torch
            token_ids = torch.tensor([[101, 102, 103, 104, 105, 102, 103, 104, 105, 102]])
            return token_ids
            
        model_instance.generate = mock_generate
        model_instance.to.return_value = None
        mock.from_pretrained.return_value = model_instance
        yield mock

@pytest.fixture
def mock_summarization_tokenizer():
    """Create a mock summarization tokenizer"""
    with patch("backend.app.utils.legal_principle_synthesizer.AutoTokenizer") as mock:
        tokenizer_instance = MagicMock()
        
        # Mock the tokenizer to return predefined inputs
        def mock_tokenizer_func(texts, **kwargs):
            import torch
            return {
                "input_ids": torch.tensor([[101, 102, 103, 104, 105, 102, 103, 104, 105, 102]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
            }
            
        # Mock the batch_decode method to return sample summaries
        def mock_batch_decode(token_ids, **kwargs):
            return ["This is a summarized legal principle about contract breach."]
            
        tokenizer_instance.return_value = tokenizer_instance
        tokenizer_instance.__call__ = mock_tokenizer_func
        tokenizer_instance.batch_decode = mock_batch_decode
        mock.from_pretrained.return_value = tokenizer_instance
        yield mock

@pytest.fixture
def mock_classification_model():
    """Create a mock classification model"""
    with patch("backend.app.utils.legal_principle_synthesizer.AutoModelForSequenceClassification") as mock:
        model_instance = MagicMock()
        model_instance.to.return_value = None
        mock.from_pretrained.return_value = model_instance
        yield mock

@pytest.fixture
def mock_classification_tokenizer():
    """Create a mock classification tokenizer"""
    with patch("backend.app.utils.legal_principle_synthesizer.AutoTokenizer") as mock:
        tokenizer_instance = MagicMock()
        tokenizer_instance.from_pretrained.return_value = tokenizer_instance
        mock.from_pretrained.return_value = tokenizer_instance
        yield mock

@pytest.fixture
def synthesizer(
    temp_cache_dir, 
    mock_extraction_model, 
    mock_summarization_model, 
    mock_summarization_tokenizer,
    mock_classification_model,
    mock_classification_tokenizer
):
    """Create a LegalPrincipleSynthesizer instance with mocked dependencies"""
    with patch("backend.app.utils.legal_principle_synthesizer.torch.cuda.is_available", return_value=False):
        synthesizer = LegalPrincipleSynthesizer(
            extraction_model="test-extraction-model",
            summarization_model="test-summarization-model",
            classification_model="test-classification-model",
            device="cpu",
            cache_dir=temp_cache_dir
        )
        return synthesizer

def test_initialization(synthesizer, temp_cache_dir):
    """Test initialization of LegalPrincipleSynthesizer"""
    assert synthesizer.extraction_model_name == "test-extraction-model"
    assert synthesizer.summarization_model_name == "test-summarization-model"
    assert synthesizer.classification_model_name == "test-classification-model"
    assert synthesizer.device == "cpu"
    assert synthesizer.cache_dir == temp_cache_dir
    assert os.path.exists(temp_cache_dir)
    assert len(synthesizer.categories) > 0
    assert len(synthesizer.category_embeddings) == len(synthesizer.categories)

def test_split_into_sentences(synthesizer):
    """Test splitting text into sentences"""
    text = "This is sentence one. This is sentence two.\n\nThis is sentence three. This is sentence four."
    sentences = synthesizer._split_into_sentences(text)
    
    assert len(sentences) == 4
    assert "sentence one" in sentences[0]
    assert "sentence two" in sentences[1]
    assert "sentence three" in sentences[2]
    assert "sentence four" in sentences[3]

def test_filter_principle_candidates(synthesizer):
    """Test filtering principle candidates"""
    sentences = [
        "This is a regular sentence.",
        "The court finds that the defendant is liable.",
        "The weather was nice that day.",
        "The court holds that the contract was breached.",
        "The principle of stare decisis applies here."
    ]
    
    candidates = synthesizer._filter_principle_candidates(sentences)
    
    assert len(candidates) == 3
    assert "court finds" in candidates[0].lower()
    assert "court holds" in candidates[1].lower()
    assert "principle" in candidates[2].lower()

def test_extract_with_model(synthesizer, mock_extraction_model):
    """Test extracting principles with the model"""
    candidates = [
        "The court finds that the defendant is liable.",
        "The court holds that the contract was breached.",
        "The principle of stare decisis applies here."
    ]
    
    principles = synthesizer._extract_with_model(candidates)
    
    assert len(principles) > 0
    assert isinstance(principles, list)
    assert all(isinstance(p, str) for p in principles)
    
    # Verify that the extraction model was called
    mock_extraction_model.return_value.encode.assert_called()

def test_extract_principles(synthesizer):
    """Test extracting principles from case text"""
    principles = synthesizer.extract_principles(SAMPLE_CASE_TEXT)
    
    assert len(principles) > 0
    assert isinstance(principles, list)
    assert all(isinstance(p, str) for p in principles)
    
    # Test with holding and reasoning
    principles_with_holding = synthesizer.extract_principles(
        SAMPLE_CASE_TEXT, 
        holding=SAMPLE_HOLDING
    )
    
    assert len(principles_with_holding) > 0
    
    principles_with_reasoning = synthesizer.extract_principles(
        SAMPLE_CASE_TEXT, 
        holding=SAMPLE_HOLDING,
        reasoning=SAMPLE_REASONING
    )
    
    assert len(principles_with_reasoning) > 0

def test_summarize_principles(synthesizer, mock_summarization_model, mock_summarization_tokenizer):
    """Test summarizing principles"""
    principles = [
        "The court finds that the defendant's failure to deliver the goods within the specified time frame constitutes a material breach of the contract.",
        "The court holds that time was of the essence in this agreement, and the delay caused significant financial harm to the plaintiff."
    ]
    
    summarized = synthesizer.summarize_principles(principles)
    
    assert len(summarized) == len(principles)
    assert all(isinstance(s, str) for s in summarized)
    assert all("summarized" in s.lower() for s in summarized)  # Based on our mock
    
    # Test with empty list
    empty_summarized = synthesizer.summarize_principles([])
    assert empty_summarized == []

def test_classify_principles(synthesizer, mock_extraction_model):
    """Test classifying principles"""
    principles = [
        "The court finds that the defendant's failure to deliver the goods constitutes a breach of contract.",
        "The court holds that the defendant's actions violated the plaintiff's constitutional rights."
    ]
    
    classified = synthesizer.classify_principles(principles)
    
    assert len(classified) == len(principles)
    assert all(isinstance(c, dict) for c in classified)
    assert all("principle" in c for c in classified)
    assert all("categories" in c for c in classified)
    assert all("scores" in c for c in classified)
    assert all(len(c["categories"]) == 3 for c in classified)  # Top 3 categories
    
    # Test with empty list
    empty_classified = synthesizer.classify_principles([])
    assert empty_classified == []

def test_process_case(synthesizer):
    """Test processing a case to extract, summarize, and classify principles"""
    result = synthesizer.process_case(
        SAMPLE_CASE_TEXT,
        holding=SAMPLE_HOLDING,
        reasoning=SAMPLE_REASONING
    )
    
    assert isinstance(result, dict)
    assert "raw_principles" in result
    assert "summarized_principles" in result
    assert "classified_principles" in result
    assert "metadata" in result
    
    assert len(result["raw_principles"]) > 0
    assert len(result["summarized_principles"]) > 0
    assert len(result["classified_principles"]) > 0
    
    assert result["metadata"]["num_raw_principles"] == len(result["raw_principles"])
    assert result["metadata"]["num_summarized_principles"] == len(result["summarized_principles"])
    assert isinstance(result["metadata"]["categories_found"], list)

def test_process_case_with_cache(synthesizer, temp_cache_dir):
    """Test processing a case with caching"""
    # Process a case with a cache key
    cache_key = "test_case_123"
    result1 = synthesizer.process_case(
        SAMPLE_CASE_TEXT,
        holding=SAMPLE_HOLDING,
        reasoning=SAMPLE_REASONING,
        cache_key=cache_key
    )
    
    # Verify that the result was cached
    cache_file = os.path.join(temp_cache_dir, f"{cache_key}.json")
    assert os.path.exists(cache_file)
    
    # Process the same case again with the same cache key
    result2 = synthesizer.process_case(
        SAMPLE_CASE_TEXT,
        holding=SAMPLE_HOLDING,
        reasoning=SAMPLE_REASONING,
        cache_key=cache_key
    )
    
    # Verify that the results are the same
    assert result1 == result2

def test_get_from_cache(synthesizer, temp_cache_dir):
    """Test retrieving from cache"""
    # Create a cache file
    cache_key = "test_cache_key"
    cache_data = {"test": "data"}
    cache_file = os.path.join(temp_cache_dir, f"{cache_key}.json")
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
        
    # Retrieve from cache
    result = synthesizer._get_from_cache(cache_key)
    
    assert result == cache_data
    
    # Test with non-existent cache
    non_existent = synthesizer._get_from_cache("non_existent")
    assert non_existent is None

def test_add_to_cache(synthesizer, temp_cache_dir):
    """Test adding to cache"""
    # Add to cache
    cache_key = "test_add_cache_key"
    cache_data = {"test": "data"}
    
    synthesizer._add_to_cache(cache_key, cache_data)
    
    # Verify that the data was cached
    cache_file = os.path.join(temp_cache_dir, f"{cache_key}.json")
    assert os.path.exists(cache_file)
    
    with open(cache_file, 'r') as f:
        cached_data = json.load(f)
        
    assert cached_data == cache_data

def test_get_model_info(synthesizer):
    """Test getting model information"""
    info = synthesizer.get_model_info()
    
    assert isinstance(info, dict)
    assert info["extraction_model"] == "test-extraction-model"
    assert info["summarization_model"] == "test-summarization-model"
    assert info["classification_model"] == "test-classification-model"
    assert info["device"] == "cpu"
    assert "categories" in info
    assert "max_length" in info
    assert "batch_size" in info

def test_error_handling(synthesizer):
    """Test error handling in the synthesizer"""
    # Test with invalid case text
    result = synthesizer.process_case("")
    assert result["raw_principles"] == []
    assert result["summarized_principles"] == []
    assert result["classified_principles"] == []
    
    # Test with invalid cache key
    result = synthesizer.process_case(
        SAMPLE_CASE_TEXT,
        cache_key="invalid/key/with/slashes"
    )
    assert "raw_principles" in result
    assert "summarized_principles" in result
    assert "classified_principles" in result 
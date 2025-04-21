import pytest
import torch
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from models.judgment_drafter import JudgmentDrafter
from backend.app.utils.legal_precedent import LegalPrecedentManager
from backend.app.utils.legal_knowledge_augmenter import LegalKnowledgeAugmenter

# Sample test data
SAMPLE_CASE_TEXT = """
The defendant is charged with second-degree assault. On the night of June 15, 2023,
the defendant allegedly struck the victim with a baseball bat, causing serious injury.
Witnesses testified that they saw the defendant approach the victim from behind and
strike them without provocation. The defendant claims they acted in self-defense,
but no evidence supports this claim.
"""

SAMPLE_CASE_SUMMARY = """
This case involves a second-degree assault charge where the defendant allegedly struck
the victim with a baseball bat, causing serious injury. The defendant claims self-defense,
but witnesses testified that the attack was unprovoked.
"""

SAMPLE_CLASSIFICATION = {
    "criminal": 0.85,
    "urgent": 0.65,
    "civil": 0.15,
    "family": 0.05,
    "confidence": {
        "primary_category": "criminal",
        "primary_confidence": 0.85,
        "secondary_categories": ["urgent"],
        "uncertainty_score": 0.15
    }
}

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
    with patch("models.judgment_drafter.AutoModelForSeq2SeqGeneration") as mock:
        # Create a mock model that returns predefined outputs
        model_instance = MagicMock()
        model_instance.eval.return_value = None
        model_instance.to.return_value = None
        
        # Mock the generate method to return predefined outputs
        def mock_generate(**kwargs):
            # Create a tensor of token IDs that will decode to a sample judgment
            token_ids = torch.tensor([[101, 102, 103, 104, 105, 102, 103, 104, 105, 102]])
            return token_ids
        
        model_instance.return_value = model_instance
        model_instance.generate = mock_generate
        mock.from_pretrained.return_value = model_instance
        yield mock

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    with patch("models.judgment_drafter.AutoTokenizer") as mock:
        tokenizer_instance = MagicMock()
        tokenizer_instance.from_pretrained.return_value = tokenizer_instance
        
        # Mock the tokenizer to return predefined inputs
        def mock_tokenizer_func(text, **kwargs):
            return {
                "input_ids": torch.tensor([[101, 102, 103, 104, 105, 102, 103, 104, 105, 102]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
            }
        
        # Mock the decode method to return a sample judgment
        def mock_decode(token_ids, **kwargs):
            return """
            Introduction:
            This court has carefully considered the evidence presented in this case.
            
            Facts of the Case:
            The defendant is charged with second-degree assault. On the night of June 15, 2023,
            the defendant allegedly struck the victim with a baseball bat, causing serious injury.
            
            Legal Issues:
            The primary issue before this court is whether the defendant's actions constituted
            second-degree assault and whether the defense of self-defense is applicable.
            
            Analysis and Reasoning:
            Based on the testimony of witnesses, the defendant approached the victim from behind
            and struck them without provocation. The defendant's claim of self-defense is not
            supported by any evidence presented to this court.
            
            Conclusion and Verdict:
            Therefore, this court finds the defendant guilty of second-degree assault.
            """
        
        tokenizer_instance.return_value = tokenizer_instance
        tokenizer_instance.__call__ = mock_tokenizer_func
        tokenizer_instance.decode = mock_decode
        mock.from_pretrained.return_value = tokenizer_instance
        yield mock

@pytest.fixture
def mock_scroll_time():
    """Create a mock scroll_time function for testing."""
    with patch("models.judgment_drafter.get_scroll_time") as mock:
        mock.return_value = MOCK_SCROLL_TIME
        yield mock

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_precedent_manager():
    """Create a mock LegalPrecedentManager"""
    manager = MagicMock(spec=LegalPrecedentManager)
    manager.precedents = []
    return manager

@pytest.fixture
def mock_knowledge_augmenter():
    """Create a mock LegalKnowledgeAugmenter"""
    augmenter = MagicMock(spec=LegalKnowledgeAugmenter)
    augmenter.format_for_prompt.return_value = "RELEVANT LEGAL PRECEDENTS:\nTest precedent content"
    return augmenter

@pytest.fixture
def drafter(temp_cache_dir, mock_precedent_manager, mock_knowledge_augmenter):
    """Create a JudgmentDrafter instance with mocked dependencies"""
    with patch('backend.app.utils.judgment_drafter.LegalPrecedentManager', return_value=mock_precedent_manager), \
         patch('backend.app.utils.judgment_drafter.LegalKnowledgeAugmenter', return_value=mock_knowledge_augmenter):
        drafter = JudgmentDrafter(
            model_name="test-model",
            cache_dir=temp_cache_dir,
            precedents_file="test_precedents.json",
            language="en",
            use_rag=True
        )
        return drafter

@pytest.fixture
def sample_case_text():
    """Sample case text for testing"""
    return """
    The plaintiff, John Doe, alleges that the defendant, Acme Corporation, 
    violated his constitutional rights by implementing a policy that 
    discriminates against individuals with disabilities. The plaintiff 
    seeks damages and an injunction to prevent further violations.
    
    The defendant argues that the policy is necessary for public safety 
    and does not violate any constitutional provisions. The court must 
    determine whether the policy violates the Equal Protection Clause 
    of the Fourteenth Amendment.
    """

def test_judgment_drafter_initialization(mock_model, mock_tokenizer):
    """Test the initialization of the JudgmentDrafter."""
    drafter = JudgmentDrafter()
    assert drafter is not None
    assert drafter.device in ["cuda", "cpu"]
    assert hasattr(drafter, "model")
    assert hasattr(drafter, "tokenizer")
    assert hasattr(drafter, "cache")

def test_preprocess_input(mock_model, mock_tokenizer, mock_scroll_time):
    """Test the preprocess_input method."""
    drafter = JudgmentDrafter()
    
    # Test with basic parameters
    prompt = drafter.preprocess_input(SAMPLE_CASE_TEXT, SAMPLE_CASE_SUMMARY, SAMPLE_CLASSIFICATION)
    assert "Case Type: criminal" in prompt
    assert "Case Summary:" in prompt
    assert "Key Facts:" in prompt
    assert "Generate a legal judgment" in prompt
    
    # Test with additional parameters
    prompt = drafter.preprocess_input(
        SAMPLE_CASE_TEXT, 
        SAMPLE_CASE_SUMMARY, 
        SAMPLE_CLASSIFICATION,
        case_id="CASE-2023-001",
        jurisdiction="Supreme Court",
        language="Spanish"
    )
    assert "Case ID: CASE-2023-001" in prompt
    assert "Jurisdiction: Supreme Court" in prompt
    assert "Translate to Spanish:" in prompt
    assert "Scroll Phase: dawn" in prompt
    assert "Gate: 3" in prompt

def test_generate_judgment(mock_model, mock_tokenizer, mock_scroll_time):
    """Test the generate_judgment method."""
    drafter = JudgmentDrafter()
    
    # Test with basic parameters
    judgment = drafter.generate_judgment(SAMPLE_CASE_TEXT, SAMPLE_CASE_SUMMARY, SAMPLE_CLASSIFICATION)
    assert "Introduction:" in judgment
    assert "Facts of the Case:" in judgment
    assert "Legal Issues:" in judgment
    assert "Analysis and Reasoning:" in judgment
    assert "Conclusion and Verdict:" in judgment
    
    # Test with additional parameters
    judgment = drafter.generate_judgment(
        SAMPLE_CASE_TEXT, 
        SAMPLE_CASE_SUMMARY, 
        SAMPLE_CLASSIFICATION,
        case_id="CASE-2023-001",
        jurisdiction="Supreme Court",
        language="Spanish"
    )
    assert "Introduction:" in judgment
    assert "Facts of the Case:" in judgment
    
    # Test caching
    # The second call should use the cache
    with patch.object(drafter, "preprocess_input") as mock_preprocess:
        judgment = drafter.generate_judgment(
            SAMPLE_CASE_TEXT, 
            SAMPLE_CASE_SUMMARY, 
            SAMPLE_CLASSIFICATION,
            case_id="CASE-2023-001",
            jurisdiction="Supreme Court",
            language="Spanish"
        )
        # preprocess_input should not be called again
        mock_preprocess.assert_not_called()

def test_postprocess_judgment(mock_model, mock_tokenizer):
    """Test the postprocess_judgment method."""
    drafter = JudgmentDrafter()
    
    # Test with a judgment missing some sections
    incomplete_judgment = """
    Introduction:
    This court has carefully considered the evidence.
    
    Facts of the Case:
    The defendant is charged with assault.
    
    Conclusion and Verdict:
    The defendant is guilty.
    """
    
    processed_judgment = drafter.postprocess_judgment(incomplete_judgment)
    assert "Introduction:" in processed_judgment
    assert "Facts of the Case:" in processed_judgment
    assert "Legal Issues:" in processed_judgment
    assert "Analysis and Reasoning:" in processed_judgment
    assert "Conclusion and Verdict:" in processed_judgment
    
    # Test with a complete judgment
    complete_judgment = """
    Introduction:
    This court has carefully considered the evidence.
    
    Facts of the Case:
    The defendant is charged with assault.
    
    Legal Issues:
    Whether the defendant committed assault.
    
    Analysis and Reasoning:
    The evidence shows the defendant committed assault.
    
    Conclusion and Verdict:
    The defendant is guilty.
    """
    
    processed_judgment = drafter.postprocess_judgment(complete_judgment)
    assert processed_judgment == complete_judgment.strip()

def test_evaluate_judgment(mock_model, mock_tokenizer):
    """Test the evaluate_judgment method."""
    drafter = JudgmentDrafter()
    
    # Create sample judgment and reference
    judgment = """
    Introduction:
    This court has carefully considered the evidence.
    
    Facts of the Case:
    The defendant is charged with assault.
    
    Legal Issues:
    Whether the defendant committed assault.
    
    Analysis and Reasoning:
    The evidence shows the defendant committed assault.
    
    Conclusion and Verdict:
    The defendant is guilty.
    """
    
    reference = """
    Introduction:
    This court has carefully considered the evidence presented.
    
    Facts of the Case:
    The defendant is charged with second-degree assault.
    
    Legal Issues:
    Whether the defendant committed second-degree assault.
    
    Analysis and Reasoning:
    The evidence clearly shows the defendant committed second-degree assault.
    
    Conclusion and Verdict:
    The defendant is found guilty of second-degree assault.
    """
    
    # Mock the Rouge class
    with patch("models.judgment_drafter.Rouge") as mock_rouge:
        mock_rouge_instance = MagicMock()
        mock_rouge_instance.get_scores.return_value = [
            {
                "rouge-1": {"f": 0.75},
                "rouge-2": {"f": 0.65},
                "rouge-l": {"f": 0.70}
            }
        ]
        mock_rouge.return_value = mock_rouge_instance
        
        # Evaluate the judgment
        metrics = drafter.evaluate_judgment(judgment, reference)
        
        # Check metrics
        assert "rouge-1" in metrics
        assert "rouge-2" in metrics
        assert "rouge-l" in metrics
        assert "legal_terminology" in metrics
        assert isinstance(metrics["legal_terminology"], dict)
        assert "jurisdiction" in metrics["legal_terminology"]
        assert "procedure" in metrics["legal_terminology"]
        assert "substantive" in metrics["legal_terminology"]
        assert "conclusion" in metrics["legal_terminology"]

def test_extract_legal_terminology(mock_model, mock_tokenizer):
    """Test the _extract_legal_terminology method."""
    drafter = JudgmentDrafter()
    
    # Create a text with legal terminology
    text = """
    This court has jurisdiction over this matter. The procedure for filing a motion
    has been followed correctly. The defendant's liability for negligence is clear.
    Therefore, the court finds in favor of the plaintiff.
    """
    
    # Extract terminology
    terminology = drafter._extract_legal_terminology(text)
    
    # Check terminology
    assert isinstance(terminology, dict)
    assert "jurisdiction" in terminology
    assert "procedure" in terminology
    assert "substantive" in terminology
    assert "conclusion" in terminology
    assert terminology["jurisdiction"] > 0
    assert terminology["procedure"] > 0
    assert terminology["substantive"] > 0
    assert terminology["conclusion"] > 0

def test_log_judgment_generation(mock_model, mock_tokenizer, mock_scroll_time, tmp_path):
    """Test the log_judgment_generation method."""
    # Create a temporary log directory
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "judgment_generations.log"
    
    # Patch the log file path
    with patch("models.judgment_drafter.os.path.join") as mock_join:
        mock_join.side_effect = lambda *args: str(log_file)
        
        drafter = JudgmentDrafter()
        drafter.log_judgment_generation(
            SAMPLE_CASE_TEXT, 
            SAMPLE_CASE_SUMMARY, 
            SAMPLE_CLASSIFICATION, 
            "Sample judgment text"
        )
        
        # Check if log file was created
        assert log_file.exists()
        
        # Check log content
        with open(log_file, "r") as f:
            log_content = f.read()
            log_data = json.loads(log_content)
            assert "timestamp" in log_data
            assert "text_preview" in log_data
            assert "summary" in log_data
            assert "classification" in log_data
            assert "judgment_preview" in log_data

def test_save_model(mock_model, mock_tokenizer, tmp_path):
    """Test the save_model method."""
    # Create a temporary output directory
    output_dir = tmp_path / "model_output"
    
    # Patch the model and tokenizer save methods
    with patch.object(mock_model.from_pretrained.return_value, "save_pretrained") as mock_save_model, \
         patch.object(mock_tokenizer.from_pretrained.return_value, "save_pretrained") as mock_save_tokenizer:
        
        drafter = JudgmentDrafter()
        drafter.save_model(str(output_dir))
        
        # Check if save methods were called
        mock_save_model.assert_called_once_with(str(output_dir))
        mock_save_tokenizer.assert_called_once_with(str(output_dir))

def test_scroll_phase_enrichment(mock_model, mock_tokenizer):
    """Test the scroll phase enrichment in judgment generation."""
    drafter = JudgmentDrafter()
    
    # Test with different scroll phases
    phases = ["dawn", "noon", "dusk", "night"]
    for phase in phases:
        with patch("models.judgment_drafter.get_scroll_time") as mock:
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
            
            # Get the prompt
            prompt = drafter.preprocess_input(SAMPLE_CASE_TEXT, SAMPLE_CASE_SUMMARY, SAMPLE_CLASSIFICATION)
            
            # Check for phase-specific enrichment
            if phase == "dawn":
                assert "awakening and clarity" in prompt
                assert "new beginnings and fresh perspectives" in prompt
            elif phase == "noon":
                assert "balance and justice" in prompt
                assert "equilibrium and fairness" in prompt
            elif phase == "dusk":
                assert "resolution and closure" in prompt
                assert "conclusion and finality" in prompt
            elif phase == "night":
                assert "reflection and wisdom" in prompt
                assert "contemplation and deeper understanding" in prompt
            
            # Check for gate-specific enrichment
            assert "Gate: 3" in prompt
            if phase == "dawn":
                assert "courageous judgment with balanced wisdom" in prompt
            elif phase == "noon":
                assert "courageous judgment with divine insight" in prompt
            elif phase == "dusk":
                assert "courageous judgment with clarity of purpose" in prompt
            elif phase == "night":
                assert "courageous judgment with compassionate understanding" in prompt

def test_initialization(drafter, temp_cache_dir):
    """Test initialization of JudgmentDrafter"""
    assert drafter.model_name == "test-model"
    assert drafter.cache_dir == temp_cache_dir
    assert drafter.language == "en"
    assert drafter.use_rag is True
    assert drafter.precedent_manager is not None
    assert drafter.knowledge_augmenter is not None
    assert os.path.exists(temp_cache_dir)

def test_initialization_without_rag(temp_cache_dir):
    """Test initialization without RAG"""
    drafter = JudgmentDrafter(
        model_name="test-model",
        cache_dir=temp_cache_dir,
        use_rag=False
    )
    assert drafter.use_rag is False
    assert drafter.precedent_manager is None
    assert drafter.knowledge_augmenter is None

def test_preprocess_input(drafter, sample_case_text, mock_knowledge_augmenter):
    """Test preprocessing of input"""
    result = drafter.preprocess_input(
        case_text=sample_case_text,
        case_id="TEST-2023-001",
        jurisdiction="Federal",
        scroll_phase="noon"
    )
    
    # Check basic fields
    assert result["case_text"] == sample_case_text
    assert result["case_id"] == "TEST-2023-001"
    assert result["jurisdiction"] == "Federal"
    assert result["scroll_phase"] == "noon"
    
    # Check extracted summary
    assert "John Doe" in result["case_summary"]
    assert "Acme Corporation" in result["case_summary"]
    
    # Check terminology identification
    assert "constitutional" in [term.lower() for term in result["terminology"]]
    assert "equal protection" in [term.lower() for term in result["terminology"]]
    
    # Check scroll phase enrichment
    assert result["phase_enrichment"]["tone"] == "clear and decisive"
    assert result["phase_enrichment"]["emphasis"] == "clarity and directness"
    
    # Check RAG integration
    assert result["precedent_knowledge"] == "RELEVANT LEGAL PRECEDENTS:\nTest precedent content"
    mock_knowledge_augmenter.format_for_prompt.assert_called_once_with(
        sample_case_text,
        scroll_phase="noon",
        jurisdiction="Federal"
    )

def test_preprocess_input_without_rag(temp_cache_dir, sample_case_text):
    """Test preprocessing without RAG"""
    drafter = JudgmentDrafter(
        model_name="test-model",
        cache_dir=temp_cache_dir,
        use_rag=False
    )
    
    result = drafter.preprocess_input(
        case_text=sample_case_text,
        case_id="TEST-2023-001",
        jurisdiction="Federal",
        scroll_phase="noon"
    )
    
    # Check that precedent knowledge is empty
    assert result["precedent_knowledge"] == ""

def test_generate_judgment(drafter, sample_case_text, temp_cache_dir):
    """Test judgment generation"""
    # Mock the model call
    with patch.object(drafter, '_call_model', return_value="GENERATED JUDGMENT"):
        # Preprocess the input
        preprocessed = drafter.preprocess_input(
            case_text=sample_case_text,
            case_id="TEST-2023-001",
            jurisdiction="Federal",
            scroll_phase="noon"
        )
        
        # Generate the judgment
        judgment = drafter.generate_judgment(preprocessed)
        
        # Check the result
        assert judgment == "GENERATED JUDGMENT"
        
        # Check that the judgment was cached
        cache_files = os.listdir(temp_cache_dir)
        assert len(cache_files) == 1
        assert cache_files[0].endswith(".json")
        
        # Check cache content
        with open(os.path.join(temp_cache_dir, cache_files[0]), 'r') as f:
            cache_data = json.load(f)
            assert cache_data["judgment"] == "GENERATED JUDGMENT"
            assert "timestamp" in cache_data

def test_generate_judgment_with_cache(drafter, sample_case_text, temp_cache_dir):
    """Test judgment generation with cache hit"""
    # Create a cached judgment
    preprocessed = drafter.preprocess_input(
        case_text=sample_case_text,
        case_id="TEST-2023-001",
        jurisdiction="Federal",
        scroll_phase="noon"
    )
    
    cache_key = drafter._generate_cache_key(preprocessed)
    drafter._add_to_cache(cache_key, "CACHED JUDGMENT")
    
    # Mock the model call (should not be called)
    with patch.object(drafter, '_call_model') as mock_call:
        # Generate the judgment
        judgment = drafter.generate_judgment(preprocessed)
        
        # Check the result
        assert judgment == "CACHED JUDGMENT"
        
        # Verify that the model was not called
        mock_call.assert_not_called()

def test_postprocess_judgment(drafter):
    """Test judgment postprocessing"""
    raw_judgment = """
    INTRODUCTION:
    This case involves a constitutional challenge.
    
    FACTS:
    The plaintiff alleges discrimination.
    
    ANALYSIS:
    The court finds that the policy violates equal protection.
    
    CONCLUSION:
    The court grants the injunction.
    """
    
    processed = drafter.postprocess_judgment(raw_judgment)
    
    # Check that section headers are properly formatted
    assert "\n\nINTRODUCTION:\n" in processed
    assert "\n\nFACTS:\n" in processed
    assert "\n\nANALYSIS:\n" in processed
    assert "\n\nCONCLUSION:\n" in processed
    
    # Check that formatting is cleaned up
    assert processed.count("\n\n\n") == 0  # No excessive newlines

def test_evaluate_judgment(drafter, sample_case_text):
    """Test judgment evaluation"""
    # Preprocess the input
    preprocessed = drafter.preprocess_input(
        case_text=sample_case_text,
        case_id="TEST-2023-001",
        jurisdiction="Federal",
        scroll_phase="noon"
    )
    
    # Evaluate a judgment
    judgment = "This is a test judgment about constitutional rights."
    metrics = drafter.evaluate_judgment(judgment, sample_case_text, preprocessed)
    
    # Check that metrics are returned
    assert "coherence" in metrics
    assert "legal_accuracy" in metrics
    assert "terminology_usage" in metrics
    assert "scroll_alignment" in metrics
    
    # Check that scroll alignment is calculated correctly
    assert metrics["scroll_alignment"] == 0.75  # For noon phase

def test_build_judgment_prompt(drafter, sample_case_text):
    """Test building the judgment prompt"""
    # Preprocess the input
    preprocessed = drafter.preprocess_input(
        case_text=sample_case_text,
        case_id="TEST-2023-001",
        jurisdiction="Federal",
        scroll_phase="noon"
    )
    
    # Build the prompt
    prompt = drafter._build_judgment_prompt(preprocessed, "en")
    
    # Check that the prompt contains all necessary sections
    assert "CASE:" in prompt
    assert sample_case_text in prompt
    assert "JURISDICTION: Federal" in prompt
    assert "SCROLL PHASE: noon" in prompt
    assert "PHASE TONE: clear and decisive" in prompt
    assert "RELEVANT LEGAL TERMINOLOGY:" in prompt
    assert "RELEVANT LEGAL PRECEDENTS:" in prompt
    assert "LANGUAGE: Generate the judgment in en." in prompt
    assert "STRUCTURE THE JUDGMENT AS FOLLOWS:" in prompt
    assert "1. Introduction and case summary" in prompt
    assert "2. Statement of facts" in prompt
    assert "3. Issues presented" in prompt
    assert "4. Analysis and reasoning" in prompt
    assert "5. Conclusion and judgment" in prompt

def test_extract_case_summary(drafter, sample_case_text):
    """Test case summary extraction"""
    summary = drafter._extract_case_summary(sample_case_text)
    
    # Check that the summary contains key information
    assert "John Doe" in summary
    assert "Acme Corporation" in summary
    assert "constitutional rights" in summary.lower()

def test_identify_relevant_terminology(drafter, sample_case_text):
    """Test legal terminology identification"""
    terminology = drafter._identify_relevant_terminology(sample_case_text)
    
    # Check that relevant terms are identified
    assert any("constitutional" in term.lower() for term in terminology)
    assert any("equal protection" in term.lower() for term in terminology)

def test_generate_cache_key(drafter, sample_case_text):
    """Test cache key generation"""
    # Test with case_id
    preprocessed_with_id = {
        "case_id": "TEST-2023-001",
        "scroll_phase": "noon"
    }
    key_with_id = drafter._generate_cache_key(preprocessed_with_id)
    assert key_with_id == "TEST-2023-001_noon"
    
    # Test without case_id
    preprocessed_without_id = {
        "case_text": sample_case_text,
        "scroll_phase": "noon",
        "jurisdiction": "Federal"
    }
    key_without_id = drafter._generate_cache_key(preprocessed_without_id)
    assert len(key_without_id) == 32  # MD5 hash length
    assert key_without_id != key_with_id  # Different inputs should have different keys 
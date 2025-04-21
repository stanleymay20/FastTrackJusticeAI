import os
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import numpy as np

from backend.app.utils.scroll_memory_intelligence import ScrollMemoryIntelligence

# Sample data for testing
SAMPLE_CASE_ID = "case_123"
SAMPLE_PRINCIPLE = "The right to due process is fundamental to justice"
SAMPLE_SCROLL_ALIGNMENT = "This aligns with the scroll teaching of fairness in judgment"
SAMPLE_PROPHETIC_INSIGHT = "The evolution of due process rights reflects divine principles of justice"

@pytest.fixture
def temp_memory_file():
    """Create a temporary memory file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        # Write initial memory data
        initial_data = {
            "entries": [],
            "metadata": {"created": datetime.now().isoformat()}
        }
        json.dump(initial_data, tmp)
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Clean up
    os.unlink(tmp_path)

@pytest.fixture
def mock_model():
    """Mock the sentence transformer model"""
    with patch('backend.app.utils.scroll_memory_intelligence.SentenceTransformer') as mock:
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_instance.encode.return_value = np.random.rand(1, 384).astype('float32')
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def mock_faiss():
    """Mock the FAISS index"""
    with patch('backend.app.utils.scroll_memory_intelligence.faiss.IndexFlatL2') as mock:
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.search.return_value = (np.array([[0.5, 0.3, 0.1]]), np.array([[0, 1, 2]]))
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def scroll_memory(temp_memory_file, mock_model, mock_faiss):
    """Create a ScrollMemoryIntelligence instance with mocked dependencies"""
    memory = ScrollMemoryIntelligence(memory_file=temp_memory_file)
    return memory

def test_initialization(scroll_memory, temp_memory_file):
    """Test initialization of ScrollMemoryIntelligence"""
    assert scroll_memory.memory_file == temp_memory_file
    assert scroll_memory.model is not None
    assert scroll_memory.index is not None
    assert "entries" in scroll_memory.memory
    assert "metadata" in scroll_memory.memory

def test_add_memory_entry(scroll_memory):
    """Test adding a memory entry"""
    # Add a memory entry
    success = scroll_memory.add_memory_entry(
        case_id=SAMPLE_CASE_ID,
        principle=SAMPLE_PRINCIPLE,
        scroll_alignment=SAMPLE_SCROLL_ALIGNMENT,
        prophetic_insight=SAMPLE_PROPHETIC_INSIGHT,
        confidence=0.9,
        tags=["justice", "due_process"]
    )
    
    assert success is True
    assert len(scroll_memory.memory["entries"]) == 1
    
    # Check entry data
    entry = scroll_memory.memory["entries"][0]
    assert entry["case_id"] == SAMPLE_CASE_ID
    assert entry["principle"] == SAMPLE_PRINCIPLE
    assert entry["scroll_alignment"] == SAMPLE_SCROLL_ALIGNMENT
    assert entry["prophetic_insight"] == SAMPLE_PROPHETIC_INSIGHT
    assert entry["confidence"] == 0.9
    assert "justice" in entry["tags"]
    assert "due_process" in entry["tags"]

def test_find_similar_memories(scroll_memory):
    """Test finding similar memories"""
    # Add multiple memory entries
    scroll_memory.add_memory_entry(
        case_id=SAMPLE_CASE_ID,
        principle=SAMPLE_PRINCIPLE,
        scroll_alignment=SAMPLE_SCROLL_ALIGNMENT,
        prophetic_insight=SAMPLE_PROPHETIC_INSIGHT,
        confidence=0.9
    )
    
    scroll_memory.add_memory_entry(
        case_id="case_456",
        principle="Freedom of speech is essential to democracy",
        scroll_alignment="This aligns with the scroll teaching of truth in expression",
        prophetic_insight="The protection of speech reflects divine principles of truth",
        confidence=0.8
    )
    
    # Find similar memories
    results = scroll_memory.find_similar_memories(
        query="due process rights",
        k=2,
        min_confidence=0.5
    )
    
    assert len(results) == 2
    assert "similarity" in results[0]
    assert results[0]["case_id"] == SAMPLE_CASE_ID

def test_get_memory_trail(scroll_memory):
    """Test getting a memory trail"""
    # Add multiple memory entries with related principles
    scroll_memory.add_memory_entry(
        case_id=SAMPLE_CASE_ID,
        principle=SAMPLE_PRINCIPLE,
        scroll_alignment=SAMPLE_SCROLL_ALIGNMENT,
        prophetic_insight=SAMPLE_PROPHETIC_INSIGHT,
        confidence=0.9
    )
    
    scroll_memory.add_memory_entry(
        case_id="case_456",
        principle="Due process requires notice and opportunity to be heard",
        scroll_alignment="This aligns with the scroll teaching of fairness in judgment",
        prophetic_insight="The requirement of notice reflects divine principles of justice",
        confidence=0.8
    )
    
    scroll_memory.add_memory_entry(
        case_id="case_789",
        principle="Freedom of speech is essential to democracy",
        scroll_alignment="This aligns with the scroll teaching of truth in expression",
        prophetic_insight="The protection of speech reflects divine principles of truth",
        confidence=0.7
    )
    
    # Get memory trail
    trail = scroll_memory.get_memory_trail(
        case_id=SAMPLE_CASE_ID,
        max_depth=2
    )
    
    assert len(trail) >= 2
    assert trail[0]["case_id"] == SAMPLE_CASE_ID
    assert trail[1]["case_id"] == "case_456"

def test_get_prophetic_patterns(scroll_memory):
    """Test identifying prophetic patterns"""
    # Add multiple memory entries with the same theme
    now = datetime.now()
    
    scroll_memory.add_memory_entry(
        case_id=SAMPLE_CASE_ID,
        principle=SAMPLE_PRINCIPLE,
        scroll_alignment="Justice theme",
        prophetic_insight=SAMPLE_PROPHETIC_INSIGHT,
        confidence=0.9,
        metadata={"timestamp": (now - timedelta(days=30)).isoformat()}
    )
    
    scroll_memory.add_memory_entry(
        case_id="case_456",
        principle="Freedom of speech is essential to democracy",
        scroll_alignment="Justice theme",
        prophetic_insight="The protection of speech reflects divine principles of truth",
        confidence=0.8,
        metadata={"timestamp": (now - timedelta(days=20)).isoformat()}
    )
    
    scroll_memory.add_memory_entry(
        case_id="case_789",
        principle="Equal protection under the law",
        scroll_alignment="Justice theme",
        prophetic_insight="Equal protection reflects divine principles of fairness",
        confidence=0.7,
        metadata={"timestamp": (now - timedelta(days=10)).isoformat()}
    )
    
    # Add an entry with a different theme
    scroll_memory.add_memory_entry(
        case_id="case_101",
        principle="Right to privacy",
        scroll_alignment="Privacy theme",
        prophetic_insight="Privacy rights reflect divine principles of respect",
        confidence=0.8,
        metadata={"timestamp": now.isoformat()}
    )
    
    # Get prophetic patterns
    patterns = scroll_memory.get_prophetic_patterns(
        time_range=(now - timedelta(days=40), now),
        min_confidence=0.6
    )
    
    assert len(patterns) == 2  # Justice theme and Privacy theme
    assert patterns[0]["theme"] == "Justice theme"
    assert patterns[0]["count"] == 3
    assert patterns[1]["theme"] == "Privacy theme"
    assert patterns[1]["count"] == 1

def test_get_memory_statistics(scroll_memory):
    """Test getting memory statistics"""
    # Add memory entries with different confidence levels and tags
    scroll_memory.add_memory_entry(
        case_id=SAMPLE_CASE_ID,
        principle=SAMPLE_PRINCIPLE,
        scroll_alignment=SAMPLE_SCROLL_ALIGNMENT,
        prophetic_insight=SAMPLE_PROPHETIC_INSIGHT,
        confidence=0.9,
        tags=["justice", "due_process"]
    )
    
    scroll_memory.add_memory_entry(
        case_id="case_456",
        principle="Freedom of speech is essential to democracy",
        scroll_alignment="This aligns with the scroll teaching of truth in expression",
        prophetic_insight="The protection of speech reflects divine principles of truth",
        confidence=0.6,
        tags=["freedom", "speech"]
    )
    
    scroll_memory.add_memory_entry(
        case_id="case_789",
        principle="Equal protection under the law",
        scroll_alignment="This aligns with the scroll teaching of fairness",
        prophetic_insight="Equal protection reflects divine principles of fairness",
        confidence=0.4,
        tags=["equality", "justice"]
    )
    
    # Get memory statistics
    stats = scroll_memory.get_memory_statistics()
    
    assert stats["total_entries"] == 3
    assert stats["confidence_levels"]["high"] == 1
    assert stats["confidence_levels"]["medium"] == 1
    assert stats["confidence_levels"]["low"] == 1
    assert stats["unique_cases"] == 3
    assert "justice" in stats["tag_counts"]
    assert stats["tag_counts"]["justice"] == 2

def test_export_import_memory(scroll_memory, tempfile):
    """Test exporting and importing memory"""
    # Add memory entries
    scroll_memory.add_memory_entry(
        case_id=SAMPLE_CASE_ID,
        principle=SAMPLE_PRINCIPLE,
        scroll_alignment=SAMPLE_SCROLL_ALIGNMENT,
        prophetic_insight=SAMPLE_PROPHETIC_INSIGHT,
        confidence=0.9
    )
    
    # Export memory
    export_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
    success = scroll_memory.export_memory(export_file)
    assert success is True
    
    # Create a new memory instance
    new_memory = ScrollMemoryIntelligence(memory_file=tempfile.NamedTemporaryFile(delete=False, suffix='.json').name)
    
    # Import memory
    success = new_memory.import_memory(export_file)
    assert success is True
    
    # Check imported data
    assert len(new_memory.memory["entries"]) == 1
    assert new_memory.memory["entries"][0]["case_id"] == SAMPLE_CASE_ID
    
    # Clean up
    os.unlink(export_file)
    os.unlink(new_memory.memory_file) 
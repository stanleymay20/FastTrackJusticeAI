import os
import json
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from backend.app.utils.legal_precedent import LegalPrecedentManager

@pytest.fixture
def sample_precedent():
    return {
        'title': 'Test Case v. State',
        'court': 'Supreme Court',
        'year': '2023',
        'citation': '123 S.Ct. 456',
        'summary': 'A test case about constitutional rights',
        'facts': 'The plaintiff challenged a state law',
        'holding': 'The court established a new principle of law',
        'reasoning': 'Because the law violated fundamental rights',
        'jurisdiction': 'Federal',
        'category': 'Constitutional Law'
    }

@pytest.fixture
def manager(tmp_path):
    # Create a temporary precedents file
    precedents_file = tmp_path / "test_precedents.json"
    return LegalPrecedentManager(precedents_file=str(precedents_file))

def test_initialization(manager):
    """Test initialization of LegalPrecedentManager"""
    assert manager.precedents == []
    assert manager.index is None
    assert manager.dimension > 0  # BERT model dimension should be positive

def test_load_precedents_empty(manager):
    """Test loading from empty file"""
    manager.load_precedents()
    assert manager.precedents == []
    assert manager.index is None

def test_add_precedent(manager, sample_precedent):
    """Test adding a new precedent"""
    success = manager.add_precedent(sample_precedent)
    assert success
    assert len(manager.precedents) == 1
    assert manager.index is not None
    assert manager.index.ntotal == 1

def test_add_precedent_missing_fields(manager):
    """Test adding a precedent with missing required fields"""
    incomplete_precedent = {
        'title': 'Test Case',
        'court': 'Supreme Court'
    }
    success = manager.add_precedent(incomplete_precedent)
    assert not success
    assert len(manager.precedents) == 0

def test_find_relevant_precedents(manager, sample_precedent):
    """Test finding relevant precedents"""
    # Add a precedent
    manager.add_precedent(sample_precedent)
    
    # Search with similar query
    results = manager.find_relevant_precedents(
        "constitutional rights challenge state law",
        top_k=1
    )
    
    assert len(results) == 1
    assert results[0]['title'] == sample_precedent['title']
    assert 'similarity_score' in results[0]
    assert 0 <= results[0]['similarity_score'] <= 1

def test_find_relevant_precedents_empty(manager):
    """Test finding precedents with empty database"""
    results = manager.find_relevant_precedents("test query")
    assert results == []

def test_extract_legal_principles(manager, sample_precedent):
    """Test extracting legal principles from a precedent"""
    principles = manager.extract_legal_principles(sample_precedent)
    assert len(principles) > 0
    assert any('principle' in p.lower() for p in principles)
    assert any('because' in p.lower() for p in principles)

def test_format_citation(manager, sample_precedent):
    """Test formatting legal citation"""
    citation = manager.format_citation(sample_precedent)
    assert sample_precedent['title'] in citation
    assert sample_precedent['court'] in citation
    assert sample_precedent['year'] in citation
    assert sample_precedent['citation'] in citation

def test_get_statistics(manager, sample_precedent):
    """Test getting database statistics"""
    # Add a precedent
    manager.add_precedent(sample_precedent)
    
    stats = manager.get_statistics()
    assert stats['total_precedents'] == 1
    assert sample_precedent['court'] in stats['courts']
    assert sample_precedent['year'] in stats['years']
    assert stats['latest_addition'] is not None

@patch('sentence_transformers.SentenceTransformer')
def test_bert_model_initialization(mock_transformer, tmp_path):
    """Test BERT model initialization"""
    # Mock the model's dimension
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_transformer.return_value = mock_model
    
    precedents_file = tmp_path / "test_precedents.json"
    manager = LegalPrecedentManager(precedents_file=str(precedents_file))
    
    assert manager.dimension == 384
    mock_transformer.assert_called_once()

def test_save_and_load_precedents(manager, sample_precedent, tmp_path):
    """Test saving and loading precedents"""
    # Add a precedent
    manager.add_precedent(sample_precedent)
    
    # Create a new manager instance with the same file
    new_manager = LegalPrecedentManager(precedents_file=str(manager.precedents_file))
    
    # Load precedents
    new_manager.load_precedents()
    
    assert len(new_manager.precedents) == 1
    assert new_manager.precedents[0]['title'] == sample_precedent['title']
    assert new_manager.index is not None
    assert new_manager.index.ntotal == 1

def test_normalized_embeddings(manager, sample_precedent):
    """Test that embeddings are properly normalized"""
    # Add a precedent
    manager.add_precedent(sample_precedent)
    
    # Get the embedding from the index
    embedding = manager.index.reconstruct(0)
    
    # Check if the embedding is normalized
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-6  # Should be close to 1

def test_similarity_score_range(manager, sample_precedent):
    """Test that similarity scores are in the correct range"""
    # Add a precedent
    manager.add_precedent(sample_precedent)
    
    # Search with a query
    results = manager.find_relevant_precedents("test query")
    
    # Check similarity score range
    assert 0 <= results[0]['similarity_score'] <= 1 
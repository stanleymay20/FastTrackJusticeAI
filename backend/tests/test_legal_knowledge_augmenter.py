import pytest
from unittest.mock import MagicMock, patch
from backend.app.utils.legal_knowledge_augmenter import LegalKnowledgeAugmenter
from backend.app.utils.legal_precedent import LegalPrecedentManager

@pytest.fixture
def mock_precedent_manager():
    """Create a mock LegalPrecedentManager"""
    manager = MagicMock(spec=LegalPrecedentManager)
    manager.precedents = []
    return manager

@pytest.fixture
def sample_precedents():
    """Sample precedents for testing"""
    return [
        {
            'title': 'Test Case v. State',
            'court': 'Supreme Court',
            'year': '2023',
            'citation': '123 S.Ct. 456',
            'summary': 'A test case about constitutional rights',
            'facts': 'The plaintiff challenged a state law',
            'holding': 'The court established a new principle of law',
            'reasoning': 'Because the law violated fundamental rights',
            'jurisdiction': 'Federal',
            'category': 'Constitutional Law',
            'similarity_score': 0.85
        },
        {
            'title': 'Another Case v. City',
            'court': 'Circuit Court',
            'year': '2022',
            'citation': '456 F.3d 789',
            'summary': 'A case about property rights',
            'facts': 'The defendant challenged a zoning ordinance',
            'holding': 'The court found the ordinance unconstitutional',
            'reasoning': 'Because it violated property rights',
            'jurisdiction': 'State',
            'category': 'Property Law',
            'similarity_score': 0.75
        }
    ]

@pytest.fixture
def augmenter(mock_precedent_manager):
    """Create a LegalKnowledgeAugmenter instance"""
    return LegalKnowledgeAugmenter(
        precedent_manager=mock_precedent_manager,
        max_precedents=2,
        min_similarity_score=0.7,
        include_citations=True,
        include_principles=True
    )

def test_initialization(augmenter):
    """Test initialization of LegalKnowledgeAugmenter"""
    assert augmenter.max_precedents == 2
    assert augmenter.min_similarity_score == 0.7
    assert augmenter.include_citations is True
    assert augmenter.include_principles is True

def test_augment_with_precedents_no_results(augmenter, mock_precedent_manager):
    """Test augmentation with no relevant precedents"""
    # Mock find_relevant_precedents to return empty list
    mock_precedent_manager.find_relevant_precedents.return_value = []
    
    result = augmenter.augment_with_precedents("test query")
    
    assert result["precedent_text"] == ""
    assert result["citations"] == []
    assert result["principles"] == []
    mock_precedent_manager.find_relevant_precedents.assert_called_once_with(
        "test query",
        top_k=2
    )

def test_augment_with_precedents_with_results(augmenter, mock_precedent_manager, sample_precedents):
    """Test augmentation with relevant precedents"""
    # Mock find_relevant_precedents to return sample precedents
    mock_precedent_manager.find_relevant_precedents.return_value = sample_precedents
    
    # Mock format_citation
    mock_precedent_manager.format_citation.side_effect = [
        "Test Case v. State, 123 S.Ct. 456 (2023)",
        "Another Case v. City, 456 F.3d 789 (2022)"
    ]
    
    # Mock extract_legal_principles
    mock_precedent_manager.extract_legal_principles.side_effect = [
        ["Principle 1", "Principle 2"],
        ["Principle 3", "Principle 4"]
    ]
    
    result = augmenter.augment_with_precedents("constitutional rights")
    
    # Check that the result contains the expected content
    assert "Test Case v. State" in result["precedent_text"]
    assert "Another Case v. City" in result["precedent_text"]
    assert len(result["citations"]) == 2
    assert len(result["principles"]) == 4  # All principles should be included
    
    # Verify that the mocks were called correctly
    mock_precedent_manager.find_relevant_precedents.assert_called_once_with(
        "constitutional rights",
        top_k=2
    )
    assert mock_precedent_manager.format_citation.call_count == 2
    assert mock_precedent_manager.extract_legal_principles.call_count == 2

def test_augment_with_precedents_filter_by_jurisdiction(augmenter, mock_precedent_manager, sample_precedents):
    """Test augmentation with jurisdiction filter"""
    # Mock find_relevant_precedents to return sample precedents
    mock_precedent_manager.find_relevant_precedents.return_value = sample_precedents
    
    # Mock format_citation and extract_legal_principles
    mock_precedent_manager.format_citation.return_value = "Test Case v. State, 123 S.Ct. 456 (2023)"
    mock_precedent_manager.extract_legal_principles.return_value = ["Principle 1", "Principle 2"]
    
    result = augmenter.augment_with_precedents("constitutional rights", jurisdiction="Federal")
    
    # Only the Federal jurisdiction precedent should be included
    assert "Test Case v. State" in result["precedent_text"]
    assert "Another Case v. City" not in result["precedent_text"]
    assert len(result["citations"]) == 1
    assert len(result["principles"]) == 2

def test_augment_with_precedents_filter_by_similarity(augmenter, mock_precedent_manager):
    """Test augmentation with similarity score filter"""
    # Create precedents with different similarity scores
    precedents = [
        {
            'title': 'High Similarity Case',
            'court': 'Supreme Court',
            'year': '2023',
            'citation': '123 S.Ct. 456',
            'summary': 'A test case',
            'similarity_score': 0.85
        },
        {
            'title': 'Low Similarity Case',
            'court': 'Circuit Court',
            'year': '2022',
            'citation': '456 F.3d 789',
            'summary': 'Another test case',
            'similarity_score': 0.65  # Below the threshold
        }
    ]
    
    # Mock find_relevant_precedents to return the precedents
    mock_precedent_manager.find_relevant_precedents.return_value = precedents
    
    # Mock format_citation and extract_legal_principles
    mock_precedent_manager.format_citation.return_value = "High Similarity Case, 123 S.Ct. 456 (2023)"
    mock_precedent_manager.extract_legal_principles.return_value = ["Principle 1"]
    
    result = augmenter.augment_with_precedents("test query")
    
    # Only the high similarity precedent should be included
    assert "High Similarity Case" in result["precedent_text"]
    assert "Low Similarity Case" not in result["precedent_text"]
    assert len(result["citations"]) == 1
    assert len(result["principles"]) == 1

def test_format_for_prompt(augmenter, mock_precedent_manager, sample_precedents):
    """Test formatting for prompt"""
    # Mock augment_with_precedents to return a result
    with patch.object(augmenter, 'augment_with_precedents') as mock_augment:
        mock_augment.return_value = {
            "precedent_text": "Case: Test Case\nCourt: Supreme Court\nSummary: A test case",
            "citations": ["Test Case v. State, 123 S.Ct. 456 (2023)"],
            "principles": ["Principle 1", "Principle 2"]
        }
        
        result = augmenter.format_for_prompt("test query")
        
        # Check that the result contains the expected sections
        assert "RELEVANT LEGAL PRECEDENTS:" in result
        assert "LEGAL CITATIONS:" in result
        assert "KEY LEGAL PRINCIPLES:" in result
        assert "Test Case v. State" in result
        assert "Principle 1" in result
        assert "Principle 2" in result
        
        # Verify that augment_with_precedents was called correctly
        mock_augment.assert_called_once_with("test query", None, None)

def test_format_for_prompt_no_precedents(augmenter, mock_precedent_manager):
    """Test formatting for prompt with no precedents"""
    # Mock augment_with_precedents to return empty result
    with patch.object(augmenter, 'augment_with_precedents') as mock_augment:
        mock_augment.return_value = {
            "precedent_text": "",
            "citations": [],
            "principles": []
        }
        
        result = augmenter.format_for_prompt("test query")
        
        # Result should be an empty string
        assert result == ""
        
        # Verify that augment_with_precedents was called correctly
        mock_augment.assert_called_once_with("test query", None, None)

def test_get_augmentation_stats(augmenter, mock_precedent_manager):
    """Test getting augmentation statistics"""
    # Set the number of precedents in the manager
    mock_precedent_manager.precedents = [1, 2, 3]  # 3 precedents
    
    stats = augmenter.get_augmentation_stats()
    
    assert stats["total_precedents"] == 3
    assert stats["max_precedents"] == 2
    assert stats["min_similarity_score"] == 0.7
    assert stats["include_citations"] is True
    assert stats["include_principles"] is True 
import pytest
from app.utils.scroll_judgment import generate_scroll_aligned_judgment

def test_generate_scroll_aligned_judgment():
    """Test basic functionality of judgment generation."""
    # Sample criminal case text
    case_text = """
    The defendant is charged with second-degree assault. On the night of June 15, 2023,
    the defendant allegedly struck the victim with a baseball bat, causing serious injury.
    Witnesses testified that they saw the defendant approach the victim from behind and
    strike them without provocation. The defendant claims they acted in self-defense,
    but no evidence supports this claim.
    """
    
    # Sample classification results
    classification = {
        "category": "Criminal",
        "confidence": 0.85,
        "subcategories": ["Assault", "Violent Crime"],
        "subcategory_confidences": [0.9, 0.8]
    }
    
    # Generate judgment
    result = generate_scroll_aligned_judgment(case_text, classification)
    
    # Check result structure
    assert isinstance(result, dict)
    assert "judgment_text" in result
    assert "metadata" in result
    assert isinstance(result["judgment_text"], str)
    assert isinstance(result["metadata"], dict)
    
    # Check metadata
    metadata = result["metadata"]
    assert "scroll_phase" in metadata
    assert "case_category" in metadata
    assert "verdict" in metadata
    assert "confidence" in metadata
    
    # Check specific values
    assert metadata["case_category"] == "Criminal"
    assert metadata["verdict"] == "Guilty"  # High confidence should result in guilty verdict
    assert metadata["confidence"] == 0.85
    
    # Check judgment text content
    assert "Guilty" in result["judgment_text"]
    assert any(phase in result["judgment_text"] for phase in ["dawn", "noon", "dusk", "night"])

def test_judgment_thematic_consistency():
    """Test that generated judgments are thematically consistent with scroll phases."""
    # Sample civil case text
    case_text = """
    The plaintiff alleges breach of contract against the defendant. The parties entered
    into a written agreement for the defendant to provide consulting services. The
    defendant failed to complete the agreed-upon work and refused to refund the advance
    payment. The plaintiff seeks damages in the amount of $50,000.
    """
    
    # Sample classification results
    classification = {
        "category": "Civil",
        "confidence": 0.65,
        "subcategories": ["Contract Law", "Business Dispute"],
        "subcategory_confidences": [0.7, 0.6]
    }
    
    # Generate judgment
    result = generate_scroll_aligned_judgment(case_text, classification)
    
    # Check thematic consistency
    judgment_text = result["judgment_text"].lower()
    scroll_phase = result["metadata"]["scroll_phase"]
    
    # Verify phase-specific themes are present
    if scroll_phase == "dawn":
        assert any(theme in judgment_text for theme in ["dawn", "morning", "rising", "first light"])
    elif scroll_phase == "noon":
        assert any(theme in judgment_text for theme in ["noon", "midday", "height of day", "peak"])
    elif scroll_phase == "dusk":
        assert any(theme in judgment_text for theme in ["dusk", "setting", "evening", "waning"])
    else:  # night
        assert any(theme in judgment_text for theme in ["night", "darkness", "shadows", "veiled"])

def test_judgment_category_consistency():
    """Test that judgments are consistent with different case categories."""
    # Test cases for each category
    test_cases = [
        {
            "category": "Criminal",
            "text": "The defendant is charged with theft of property valued at $5,000.",
            "confidence": 0.9,
            "expected_verdict": "Guilty"
        },
        {
            "category": "Civil",
            "text": "The plaintiff seeks damages for breach of contract.",
            "confidence": 0.3,
            "expected_verdict": "Not Liable"
        },
        {
            "category": "Family",
            "text": "The petitioner seeks modification of child custody arrangements.",
            "confidence": 0.8,
            "expected_verdict": "Granted"
        },
        {
            "category": "Administrative",
            "text": "The applicant seeks a building permit for commercial construction.",
            "confidence": 0.4,
            "expected_verdict": "Denied"
        }
    ]
    
    for test_case in test_cases:
        classification = {
            "category": test_case["category"],
            "confidence": test_case["confidence"],
            "subcategories": ["Test"],
            "subcategory_confidences": [0.5]
        }
        
        result = generate_scroll_aligned_judgment(test_case["text"], classification)
        
        # Check category-specific elements
        assert result["metadata"]["case_category"] == test_case["category"]
        assert result["metadata"]["verdict"] == test_case["expected_verdict"]
        
        # Check category-specific terminology
        judgment_text = result["judgment_text"].lower()
        if test_case["category"] == "Criminal":
            assert "defendant" in judgment_text
        elif test_case["category"] == "Civil":
            assert "defendant" in judgment_text and "plaintiff" in judgment_text
        elif test_case["category"] == "Family":
            assert any(term in judgment_text for term in ["custody", "child", "family"])
        else:  # Administrative
            assert "application" in judgment_text 
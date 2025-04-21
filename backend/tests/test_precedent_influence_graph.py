import os
import json
import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime
from backend.app.utils.precedent_influence_graph import PrecedentInfluenceGraph
from backend.app.utils.legal_principle_synthesizer import LegalPrincipleSynthesizer

# Sample case data for testing
SAMPLE_CASES = [
    {
        "id": "case_001",
        "text": """
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
        """,
        "holding": """
        The court holds that the defendant's failure to deliver goods within the specified time
        frame constitutes a material breach of the contract, excusing the plaintiff from further
        performance and entitling the plaintiff to damages.
        """,
        "reasoning": """
        The court reasons that time was of the essence in this agreement, as evidenced by the
        explicit language in the contract and the plaintiff's need for timely delivery to fulfill
        obligations to third parties. The court applies the established principle that a material
        breach of contract excuses the non-breaching party from further performance and allows for
        recovery of damages. The court finds that the defendant's actions were not excused by any
        force majeure clause or other provision in the contract.
        """,
        "metadata": {
            "date": "2020-01-15",
            "court": "Supreme Court",
            "jurisdiction": "New York"
        }
    },
    {
        "id": "case_002",
        "text": """
        In Brown v. Wilson, the plaintiff alleges that the defendant's actions constituted negligence
        in the maintenance of their property, resulting in injury to the plaintiff. The court must
        determine whether the defendant breached their duty of care.

        The court finds that the defendant failed to maintain their property in a reasonably safe
        condition, which constituted a breach of the duty of care owed to invitees. The court holds
        that the plaintiff's injury was a direct result of this breach.

        The court applies the principle that property owners owe a duty of care to invitees to
        maintain their property in a reasonably safe condition. The court recognizes that the
        defendant had actual or constructive notice of the dangerous condition but failed to address it.
        """,
        "holding": """
        The court holds that the defendant's failure to maintain their property in a reasonably safe
        condition constituted negligence, and the plaintiff is entitled to damages for their injuries.
        """,
        "reasoning": """
        The court reasons that property owners owe a duty of care to invitees, which includes
        maintaining their property in a reasonably safe condition. The court finds that the defendant
        had actual or constructive notice of the dangerous condition but failed to address it within
        a reasonable time. The court applies the established principle that a breach of this duty
        that proximately causes injury to an invitee results in liability for negligence.
        """,
        "metadata": {
            "date": "2021-03-22",
            "court": "Appellate Court",
            "jurisdiction": "California"
        }
    },
    {
        "id": "case_003",
        "text": """
        In Johnson v. State, the defendant challenges their conviction on the grounds that evidence
        was obtained in violation of their Fourth Amendment rights. The court must determine whether
        the search and seizure was conducted without a valid warrant and without probable cause.

        The court finds that the search was conducted without a valid warrant and without probable
        cause, violating the defendant's Fourth Amendment rights. The court holds that the evidence
        obtained as a result of this unconstitutional search must be excluded from trial.

        The court applies the principle that evidence obtained in violation of the Fourth Amendment
        is inadmissible in criminal proceedings, known as the exclusionary rule. The court recognizes
        that this rule serves to deter law enforcement from conducting unconstitutional searches and
        seizures.
        """,
        "holding": """
        The court holds that the search was conducted in violation of the defendant's Fourth Amendment
        rights, and the evidence obtained must be excluded from trial.
        """,
        "reasoning": """
        The court reasons that the Fourth Amendment protects individuals from unreasonable searches
        and seizures. The court finds that the search was conducted without a valid warrant and
        without probable cause, which violates this constitutional protection. The court applies the
        established principle that evidence obtained in violation of the Fourth Amendment is
        inadmissible in criminal proceedings, known as the exclusionary rule. This rule serves to
        deter law enforcement from conducting unconstitutional searches and seizures.
        """,
        "metadata": {
            "date": "2022-06-10",
            "court": "Federal Court",
            "jurisdiction": "Federal"
        }
    }
]

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_synthesizer():
    """Create a mock LegalPrincipleSynthesizer"""
    with patch("backend.app.utils.legal_principle_synthesizer.LegalPrincipleSynthesizer") as mock:
        synthesizer_instance = MagicMock()
        
        # Mock the process_case method
        def mock_process_case(case_text, holding=None, reasoning=None, cache_key=None):
            # Return different results based on the case text
            if "contract" in case_text.lower():
                return {
                    "raw_principles": [
                        "A material breach of contract excuses the non-breaching party from further performance.",
                        "Time was of the essence in this agreement."
                    ],
                    "summarized_principles": [
                        "Material breach excuses non-breaching party from performance.",
                        "Time was of the essence in the agreement."
                    ],
                    "classified_principles": [
                        {
                            "principle": "Material breach excuses non-breaching party from performance.",
                            "categories": ["contract", "civil", "procedural"],
                            "scores": [0.9, 0.7, 0.5]
                        },
                        {
                            "principle": "Time was of the essence in the agreement.",
                            "categories": ["contract", "civil", "other"],
                            "scores": [0.8, 0.6, 0.4]
                        }
                    ],
                    "metadata": {
                        "num_raw_principles": 2,
                        "num_summarized_principles": 2,
                        "categories_found": ["contract", "civil", "procedural", "other"]
                    }
                }
            elif "negligence" in case_text.lower():
                return {
                    "raw_principles": [
                        "Property owners owe a duty of care to invitees.",
                        "A breach of the duty of care that proximately causes injury results in liability."
                    ],
                    "summarized_principles": [
                        "Property owners owe duty of care to invitees.",
                        "Breach of duty causing injury results in liability."
                    ],
                    "classified_principles": [
                        {
                            "principle": "Property owners owe duty of care to invitees.",
                            "categories": ["tort", "civil", "property"],
                            "scores": [0.9, 0.7, 0.6]
                        },
                        {
                            "principle": "Breach of duty causing injury results in liability.",
                            "categories": ["tort", "civil", "other"],
                            "scores": [0.8, 0.7, 0.4]
                        }
                    ],
                    "metadata": {
                        "num_raw_principles": 2,
                        "num_summarized_principles": 2,
                        "categories_found": ["tort", "civil", "property", "other"]
                    }
                }
            else:
                return {
                    "raw_principles": [
                        "Evidence obtained in violation of the Fourth Amendment is inadmissible.",
                        "The exclusionary rule deters unconstitutional searches and seizures."
                    ],
                    "summarized_principles": [
                        "Evidence from Fourth Amendment violations is inadmissible.",
                        "Exclusionary rule deters unconstitutional searches."
                    ],
                    "classified_principles": [
                        {
                            "principle": "Evidence from Fourth Amendment violations is inadmissible.",
                            "categories": ["constitutional", "criminal", "evidence"],
                            "scores": [0.9, 0.8, 0.7]
                        },
                        {
                            "principle": "Exclusionary rule deters unconstitutional searches.",
                            "categories": ["constitutional", "criminal", "procedural"],
                            "scores": [0.8, 0.7, 0.6]
                        }
                    ],
                    "metadata": {
                        "num_raw_principles": 2,
                        "num_summarized_principles": 2,
                        "categories_found": ["constitutional", "criminal", "evidence", "procedural"]
                    }
                }
                
        synthesizer_instance.process_case = mock_process_case
        
        # Mock the extraction_model
        extraction_model = MagicMock()
        
        # Mock the encode method
        def mock_encode(texts, **kwargs):
            # Create random embeddings for testing
            if isinstance(texts, str):
                return np.random.rand(384)  # Typical BERT embedding size
            else:
                return np.array([np.random.rand(384) for _ in texts])
                
        extraction_model.encode = mock_encode
        synthesizer_instance.extraction_model = extraction_model
        
        mock.return_value = synthesizer_instance
        yield mock

@pytest.fixture
def graph(temp_cache_dir, temp_output_dir, mock_synthesizer):
    """Create a PrecedentInfluenceGraph instance with mocked dependencies"""
    with patch("backend.app.utils.precedent_influence_graph.LegalPrincipleSynthesizer", mock_synthesizer):
        graph = PrecedentInfluenceGraph(
            cache_dir=temp_cache_dir,
            output_dir=temp_output_dir,
            similarity_threshold=0.5,  # Lower threshold for testing
            max_principles_per_case=5
        )
        return graph

def test_initialization(graph, temp_cache_dir, temp_output_dir):
    """Test initialization of PrecedentInfluenceGraph"""
    assert graph.cache_dir == temp_cache_dir
    assert graph.output_dir == temp_output_dir
    assert graph.similarity_threshold == 0.5
    assert graph.max_principles_per_case == 5
    assert graph.layout_algorithm == "spring"
    assert isinstance(graph.graph, type(graph.graph))  # Check it's a DiGraph
    assert len(graph.case_data) == 0
    assert os.path.exists(temp_cache_dir)
    assert os.path.exists(temp_output_dir)

def test_add_case(graph, mock_synthesizer):
    """Test adding a single case to the graph"""
    case = SAMPLE_CASES[0]
    result = graph.add_case(
        case_id=case["id"],
        case_text=case["text"],
        holding=case["holding"],
        reasoning=case["reasoning"],
        metadata=case["metadata"]
    )
    
    # Check result
    assert result["id"] == case["id"]
    assert "principles" in result
    assert "summarized_principles" in result
    assert "classified_principles" in result
    assert "metadata" in result
    assert "categories" in result
    
    # Check graph
    assert case["id"] in graph.graph.nodes
    assert len(graph.graph.nodes) == 1
    assert len(graph.graph.edges) == 0  # No edges yet with only one case
    
    # Check case data
    assert case["id"] in graph.case_data
    assert graph.case_data[case["id"]]["id"] == case["id"]

def test_add_cases(graph, mock_synthesizer):
    """Test adding multiple cases to the graph"""
    results = graph.add_cases(SAMPLE_CASES)
    
    # Check results
    assert len(results) == len(SAMPLE_CASES)
    assert all(result["id"] in [case["id"] for case in SAMPLE_CASES] for result in results)
    
    # Check graph
    assert len(graph.graph.nodes) == len(SAMPLE_CASES)
    assert all(case["id"] in graph.graph.nodes for case in SAMPLE_CASES)
    
    # Check case data
    assert len(graph.case_data) == len(SAMPLE_CASES)
    assert all(case["id"] in graph.case_data for case in SAMPLE_CASES)

def test_calculate_case_similarity(graph):
    """Test calculating similarity between cases"""
    # Add cases to the graph
    graph.add_cases(SAMPLE_CASES)
    
    # Get principles from the first two cases
    principles1 = graph.case_data[SAMPLE_CASES[0]["id"]]["summarized_principles"]
    principles2 = graph.case_data[SAMPLE_CASES[1]["id"]]["summarized_principles"]
    
    # Calculate similarity
    similarity = graph._calculate_case_similarity(principles1, principles2)
    
    # Check result
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1
    
    # Test with empty principles
    empty_similarity = graph._calculate_case_similarity([], principles2)
    assert empty_similarity == 0.0

def test_get_shared_principles(graph):
    """Test finding shared principles between cases"""
    # Add cases to the graph
    graph.add_cases(SAMPLE_CASES)
    
    # Get principles from the first two cases
    principles1 = graph.case_data[SAMPLE_CASES[0]["id"]]["summarized_principles"]
    principles2 = graph.case_data[SAMPLE_CASES[1]["id"]]["summarized_principles"]
    
    # Find shared principles
    shared = graph._get_shared_principles(principles1, principles2)
    
    # Check result
    assert isinstance(shared, list)
    assert all(isinstance(p, str) for p in shared)
    
    # Test with empty principles
    empty_shared = graph._get_shared_principles([], principles2)
    assert empty_shared == []

def test_determine_influence_direction(graph):
    """Test determining influence direction between cases"""
    # Add cases to the graph
    graph.add_cases(SAMPLE_CASES)
    
    # Get case data
    case1 = graph.case_data[SAMPLE_CASES[0]["id"]]
    case2 = graph.case_data[SAMPLE_CASES[1]["id"]]
    
    # Determine direction
    direction = graph._determine_influence_direction(case1, case2)
    
    # Check result
    assert direction in ["forward", "backward"]
    
    # Test with cases without dates
    case1_no_date = {**case1, "metadata": {}}
    case2_no_date = {**case2, "metadata": {}}
    
    direction_no_date = graph._determine_influence_direction(case1_no_date, case2_no_date)
    assert direction_no_date == "forward"  # Default direction

def test_extract_date(graph):
    """Test extracting date from metadata"""
    # Test with different date formats
    metadata_formats = [
        {"date": "2020-01-15"},
        {"date": "2020/01/15"},
        {"date": "15-01-2020"},
        {"date": "15/01/2020"},
        {"date": "January 15, 2020"},
        {"date": "15 January 2020"},
        {"date": 1579046400},  # Unix timestamp
        {"decision_date": "2020-01-15"},
        {"issued_date": "2020-01-15"},
        {"filed_date": "2020-01-15"},
        {"no_date": "not a date"}
    ]
    
    for metadata in metadata_formats:
        date = graph._extract_date(metadata)
        if "no_date" in metadata:
            assert date is None
        else:
            assert isinstance(date, datetime) or date is None

def test_get_case_influences(graph):
    """Test getting case influences"""
    # Add cases to the graph
    graph.add_cases(SAMPLE_CASES)
    
    # Get influences for the first case
    influences = graph.get_case_influences(SAMPLE_CASES[0]["id"])
    
    # Check result
    assert "influencing" in influences
    assert "influenced" in influences
    assert isinstance(influences["influencing"], list)
    assert isinstance(influences["influenced"], list)
    
    # Test with non-existent case
    non_existent = graph.get_case_influences("non_existent")
    assert non_existent["influencing"] == []
    assert non_existent["influenced"] == []

def test_get_principle_evolution(graph):
    """Test tracking principle evolution"""
    # Add cases to the graph
    graph.add_cases(SAMPLE_CASES)
    
    # Track evolution of a principle
    principle = "Material breach excuses non-breaching party from performance."
    evolution = graph.get_principle_evolution(principle)
    
    # Check result
    assert isinstance(evolution, list)
    assert all(isinstance(case, dict) for case in evolution)
    assert all("id" in case for case in evolution)
    assert all("similarity" in case for case in evolution)
    assert all("date" in case for case in evolution)
    assert all("principle" in case for case in evolution)
    assert all("metadata" in case for case in evolution)

def test_visualize(graph, temp_output_dir):
    """Test graph visualization"""
    # Add cases to the graph
    graph.add_cases(SAMPLE_CASES)
    
    # Generate visualization
    output_file = os.path.join(temp_output_dir, "test_graph.png")
    result = graph.visualize(
        output_file=output_file,
        title="Test Graph",
        node_size=500,
        edge_width_scale=1.0,
        show_labels=True,
        highlight_cases=[SAMPLE_CASES[0]["id"]],
        color_by_category=False
    )
    
    # Check result
    assert result == output_file
    assert os.path.exists(output_file)
    
    # Test with empty graph
    empty_graph = PrecedentInfluenceGraph(
        cache_dir=temp_output_dir,
        output_dir=temp_output_dir
    )
    empty_result = empty_graph.visualize()
    assert empty_result == ""

def test_export_graph_data(graph, temp_output_dir):
    """Test exporting graph data"""
    # Add cases to the graph
    graph.add_cases(SAMPLE_CASES)
    
    # Export data
    output_file = os.path.join(temp_output_dir, "test_graph_data.json")
    result = graph.export_graph_data(output_file=output_file)
    
    # Check result
    assert result == output_file
    assert os.path.exists(output_file)
    
    # Check file content
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == len(SAMPLE_CASES)

def test_get_graph_statistics(graph):
    """Test getting graph statistics"""
    # Test with empty graph
    empty_stats = graph.get_graph_statistics()
    assert empty_stats["num_cases"] == 0
    assert empty_stats["num_edges"] == 0
    assert empty_stats["density"] == 0.0
    assert empty_stats["avg_degree"] == 0.0
    assert empty_stats["categories"] == {}
    
    # Add cases to the graph
    graph.add_cases(SAMPLE_CASES)
    
    # Get statistics
    stats = graph.get_graph_statistics()
    
    # Check result
    assert stats["num_cases"] == len(SAMPLE_CASES)
    assert "num_edges" in stats
    assert "density" in stats
    assert "avg_degree" in stats
    assert "categories" in stats
    assert "top_influential_cases" in stats

def test_save_and_load(graph, temp_cache_dir):
    """Test saving and loading the graph"""
    # Add cases to the graph
    graph.add_cases(SAMPLE_CASES)
    
    # Save graph
    save_file = os.path.join(temp_cache_dir, "test_save.json")
    save_result = graph.save(filename=save_file)
    
    # Check result
    assert save_result == save_file
    assert os.path.exists(save_file)
    
    # Load graph
    loaded_graph = PrecedentInfluenceGraph.load(save_file)
    
    # Check loaded graph
    assert loaded_graph.similarity_threshold == graph.similarity_threshold
    assert loaded_graph.max_principles_per_case == graph.max_principles_per_case
    assert loaded_graph.layout_algorithm == graph.layout_algorithm
    assert len(loaded_graph.graph.nodes) == len(graph.graph.nodes)
    assert len(loaded_graph.case_data) == len(graph.case_data)
    assert all(case_id in loaded_graph.graph.nodes for case_id in graph.graph.nodes) 
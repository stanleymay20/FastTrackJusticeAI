import os
import pytest
import datetime
from unittest.mock import patch, MagicMock
from app.utils.scroll_forecast_generator import ScrollForecastGenerator

@pytest.fixture
def generator():
    """Create a ScrollForecastGenerator instance for testing."""
    return ScrollForecastGenerator(output_dir="test_reports")

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        'phase_data': {
            'Dawn': 25.0,
            'Noon': 30.0,
            'Dusk': 25.0,
            'Night': 20.0
        },
        'severity_data': {
            'Low': 10,
            'Medium': 15,
            'High': 8,
            'Critical': 3
        },
        'top_insights': [
            {
                'text': 'Test insight 1',
                'severity': 'high',
                'phase': 'dawn'
            },
            {
                'text': 'Test insight 2',
                'severity': 'medium',
                'phase': 'night'
            }
        ],
        'historical_data': [
            {
                'date': '2024-01-01',
                'dawn': 20,
                'noon': 25,
                'dusk': 30,
                'night': 25
            },
            {
                'date': '2024-01-02',
                'dawn': 22,
                'noon': 26,
                'dusk': 29,
                'night': 23
            }
        ],
        'prophetic_window': "Test prophetic window text"
    }

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """Setup and cleanup for tests."""
    # Create test directories
    os.makedirs("test_reports", exist_ok=True)
    os.makedirs("test_assets/fonts", exist_ok=True)
    
    yield
    
    # Cleanup
    import shutil
    if os.path.exists("test_reports"):
        shutil.rmtree("test_reports")
    if os.path.exists("test_assets"):
        shutil.rmtree("test_assets")

def test_initialization(generator):
    """Test ScrollForecastGenerator initialization."""
    assert generator.output_dir == "test_reports"
    assert os.path.exists("test_reports")
    assert hasattr(generator, 'styles')

def test_ensure_output_dir(generator):
    """Test output directory creation."""
    # Delete the directory to test creation
    import shutil
    if os.path.exists("test_reports"):
        shutil.rmtree("test_reports")
    
    generator._ensure_output_dir()
    assert os.path.exists("test_reports")

def test_register_fonts(generator):
    """Test font registration."""
    # Create a test font file
    os.makedirs("test_assets/fonts", exist_ok=True)
    with open("test_assets/fonts/test_font.ttf", "wb") as f:
        f.write(b"dummy font data")
    
    with patch('reportlab.pdfbase.pdfmetrics.registerFont') as mock_register:
        with patch('reportlab.pdfbase.ttfonts.TTFont') as mock_ttfont:
            generator._register_fonts()
            mock_ttfont.assert_called()
            mock_register.assert_called()

def test_define_custom_styles(generator):
    """Test custom style definitions."""
    generator._define_custom_styles()
    assert 'ScrollTitle' in generator.styles
    assert 'ScrollSubtitle' in generator.styles
    assert 'ScrollHeading' in generator.styles
    assert 'ScrollBody' in generator.styles
    assert 'ScrollInsight' in generator.styles

def test_generate_phase_chart(generator, sample_data):
    """Test phase chart generation."""
    chart_data = generator.generate_phase_chart(sample_data['phase_data'])
    assert isinstance(chart_data, bytes)
    assert len(chart_data) > 0

def test_generate_severity_chart(generator, sample_data):
    """Test severity chart generation."""
    chart_data = generator.generate_severity_chart(sample_data['severity_data'])
    assert isinstance(chart_data, bytes)
    assert len(chart_data) > 0

def test_generate_historical_chart(generator, sample_data):
    """Test historical chart generation."""
    chart_data = generator.generate_historical_chart(sample_data['historical_data'])
    assert isinstance(chart_data, bytes)
    assert len(chart_data) > 0

def test_generate_forecast_pdf(generator, sample_data):
    """Test PDF generation with sample data."""
    test_date = datetime.date(2024, 1, 1)
    
    pdf_path = generator.generate_forecast_pdf(
        date=test_date,
        phase_data=sample_data['phase_data'],
        severity_data=sample_data['severity_data'],
        top_insights=sample_data['top_insights'],
        historical_data=sample_data['historical_data'],
        prophetic_window=sample_data['prophetic_window']
    )
    
    assert os.path.exists(pdf_path)
    assert pdf_path.endswith('.pdf')
    assert os.path.getsize(pdf_path) > 0

def test_generate_forecast_pdf_default_values(generator):
    """Test PDF generation with default values."""
    pdf_path = generator.generate_forecast_pdf()
    assert os.path.exists(pdf_path)
    assert pdf_path.endswith('.pdf')
    assert os.path.getsize(pdf_path) > 0

def test_send_to_telegram_success(generator):
    """Test successful Telegram PDF delivery."""
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        
        # Create a dummy PDF file
        test_pdf = os.path.join(generator.output_dir, "test.pdf")
        with open(test_pdf, "wb") as f:
            f.write(b"dummy pdf content")
        
        result = generator.send_to_telegram(
            test_pdf,
            "test_bot_token",
            "test_chat_id"
        )
        
        assert result is True
        mock_post.assert_called_once()

def test_send_to_telegram_failure(generator):
    """Test failed Telegram PDF delivery."""
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad Request"
        
        # Create a dummy PDF file
        test_pdf = os.path.join(generator.output_dir, "test.pdf")
        with open(test_pdf, "wb") as f:
            f.write(b"dummy pdf content")
        
        result = generator.send_to_telegram(
            test_pdf,
            "test_bot_token",
            "test_chat_id"
        )
        
        assert result is False

def test_generate_forecast_pdf_with_missing_data(generator):
    """Test PDF generation with partially missing data."""
    test_date = datetime.date(2024, 1, 1)
    
    # Test with only required date parameter
    pdf_path = generator.generate_forecast_pdf(date=test_date)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0
    
    # Test with some optional parameters
    pdf_path = generator.generate_forecast_pdf(
        date=test_date,
        phase_data={'Dawn': 100.0},
        severity_data={'Low': 5}
    )
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0

def test_error_handling(generator):
    """Test error handling in various scenarios."""
    # Test with invalid date
    with pytest.raises(AttributeError):
        generator.generate_forecast_pdf(date="invalid_date")
    
    # Test with invalid phase data
    with pytest.raises(ValueError):
        generator.generate_phase_chart({'Invalid': 'data'})
    
    # Test with invalid severity data
    with pytest.raises(ValueError):
        generator.generate_severity_chart({'Invalid': 'data'})
    
    # Test with invalid historical data
    with pytest.raises(ValueError):
        generator.generate_historical_chart([{'invalid': 'data'}])

def test_pdf_content_structure(generator, sample_data):
    """Test the structure of the generated PDF content."""
    test_date = datetime.date(2024, 1, 1)
    
    # Generate PDF
    pdf_path = generator.generate_forecast_pdf(
        date=test_date,
        phase_data=sample_data['phase_data'],
        severity_data=sample_data['severity_data'],
        top_insights=sample_data['top_insights'],
        historical_data=sample_data['historical_data'],
        prophetic_window=sample_data['prophetic_window']
    )
    
    # Verify file exists and has content
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0
    
    # In a real implementation, you might want to use a PDF parsing library
    # to verify the actual content structure 
import os
import pytest
from unittest.mock import patch, mock_open
from app.utils.env_validator import EnvironmentValidator

@pytest.fixture
def validator():
    return EnvironmentValidator()

@pytest.fixture
def mock_env_file():
    return """
    # API Keys
    OPENAI_API_KEY=test_openai_key
    HUGGINGFACE_API_KEY=test_huggingface_key
    GOOGLE_API_KEY=test_google_key
    ELEVENLABS_API_KEY=test_elevenlabs_key

    # Authentication & Security
    NGROK_TOKEN=test_ngrok_token
    GITHUB_TOKEN=test_github_token
    BUFFER_RECOVERY_KEY=test_buffer_key

    # Telegram Configuration
    TELEGRAM_BOT_TOKEN=test_telegram_bot_token
    TELEGRAM_CHAT_ID=test_chat_id
    TELEGRAM_API_URL=test_api_url

    # Payment Processing
    STRIPE_SECRET_KEY=test_stripe_secret
    STRIPE_PUBLISHABLE_KEY=test_stripe_publishable
    PAYPAL_CLIENT_ID=test_paypal_client_id
    PAYPAL_CLIENT_SECRET=test_paypal_secret

    # Google OAuth
    GOOGLE_CLIENT_ID=test_google_client_id
    GOOGLE_CLIENT_SECRET=test_google_client_secret

    # Monitoring & Error Tracking
    SENTRY_DSN=test_sentry_dsn

    # Firebase Configuration
    FIREBASE_API_KEY=test_firebase_api_key
    FIREBASE_AUTH_DOMAIN=test_auth_domain
    FIREBASE_PROJECT_ID=test_project_id
    FIREBASE_STORAGE_BUCKET=test_storage_bucket
    FIREBASE_MESSAGING_SENDER_ID=test_sender_id
    FIREBASE_APP_ID=test_app_id
    """

def test_initialization():
    """Test validator initialization with default and custom env file."""
    # Test with default env file
    validator = EnvironmentValidator()
    assert validator.env_file == '.env'
    
    # Test with custom env file
    validator = EnvironmentValidator('.env.production')
    assert validator.env_file == '.env.production'

def test_load_environment_success(validator, mock_env_file):
    """Test successful environment loading."""
    with patch('dotenv.load_dotenv') as mock_load:
        with patch('os.path.exists', return_value=True):
            result = validator.load_environment()
            assert result is True
            mock_load.assert_called_once_with(validator.env_file)

def test_load_environment_failure(validator):
    """Test environment loading failure."""
    with patch('dotenv.load_dotenv', side_effect=Exception("Load failed")):
        result = validator.load_environment()
        assert result is False

def test_validate_environment_all_present(validator, mock_env_file):
    """Test environment validation with all variables present."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test',
        'HUGGINGFACE_API_KEY': 'test',
        'GOOGLE_API_KEY': 'test',
        'ELEVENLABS_API_KEY': 'test',
        'NGROK_TOKEN': 'test',
        'GITHUB_TOKEN': 'test',
        'BUFFER_RECOVERY_KEY': 'test',
        'TELEGRAM_BOT_TOKEN': 'test',
        'TELEGRAM_CHAT_ID': 'test',
        'TELEGRAM_API_URL': 'test',
        'STRIPE_SECRET_KEY': 'test',
        'STRIPE_PUBLISHABLE_KEY': 'test',
        'PAYPAL_CLIENT_ID': 'test',
        'PAYPAL_CLIENT_SECRET': 'test',
        'GOOGLE_CLIENT_ID': 'test',
        'GOOGLE_CLIENT_SECRET': 'test',
        'SENTRY_DSN': 'test',
        'FIREBASE_API_KEY': 'test',
        'FIREBASE_AUTH_DOMAIN': 'test',
        'FIREBASE_PROJECT_ID': 'test',
        'FIREBASE_STORAGE_BUCKET': 'test',
        'FIREBASE_MESSAGING_SENDER_ID': 'test',
        'FIREBASE_APP_ID': 'test'
    }):
        missing_vars = validator.validate_environment()
        assert not missing_vars

def test_validate_environment_missing_vars(validator):
    """Test environment validation with missing variables."""
    with patch.dict(os.environ, {}, clear=True):
        missing_vars = validator.validate_environment()
        assert missing_vars
        assert 'API Keys' in missing_vars
        assert 'OPENAI_API_KEY' in missing_vars['API Keys']

def test_check_env_file_tampering_windows(validator):
    """Test env file tampering check on Windows."""
    with patch('os.name', 'nt'):
        with patch('os.path.exists', return_value=True):
            with patch('win32security.GetFileSecurity') as mock_security:
                with patch('win32security.GetTokenInformation') as mock_token:
                    # Mock same owner SID
                    mock_security.return_value.GetSecurityDescriptorOwner.return_value = 'test_sid'
                    mock_token.return_value = ('test_sid',)
                    assert validator.check_env_file_tampering() is True
                    
                    # Mock different owner SID
                    mock_security.return_value.GetSecurityDescriptorOwner.return_value = 'different_sid'
                    assert validator.check_env_file_tampering() is False

def test_check_env_file_tampering_unix(validator):
    """Test env file tampering check on Unix-like systems."""
    with patch('os.name', 'posix'):
        with patch('os.path.exists', return_value=True):
            with patch('os.stat') as mock_stat:
                # Mock correct permissions (600)
                mock_stat.return_value.st_mode = 0o600
                assert validator.check_env_file_tampering() is True
                
                # Mock incorrect permissions
                mock_stat.return_value.st_mode = 0o777
                assert validator.check_env_file_tampering() is False

def test_validate_and_load_success(validator, mock_env_file):
    """Test successful validation and loading."""
    with patch('dotenv.load_dotenv'):
        with patch('os.path.exists', return_value=True):
            with patch('win32security.GetFileSecurity'):
                with patch('win32security.GetTokenInformation'):
                    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}):
                        assert validator.validate_and_load() is True

def test_validate_and_load_failure_missing_file(validator):
    """Test validation and loading failure due to missing file."""
    with patch('os.path.exists', return_value=False):
        assert validator.validate_and_load() is False

def test_validate_and_load_failure_tampering(validator):
    """Test validation and loading failure due to file tampering."""
    with patch('os.path.exists', return_value=True):
        with patch('win32security.GetFileSecurity'):
            with patch('win32security.GetTokenInformation', return_value=('different_sid',)):
                assert validator.validate_and_load() is False

def test_validate_and_load_failure_missing_vars(validator):
    """Test validation and loading failure due to missing variables."""
    with patch('dotenv.load_dotenv'):
        with patch('os.path.exists', return_value=True):
            with patch('win32security.GetFileSecurity'):
                with patch('win32security.GetTokenInformation'):
                    with patch.dict(os.environ, {}, clear=True):
                        assert validator.validate_and_load() is False 
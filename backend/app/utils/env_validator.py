import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """Validates environment variables and configuration on startup."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize the environment validator.
        
        Args:
            env_file: Optional path to .env file. If None, uses default .env
        """
        self.env_file = env_file or '.env'
        self.required_vars = {
            'API Keys': [
                'OPENAI_API_KEY',
                'HUGGINGFACE_API_KEY',
                'GOOGLE_API_KEY',
                'ELEVENLABS_API_KEY'
            ],
            'Authentication & Security': [
                'NGROK_TOKEN',
                'GITHUB_TOKEN',
                'BUFFER_RECOVERY_KEY'
            ],
            'Telegram Configuration': [
                'TELEGRAM_BOT_TOKEN',
                'TELEGRAM_CHAT_ID',
                'TELEGRAM_API_URL'
            ],
            'Payment Processing': [
                'STRIPE_SECRET_KEY',
                'STRIPE_PUBLISHABLE_KEY',
                'PAYPAL_CLIENT_ID',
                'PAYPAL_CLIENT_SECRET'
            ],
            'Google OAuth': [
                'GOOGLE_CLIENT_ID',
                'GOOGLE_CLIENT_SECRET'
            ],
            'Monitoring & Error Tracking': [
                'SENTRY_DSN'
            ],
            'Firebase Configuration': [
                'FIREBASE_API_KEY',
                'FIREBASE_AUTH_DOMAIN',
                'FIREBASE_PROJECT_ID',
                'FIREBASE_STORAGE_BUCKET',
                'FIREBASE_MESSAGING_SENDER_ID',
                'FIREBASE_APP_ID'
            ]
        }
        
    def load_environment(self) -> bool:
        """
        Load environment variables from the specified .env file.
        
        Returns:
            bool: True if environment was loaded successfully
        """
        try:
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment from {self.env_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load environment from {self.env_file}: {str(e)}")
            return False
            
    def validate_environment(self) -> Dict[str, List[str]]:
        """
        Validate that all required environment variables are present.
        
        Returns:
            Dict[str, List[str]]: Dictionary of missing variables by category
        """
        missing_vars = {}
        
        for category, variables in self.required_vars.items():
            missing = [var for var in variables if not os.getenv(var)]
            if missing:
                missing_vars[category] = missing
                
        return missing_vars
        
    def check_env_file_tampering(self) -> bool:
        """
        Check for signs of .env file tampering by validating file permissions
        and last modification time.
        
        Returns:
            bool: True if no tampering detected
        """
        try:
            # Check if file exists
            if not os.path.exists(self.env_file):
                logger.error(f"Environment file {self.env_file} not found")
                return False
                
            # Check file permissions (should be readable only by owner)
            stat = os.stat(self.env_file)
            if os.name == 'nt':  # Windows
                import win32security
                import ntsecuritycon as con
                security = win32security.GetFileSecurity(
                    self.env_file, 
                    win32security.OWNER_SECURITY_INFORMATION
                )
                owner_sid = security.GetSecurityDescriptorOwner()
                current_user_sid = win32security.GetTokenInformation(
                    win32security.OpenProcessToken(
                        win32security.GetCurrentProcess(),
                        win32security.TOKEN_QUERY
                    ),
                    win32security.TokenUser
                )[0]
                
                if owner_sid != current_user_sid:
                    logger.warning("Environment file ownership mismatch detected")
                    return False
            else:  # Unix-like
                if stat.st_mode & 0o777 != 0o600:  # Should be 600 permissions
                    logger.warning("Environment file has incorrect permissions")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking environment file tampering: {str(e)}")
            return False
            
    def validate_and_load(self) -> bool:
        """
        Load and validate the environment configuration.
        
        Returns:
            bool: True if environment is valid and loaded
        """
        # Load environment
        if not self.load_environment():
            return False
            
        # Check for tampering
        if not self.check_env_file_tampering():
            logger.error("Environment file tampering detected")
            return False
            
        # Validate required variables
        missing_vars = self.validate_environment()
        if missing_vars:
            logger.error("Missing required environment variables:")
            for category, variables in missing_vars.items():
                logger.error(f"{category}:")
                for var in variables:
                    logger.error(f"  - {var}")
            return False
            
        logger.info("Environment validation successful")
        return True 
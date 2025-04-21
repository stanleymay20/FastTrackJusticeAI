import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

class ConfigManager:
    """Manages configuration settings for the FastTrackJustice application."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
        
    def load_config(self) -> None:
        """Load configuration from the YAML file."""
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            logger.info("Configuration loaded successfully")
            self._validate_paths()
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def _validate_paths(self) -> None:
        """Validate that all required paths exist or can be created."""
        paths_to_check = [
            self.config['data']['base_path'],
            self.config['data']['cases_path'],
            self.config['data']['principles_path'],
            self.config['data']['memory_path'],
            self.config['data']['logs_path'],
            self.config['data']['cache_path'],
            self.config['data']['temp_path']
        ]
        
        for path in paths_to_check:
            path_obj = Path(path)
            if not path_obj.exists():
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
                
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
        
    def save(self) -> None:
        """Save the current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
            
    def get_app_settings(self) -> Dict[str, Any]:
        """Get application-specific settings."""
        return {
            'name': self.get('app.name'),
            'version': self.get('app.version'),
            'description': self.get('app.description'),
            'debug': self.get('app.debug', False),
            'host': self.get('app.host', '0.0.0.0'),
            'port': self.get('app.port', 8501)
        }
        
    def get_data_paths(self) -> Dict[str, Path]:
        """Get all data-related paths."""
        return {
            'base': Path(self.get('data.base_path')),
            'cases': Path(self.get('data.cases_path')),
            'principles': Path(self.get('data.principles_path')),
            'memory': Path(self.get('data.memory_path')),
            'logs': Path(self.get('data.logs_path')),
            'cache': Path(self.get('data.cache_path')),
            'temp': Path(self.get('data.temp_path'))
        }
        
    def get_database_settings(self) -> Dict[str, Any]:
        """Get database connection settings."""
        return {
            'type': self.get('database.type', 'sqlite'),
            'path': self.get('database.path'),
            'host': self.get('database.host'),
            'port': self.get('database.port'),
            'name': self.get('database.name'),
            'user': self.get('database.user'),
            'password': self.get('database.password')
        }
        
    def get_api_settings(self) -> Dict[str, Any]:
        """Get API settings."""
        return {
            'enabled': self.get('api.enabled', True),
            'host': self.get('api.host', '0.0.0.0'),
            'port': self.get('api.port', 8000),
            'debug': self.get('api.debug', False),
            'cors_origins': self.get('api.cors_origins', ['*']),
            'rate_limit': self.get('api.rate_limit', 100)
        }
        
    def get_logging_settings(self) -> Dict[str, Any]:
        """Get logging settings."""
        return {
            'level': self.get('logging.level', 'INFO'),
            'format': self.get('logging.format'),
            'file': self.get('logging.file'),
            'max_size': self.get('logging.max_size', 10485760),
            'backup_count': self.get('logging.backup_count', 5)
        }
        
    def get_security_settings(self) -> Dict[str, Any]:
        """Get security settings."""
        return {
            'secret_key': self.get('security.secret_key'),
            'token_expiry': self.get('security.token_expiry', 3600),
            'password_hash_algorithm': self.get('security.password_hash_algorithm', 'bcrypt'),
            'session_timeout': self.get('security.session_timeout', 86400)
        }
        
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags."""
        return {
            'scroll_memory': self.get('features.scroll_memory', True),
            'judicial_mode': self.get('features.judicial_mode', True),
            'principle_evolution': self.get('features.principle_evolution', True),
            'case_analysis': self.get('features.case_analysis', True),
            'graph_visualization': self.get('features.graph_visualization', True),
            'export_options': self.get('features.export_options', True),
            'bulk_import': self.get('features.bulk_import', True)
        }
        
    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI settings."""
        return {
            'theme': self.get('ui.theme', 'light'),
            'language': self.get('ui.language', 'en'),
            'timezone': self.get('ui.timezone', 'UTC'),
            'date_format': self.get('ui.date_format', '%Y-%m-%d'),
            'time_format': self.get('ui.time_format', '%H:%M:%S')
        }
        
    def get_data_settings(self) -> Dict[str, Any]:
        """Get data settings."""
        return self.get('data', {})
        
    def get_graph_settings(self) -> Dict[str, Any]:
        """Get graph settings."""
        return self.get('graph', {})
        
    def get_visualization_settings(self) -> Dict[str, Any]:
        """Get visualization settings."""
        return self.get('visualization', {})
        
    def get_scroll_memory_settings(self) -> Dict[str, Any]:
        """Get scroll memory settings."""
        return self.get('scroll_memory', {})
        
    def get_judicial_mode_settings(self) -> Dict[str, Any]:
        """Get judicial mode settings."""
        return self.get('judicial_mode', {})
        
    def get_export_settings(self) -> Dict[str, Any]:
        """Get export settings."""
        return self.get('export', {})
        
    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        logging_settings = self.get_logging_settings()
        
        # Create logs directory if it doesn't exist
        log_file = logging_settings.get('file', 'logs/fasttrackjustice.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging_settings.get('level', 'INFO'),
            format=logging_settings.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            filename=log_file
        )
        
        logger.info("Logging configured successfully") 
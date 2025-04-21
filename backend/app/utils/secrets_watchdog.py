import os
import time
import hashlib
import logging
import threading
import datetime
from typing import Dict, Optional, Set
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvFileHandler(FileSystemEventHandler):
    """Handles .env file modification events."""
    
    def __init__(self, watchdog):
        """Initialize the handler with a reference to the watchdog."""
        self.watchdog = watchdog
        super().__init__()
        
    def on_modified(self, event):
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent):
            if any(str(event.src_path).endswith(pattern) for pattern in ['.env', '.env.development', '.env.production']):
                self.watchdog.handle_env_modification(event.src_path)

class SecretsWatchdog:
    """Monitors .env files for unauthorized modifications."""
    
    def __init__(self, 
                 env_files: Set[str] = None,
                 check_interval: int = 5,
                 alert_callback: Optional[callable] = None,
                 kill_on_tampering: bool = False):
        """
        Initialize the secrets watchdog.
        
        Args:
            env_files: Set of .env file paths to monitor
            check_interval: Seconds between checks
            alert_callback: Function to call when tampering is detected
            kill_on_tampering: Whether to exit the application on tampering
        """
        self.env_files = env_files or {'.env', '.env.development', '.env.production'}
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        self.kill_on_tampering = kill_on_tampering
        
        # Store file hashes and metadata
        self.file_hashes: Dict[str, str] = {}
        self.file_metadata: Dict[str, Dict] = {}
        
        # Initialize watchdog observer
        self.observer = Observer()
        self.event_handler = EnvFileHandler(self)
        
        # Thread control
        self.running = False
        self.monitor_thread = None
        
    def compute_file_hash(self, filepath: str) -> Optional[str]:
        """
        Compute SHA-256 hash of a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            str: SHA-256 hash of the file, or None if file doesn't exist
        """
        try:
            with open(filepath, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return None
            
    def initialize_file_tracking(self):
        """Initialize tracking for all monitored files."""
        for env_file in self.env_files:
            if os.path.exists(env_file):
                self.file_hashes[env_file] = self.compute_file_hash(env_file)
                self.file_metadata[env_file] = {
                    'last_modified': os.path.getmtime(env_file),
                    'size': os.path.getsize(env_file),
                    'permissions': oct(os.stat(env_file).st_mode)[-3:],
                    'owner': os.stat(env_file).st_uid
                }
                logger.info(f"Initialized tracking for {env_file}")
                
    def verify_file_integrity(self, filepath: str) -> bool:
        """
        Verify the integrity of a file.
        
        Args:
            filepath: Path to the file to verify
            
        Returns:
            bool: True if file integrity is maintained
        """
        if not os.path.exists(filepath):
            logger.warning(f"Environment file missing: {filepath}")
            return False
            
        current_hash = self.compute_file_hash(filepath)
        if current_hash != self.file_hashes.get(filepath):
            logger.warning(f"Hash mismatch detected for {filepath}")
            return False
            
        # Check metadata
        try:
            current_metadata = {
                'last_modified': os.path.getmtime(filepath),
                'size': os.path.getsize(filepath),
                'permissions': oct(os.stat(filepath).st_mode)[-3:],
                'owner': os.stat(filepath).st_uid
            }
            
            stored_metadata = self.file_metadata.get(filepath, {})
            
            # Check permissions
            if current_metadata['permissions'] != stored_metadata.get('permissions'):
                logger.warning(f"Permissions changed for {filepath}")
                return False
                
            # Check owner
            if current_metadata['owner'] != stored_metadata.get('owner'):
                logger.warning(f"Owner changed for {filepath}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking metadata for {filepath}: {str(e)}")
            return False
            
        return True
        
    def handle_env_modification(self, filepath: str):
        """
        Handle a detected modification to an env file.
        
        Args:
            filepath: Path to the modified file
        """
        if not self.verify_file_integrity(filepath):
            message = f"🚨 SECURITY ALERT: Unauthorized modification detected in {filepath}"
            logger.critical(message)
            
            # Call alert callback if provided
            if self.alert_callback:
                try:
                    self.alert_callback(filepath, message)
                except Exception as e:
                    logger.error(f"Error in alert callback: {str(e)}")
            
            if self.kill_on_tampering:
                logger.critical("Terminating application due to security breach")
                os._exit(1)
                
    def monitor_files(self):
        """Monitor files in a loop."""
        while self.running:
            for env_file in self.env_files:
                if os.path.exists(env_file):
                    self.handle_env_modification(env_file)
            time.sleep(self.check_interval)
            
    def start(self):
        """Start the watchdog monitor."""
        if self.running:
            logger.warning("Watchdog is already running")
            return
            
        logger.info("Starting secrets watchdog...")
        self.running = True
        
        # Initialize file tracking
        self.initialize_file_tracking()
        
        # Start file system observer
        for env_file in self.env_files:
            if os.path.exists(env_file):
                self.observer.schedule(
                    self.event_handler,
                    os.path.dirname(os.path.abspath(env_file)) or '.',
                    recursive=False
                )
        self.observer.start()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_files)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Secrets watchdog is active and monitoring environment files")
        
    def stop(self):
        """Stop the watchdog monitor."""
        if not self.running:
            logger.warning("Watchdog is not running")
            return
            
        logger.info("Stopping secrets watchdog...")
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        self.observer.stop()
        self.observer.join(timeout=5.0)
        
        logger.info("Secrets watchdog stopped") 
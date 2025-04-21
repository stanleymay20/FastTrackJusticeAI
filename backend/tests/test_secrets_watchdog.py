import os
import time
import pytest
import threading
from unittest.mock import patch, MagicMock, call
from app.utils.secrets_watchdog import SecretsWatchdog, EnvFileHandler

@pytest.fixture
def mock_env_file():
    """Create a temporary .env file for testing."""
    content = "TEST_KEY=test_value"
    with open(".env.test", "w") as f:
        f.write(content)
    yield ".env.test"
    # Cleanup
    if os.path.exists(".env.test"):
        os.remove(".env.test")

@pytest.fixture
def watchdog():
    """Create a SecretsWatchdog instance for testing."""
    return SecretsWatchdog(env_files={".env.test"}, check_interval=1)

def test_initialization(watchdog):
    """Test SecretsWatchdog initialization."""
    assert watchdog.env_files == {".env.test"}
    assert watchdog.check_interval == 1
    assert watchdog.alert_callback is None
    assert watchdog.kill_on_tampering is False
    assert isinstance(watchdog.observer, Observer)
    assert isinstance(watchdog.event_handler, EnvFileHandler)

def test_compute_file_hash(watchdog, mock_env_file):
    """Test file hash computation."""
    # Test existing file
    hash1 = watchdog.compute_file_hash(mock_env_file)
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hash length
    
    # Test non-existent file
    hash2 = watchdog.compute_file_hash("nonexistent.env")
    assert hash2 is None

def test_initialize_file_tracking(watchdog, mock_env_file):
    """Test file tracking initialization."""
    watchdog.initialize_file_tracking()
    
    assert mock_env_file in watchdog.file_hashes
    assert mock_env_file in watchdog.file_metadata
    
    metadata = watchdog.file_metadata[mock_env_file]
    assert 'last_modified' in metadata
    assert 'size' in metadata
    assert 'permissions' in metadata
    assert 'owner' in metadata

def test_verify_file_integrity(watchdog, mock_env_file):
    """Test file integrity verification."""
    # Initialize tracking
    watchdog.initialize_file_tracking()
    
    # Test unmodified file
    assert watchdog.verify_file_integrity(mock_env_file) is True
    
    # Test modified file
    with open(mock_env_file, "a") as f:
        f.write("\nNEW_KEY=new_value")
    assert watchdog.verify_file_integrity(mock_env_file) is False
    
    # Test missing file
    os.remove(mock_env_file)
    assert watchdog.verify_file_integrity(mock_env_file) is False

def test_handle_env_modification(watchdog, mock_env_file):
    """Test handling of environment file modifications."""
    # Setup mock alert callback
    mock_callback = MagicMock()
    watchdog.alert_callback = mock_callback
    
    # Initialize tracking
    watchdog.initialize_file_tracking()
    
    # Test with unmodified file
    watchdog.handle_env_modification(mock_env_file)
    mock_callback.assert_not_called()
    
    # Test with modified file
    with open(mock_env_file, "a") as f:
        f.write("\nNEW_KEY=new_value")
    watchdog.handle_env_modification(mock_env_file)
    mock_callback.assert_called_once()
    assert "SECURITY ALERT" in mock_callback.call_args[0][1]

@patch('os._exit')
def test_kill_on_tampering(mock_exit, watchdog, mock_env_file):
    """Test kill on tampering functionality."""
    watchdog.kill_on_tampering = True
    watchdog.initialize_file_tracking()
    
    # Modify file
    with open(mock_env_file, "a") as f:
        f.write("\nNEW_KEY=new_value")
    
    watchdog.handle_env_modification(mock_env_file)
    mock_exit.assert_called_once_with(1)

def test_start_stop(watchdog, mock_env_file):
    """Test starting and stopping the watchdog."""
    # Test start
    watchdog.start()
    assert watchdog.running is True
    assert watchdog.monitor_thread is not None
    assert watchdog.monitor_thread.is_alive()
    
    # Test double start
    watchdog.start()  # Should log warning
    
    # Test stop
    watchdog.stop()
    assert watchdog.running is False
    assert not watchdog.monitor_thread.is_alive()
    
    # Test double stop
    watchdog.stop()  # Should log warning

def test_file_system_events(watchdog, mock_env_file):
    """Test handling of file system events."""
    watchdog.start()
    
    # Give the watchdog time to initialize
    time.sleep(1)
    
    # Modify file
    with open(mock_env_file, "a") as f:
        f.write("\nNEW_KEY=new_value")
    
    # Give the watchdog time to detect changes
    time.sleep(2)
    
    # Stop watchdog
    watchdog.stop()

def test_alert_callback_error_handling(watchdog, mock_env_file):
    """Test error handling in alert callback."""
    def failing_callback(filepath, message):
        raise Exception("Test error")
    
    watchdog.alert_callback = failing_callback
    watchdog.initialize_file_tracking()
    
    # Modify file
    with open(mock_env_file, "a") as f:
        f.write("\nNEW_KEY=new_value")
    
    # Should not raise exception
    watchdog.handle_env_modification(mock_env_file)

def test_multiple_env_files():
    """Test monitoring multiple env files."""
    # Create test files
    files = [".env.test1", ".env.test2"]
    for file in files:
        with open(file, "w") as f:
            f.write(f"TEST_KEY_{file}=test_value")
    
    try:
        watchdog = SecretsWatchdog(env_files=set(files))
        watchdog.start()
        
        # Give the watchdog time to initialize
        time.sleep(1)
        
        # Modify both files
        for file in files:
            with open(file, "a") as f:
                f.write("\nNEW_KEY=new_value")
        
        # Give the watchdog time to detect changes
        time.sleep(2)
        
        watchdog.stop()
        
    finally:
        # Cleanup
        for file in files:
            if os.path.exists(file):
                os.remove(file)

def test_env_file_handler():
    """Test EnvFileHandler functionality."""
    mock_watchdog = MagicMock()
    handler = EnvFileHandler(mock_watchdog)
    
    # Create mock event
    mock_event = MagicMock()
    mock_event.src_path = ".env"
    
    # Test file modification event
    handler.on_modified(mock_event)
    mock_watchdog.handle_env_modification.assert_called_once_with(".env")

def test_monitor_thread_exception_handling(watchdog, mock_env_file):
    """Test exception handling in monitor thread."""
    def failing_verification(filepath):
        raise Exception("Test error")
    
    watchdog.verify_file_integrity = failing_verification
    watchdog.start()
    
    # Give the watchdog time to run
    time.sleep(2)
    
    # Should not crash
    assert watchdog.monitor_thread.is_alive()
    
    watchdog.stop() 
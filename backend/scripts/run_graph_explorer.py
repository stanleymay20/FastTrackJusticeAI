#!/usr/bin/env python
"""
Run the Precedent Influence Graph Explorer Streamlit app.
This script launches the interactive UI for exploring legal precedent relationships.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the Streamlit app for the Precedent Influence Graph Explorer."""
    try:
        # Get the absolute path to the app
        current_dir = Path(__file__).parent.absolute()
        app_path = current_dir.parent / "app" / "monitoring" / "precedent_graph_explorer.py"
        
        if not app_path.exists():
            logger.error(f"App file not found at {app_path}")
            return 1
        
        logger.info(f"Starting Precedent Influence Graph Explorer at {app_path}")
        
        # Run the Streamlit app
        subprocess.run([
            "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.serverAddress", "localhost"
        ])
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running the app: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
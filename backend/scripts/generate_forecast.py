#!/usr/bin/env python
"""
Scroll Forecast Generator Script

This script generates a daily forecast PDF report with scroll insights,
charts, and recommendations. It can also send the report to Telegram.
"""

import os
import sys
import argparse
import datetime
import logging
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.scroll_forecast_generator import ScrollForecastGenerator
from app.utils.env_validator import EnvironmentValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/forecast_generator.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a scroll forecast PDF report.')
    
    parser.add_argument(
        '--date',
        type=str,
        help='Report date in YYYY-MM-DD format (defaults to today)',
        default=datetime.date.today().strftime('%Y-%m-%d')
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save the generated PDF (default: reports)',
        default='reports'
    )
    
    parser.add_argument(
        '--send-telegram',
        action='store_true',
        help='Send the generated PDF to Telegram'
    )
    
    parser.add_argument(
        '--env-file',
        type=str,
        help='Path to .env file (default: .env)',
        default='.env'
    )
    
    return parser.parse_args()

def load_insights_from_api() -> Dict[str, Any]:
    """
    Load insights data from the API.
    
    Returns:
        Dict containing phase data, severity data, top insights, and historical data
    """
    # In a real implementation, this would fetch data from your API
    # For now, we'll return sample data
    
    # Sample phase data
    phase_data = {
        'Dawn': 25.0,
        'Noon': 30.0,
        'Dusk': 25.0,
        'Night': 20.0
    }
    
    # Sample severity data
    severity_data = {
        'Low': 10,
        'Medium': 15,
        'High': 8,
        'Critical': 3
    }
    
    # Sample top insights
    top_insights = [
        {
            'text': 'The dawn phase shows increased activity in civil cases.',
            'severity': 'high',
            'phase': 'dawn'
        },
        {
            'text': 'Night phase judgments show higher confidence levels.',
            'severity': 'medium',
            'phase': 'night'
        },
        {
            'text': 'Family cases are most prevalent during the dusk phase.',
            'severity': 'medium',
            'phase': 'dusk'
        },
        {
            'text': 'Administrative cases show consistent distribution across phases.',
            'severity': 'low',
            'phase': 'noon'
        },
        {
            'text': 'Criminal cases show higher severity during the night phase.',
            'severity': 'high',
            'phase': 'night'
        }
    ]
    
    # Sample historical data
    today = datetime.date.today()
    historical_data = []
    for i in range(7):
        day = today - datetime.timedelta(days=i)
        historical_data.append({
            'date': day.strftime('%Y-%m-%d'),
            'dawn': 20 + i * 2,
            'noon': 25 + i,
            'dusk': 30 - i,
            'night': 25 - i * 2
        })
    historical_data.reverse()  # Put oldest first
    
    # Sample prophetic window
    prophetic_window = (
        "Today's prophetic window aligns with the transition from dawn to noon. "
        "This period is optimal for civil case judgments, with a 15% increase in "
        "confidence levels expected. Focus on administrative cases during the noon "
        "phase for maximum effectiveness."
    )
    
    return {
        'phase_data': phase_data,
        'severity_data': severity_data,
        'top_insights': top_insights,
        'historical_data': historical_data,
        'prophetic_window': prophetic_window
    }

def main():
    """Main function to generate the forecast PDF."""
    # Parse command line arguments
    args = parse_args()
    
    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Validate environment
    env_validator = EnvironmentValidator(args.env_file)
    if not env_validator.validate_and_load():
        logger.error("Environment validation failed. Exiting.")
        sys.exit(1)
    
    # Parse the date
    try:
        report_date = datetime.datetime.strptime(args.date, '%Y-%m-%d').date()
    except ValueError:
        logger.error(f"Invalid date format: {args.date}. Expected YYYY-MM-DD.")
        sys.exit(1)
    
    # Load insights data
    logger.info("Loading insights data...")
    insights_data = load_insights_from_api()
    
    # Generate the PDF
    logger.info(f"Generating forecast PDF for {report_date}...")
    generator = ScrollForecastGenerator(args.output_dir)
    pdf_path = generator.generate_forecast_pdf(
        date=report_date,
        phase_data=insights_data['phase_data'],
        severity_data=insights_data['severity_data'],
        top_insights=insights_data['top_insights'],
        historical_data=insights_data['historical_data'],
        prophetic_window=insights_data['prophetic_window']
    )
    
    logger.info(f"Forecast PDF generated: {pdf_path}")
    
    # Send to Telegram if requested
    if args.send_telegram:
        logger.info("Sending PDF to Telegram...")
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in environment.")
            sys.exit(1)
        
        if generator.send_to_telegram(pdf_path, bot_token, chat_id):
            logger.info("PDF sent to Telegram successfully.")
        else:
            logger.error("Failed to send PDF to Telegram.")
            sys.exit(1)
    
    logger.info("Forecast generation completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScrollForecastGenerator:
    """Generates PDF forecast reports with scroll insights and recommendations."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the forecast generator.
        
        Args:
            output_dir: Directory to save generated PDFs
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        
        # Register custom fonts if available
        self._register_fonts()
        
        # Define styles
        self.styles = getSampleStyleSheet()
        self._define_custom_styles()
        
    def _ensure_output_dir(self):
        """Ensure the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
            
    def _register_fonts(self):
        """Register custom fonts for the PDF."""
        # Check for custom fonts in the fonts directory
        fonts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "fonts")
        if os.path.exists(fonts_dir):
            for font_file in os.listdir(fonts_dir):
                if font_file.endswith(('.ttf', '.otf')):
                    font_name = os.path.splitext(font_file)[0]
                    try:
                        pdfmetrics.registerFont(TTFont(font_name, os.path.join(fonts_dir, font_file)))
                        logger.info(f"Registered font: {font_name}")
                    except Exception as e:
                        logger.warning(f"Failed to register font {font_name}: {str(e)}")
                        
    def _define_custom_styles(self):
        """Define custom paragraph styles for the PDF."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ScrollTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ScrollSubtitle',
            parent=self.styles['Subtitle'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='ScrollHeading',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='ScrollBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8
        ))
        
        # Insight style
        self.styles.add(ParagraphStyle(
            name='ScrollInsight',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            leftIndent=20,
            textColor=colors.darkblue
        ))
        
    def generate_phase_chart(self, phase_data: Dict[str, float]) -> bytes:
        """
        Generate a pie chart for phase distribution.
        
        Args:
            phase_data: Dictionary of phase names and their percentages
            
        Returns:
            bytes: PNG image data
        """
        plt.figure(figsize=(8, 6))
        phases = list(phase_data.keys())
        values = list(phase_data.values())
        
        # Define colors for each phase
        colors_map = {
            'dawn': '#FFD700',  # Gold
            'noon': '#FFA500',  # Orange
            'dusk': '#8B4513',  # Saddle Brown
            'night': '#000080'  # Navy
        }
        
        # Use phase-specific colors if available, otherwise use default
        chart_colors = [colors_map.get(phase.lower(), plt.cm.Pastel1(i)) for i, phase in enumerate(phases)]
        
        plt.pie(values, labels=phases, autopct='%1.1f%%', startangle=90, colors=chart_colors)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Scroll Phase Distribution', fontsize=14)
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        return buf.getvalue()
        
    def generate_severity_chart(self, severity_data: Dict[str, int]) -> bytes:
        """
        Generate a bar chart for severity distribution.
        
        Args:
            severity_data: Dictionary of severity levels and their counts
            
        Returns:
            bytes: PNG image data
        """
        plt.figure(figsize=(8, 6))
        severities = list(severity_data.keys())
        counts = list(severity_data.values())
        
        # Define colors for each severity level
        colors_map = {
            'low': '#90EE90',    # Light Green
            'medium': '#FFD700',  # Gold
            'high': '#FF6347',    # Tomato
            'critical': '#8B0000'  # Dark Red
        }
        
        # Use severity-specific colors if available, otherwise use default
        chart_colors = [colors_map.get(severity.lower(), plt.cm.Pastel1(i)) for i, severity in enumerate(severities)]
        
        plt.bar(severities, counts, color=chart_colors)
        plt.xlabel('Severity Level', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Insight Severity Distribution', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        return buf.getvalue()
        
    def generate_historical_chart(self, historical_data: List[Dict[str, Any]]) -> bytes:
        """
        Generate a line chart for historical phase tracking.
        
        Args:
            historical_data: List of dictionaries with date and phase data
            
        Returns:
            bytes: PNG image data
        """
        plt.figure(figsize=(10, 6))
        
        # Extract dates and phase data
        dates = [item['date'] for item in historical_data]
        phases = ['dawn', 'noon', 'dusk', 'night']
        
        # Create a line for each phase
        for phase in phases:
            values = [item.get(phase, 0) for item in historical_data]
            plt.plot(dates, values, marker='o', label=phase.capitalize())
            
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.title('Historical Scroll Phase Distribution', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()
        
    def generate_forecast_pdf(self, 
                             date: Optional[datetime.date] = None,
                             phase_data: Optional[Dict[str, float]] = None,
                             severity_data: Optional[Dict[str, int]] = None,
                             top_insights: Optional[List[Dict[str, Any]]] = None,
                             historical_data: Optional[List[Dict[str, Any]]] = None,
                             prophetic_window: Optional[str] = None) -> str:
        """
        Generate a PDF forecast report.
        
        Args:
            date: Report date (defaults to today)
            phase_data: Phase distribution data
            severity_data: Severity distribution data
            top_insights: List of top insights
            historical_data: Historical phase tracking data
            prophetic_window: Description of today's prophetic window
            
        Returns:
            str: Path to the generated PDF file
        """
        # Use today's date if not provided
        if date is None:
            date = datetime.date.today()
            
        # Use sample data if not provided
        if phase_data is None:
            phase_data = {
                'Dawn': 25.0,
                'Noon': 30.0,
                'Dusk': 25.0,
                'Night': 20.0
            }
            
        if severity_data is None:
            severity_data = {
                'Low': 10,
                'Medium': 15,
                'High': 8,
                'Critical': 3
            }
            
        if top_insights is None:
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
            
        if historical_data is None:
            # Generate sample historical data for the past 7 days
            historical_data = []
            for i in range(7):
                day = date - datetime.timedelta(days=i)
                historical_data.append({
                    'date': day.strftime('%Y-%m-%d'),
                    'dawn': 20 + i * 2,
                    'noon': 25 + i,
                    'dusk': 30 - i,
                    'night': 25 - i * 2
                })
            historical_data.reverse()  # Put oldest first
            
        if prophetic_window is None:
            prophetic_window = (
                "Today's prophetic window aligns with the transition from dawn to noon. "
                "This period is optimal for civil case judgments, with a 15% increase in "
                "confidence levels expected. Focus on administrative cases during the noon "
                "phase for maximum effectiveness."
            )
            
        # Generate filename
        filename = f"scroll_forecast_{date.strftime('%Y%m%d')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build content
        content = []
        
        # Add title
        content.append(Paragraph(f"Scroll Forecast Report", self.styles['ScrollTitle']))
        content.append(Paragraph(f"{date.strftime('%B %d, %Y')}", self.styles['ScrollSubtitle']))
        content.append(Spacer(1, 12))
        
        # Add phase distribution chart
        content.append(Paragraph("Phase Distribution", self.styles['ScrollHeading']))
        phase_chart = Image(self.generate_phase_chart(phase_data), width=6*inch, height=4.5*inch)
        content.append(phase_chart)
        content.append(Spacer(1, 12))
        
        # Add severity distribution chart
        content.append(Paragraph("Severity Distribution", self.styles['ScrollHeading']))
        severity_chart = Image(self.generate_severity_chart(severity_data), width=6*inch, height=4.5*inch)
        content.append(severity_chart)
        content.append(Spacer(1, 12))
        
        # Add top insights
        content.append(Paragraph("Top Insights", self.styles['ScrollHeading']))
        for i, insight in enumerate(top_insights, 1):
            insight_text = f"{i}. {insight['text']} ({insight['severity'].capitalize()} - {insight['phase'].capitalize()})"
            content.append(Paragraph(insight_text, self.styles['ScrollInsight']))
        content.append(Spacer(1, 12))
        
        # Add historical tracking
        content.append(Paragraph("Historical Phase Tracking", self.styles['ScrollHeading']))
        historical_chart = Image(self.generate_historical_chart(historical_data), width=6*inch, height=4.5*inch)
        content.append(historical_chart)
        content.append(Spacer(1, 12))
        
        # Add prophetic window
        content.append(Paragraph("Today's Prophetic Window", self.styles['ScrollHeading']))
        content.append(Paragraph(prophetic_window, self.styles['ScrollBody']))
        content.append(Spacer(1, 12))
        
        # Add phase gate summary
        content.append(Paragraph("Phase Gate Summary", self.styles['ScrollHeading']))
        
        # Create phase gate table
        phase_gates = [
            ["Phase", "Status", "Optimal Case Types"],
            ["Dawn", "Active", "Civil, Family"],
            ["Noon", "Peak", "Administrative, Civil"],
            ["Dusk", "Transition", "Family, Criminal"],
            ["Night", "Rest", "Criminal, Administrative"]
        ]
        
        table = Table(phase_gates, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        content.append(table)
        
        # Build PDF
        doc.build(content)
        
        logger.info(f"Generated forecast PDF: {filepath}")
        return filepath
        
    def send_to_telegram(self, filepath: str, bot_token: str, chat_id: str) -> bool:
        """
        Send the generated PDF to Telegram.
        
        Args:
            filepath: Path to the PDF file
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
            
        Returns:
            bool: True if sent successfully
        """
        try:
            import requests
            
            # Prepare the API URL
            api_url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
            
            # Open the file
            with open(filepath, 'rb') as pdf_file:
                # Prepare the files and data
                files = {
                    'document': (os.path.basename(filepath), pdf_file, 'application/pdf')
                }
                data = {
                    'chat_id': chat_id,
                    'caption': f"📜 Scroll Forecast Report - {datetime.date.today().strftime('%B %d, %Y')}"
                }
                
                # Send the request
                response = requests.post(api_url, files=files, data=data)
                
                # Check if the request was successful
                if response.status_code == 200:
                    logger.info(f"Successfully sent PDF to Telegram: {filepath}")
                    return True
                else:
                    logger.error(f"Failed to send PDF to Telegram: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending PDF to Telegram: {str(e)}")
            return False 
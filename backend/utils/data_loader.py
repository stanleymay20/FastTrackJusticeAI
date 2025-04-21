import PyPDF2
import pdfplumber
from typing import Dict, List, Optional, Union
import os
from pathlib import Path
import json
import pandas as pd
from dotenv import load_dotenv

class DataLoader:
    def __init__(self):
        load_dotenv()
        self.supported_formats = [".pdf", ".txt", ".docx"]
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
    def extract_text(self, content: bytes, filename: str) -> str:
        """
        Extract text from a legal document.
        """
        try:
            # Get file extension
            ext = Path(filename).suffix.lower()
            
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {ext}")
                
            # Check file size
            if len(content) > self.max_file_size:
                raise ValueError("File size exceeds maximum limit of 10MB")
                
            # Extract text based on file type
            if ext == ".pdf":
                return self._extract_from_pdf(content)
            elif ext == ".txt":
                return content.decode("utf-8")
            elif ext == ".docx":
                return self._extract_from_docx(content)
                
        except Exception as e:
            raise Exception(f"Error extracting text: {str(e)}")
            
    def _extract_from_pdf(self, content: bytes) -> str:
        """
        Extract text from PDF using pdfplumber.
        """
        try:
            text = ""
            with pdfplumber.open(content) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            # Fallback to PyPDF2 if pdfplumber fails
            try:
                reader = PyPDF2.PdfReader(content)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
            except Exception as e2:
                raise Exception(f"Failed to extract text from PDF: {str(e2)}")
                
    def _extract_from_docx(self, content: bytes) -> str:
        """
        Extract text from DOCX files.
        """
        try:
            from docx import Document
            from io import BytesIO
            
            doc = Document(BytesIO(content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
            
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load a legal dataset from various sources.
        """
        try:
            # Check if path exists
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
                
            # Load based on file type
            ext = Path(dataset_path).suffix.lower()
            
            if ext == ".csv":
                return pd.read_csv(dataset_path)
            elif ext == ".json":
                return pd.read_json(dataset_path)
            elif ext == ".jsonl":
                return pd.read_json(dataset_path, lines=True)
            else:
                raise ValueError(f"Unsupported dataset format: {ext}")
                
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
            
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded dataset.
        """
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Remove rows with missing values
            df = df.dropna()
            
            # Clean text columns
            text_columns = df.select_dtypes(include=["object"]).columns
            for col in text_columns:
                df[col] = df[col].apply(self._clean_text)
                
            return df
            
        except Exception as e:
            raise Exception(f"Error preprocessing dataset: {str(e)}")
            
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        """
        if not isinstance(text, str):
            return ""
            
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters
        text = text.replace("\n", " ").replace("\t", " ")
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        
        return text.strip()
        
    def save_processed_data(self, data: Union[pd.DataFrame, Dict, List], 
                          output_path: str) -> None:
        """
        Save processed data to file.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save based on file type
            ext = Path(output_path).suffix.lower()
            
            if ext == ".csv":
                data.to_csv(output_path, index=False)
            elif ext == ".json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported output format: {ext}")
                
        except Exception as e:
            raise Exception(f"Error saving processed data: {str(e)}")
            
    def validate_document(self, content: bytes, filename: str) -> Dict[str, bool]:
        """
        Validate a legal document.
        """
        validation = {
            "format_supported": False,
            "size_ok": False,
            "text_extractable": False,
            "content_valid": False
        }
        
        try:
            # Check format
            ext = Path(filename).suffix.lower()
            validation["format_supported"] = ext in self.supported_formats
            
            # Check size
            validation["size_ok"] = len(content) <= self.max_file_size
            
            # Try to extract text
            text = self.extract_text(content, filename)
            validation["text_extractable"] = bool(text)
            
            # Check content validity
            validation["content_valid"] = len(text.split()) >= 50  # Minimum word count
            
            return validation
            
        except Exception:
            return validation 
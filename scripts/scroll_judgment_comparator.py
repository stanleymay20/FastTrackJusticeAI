#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scroll Judgment Comparator - Analyzes and compares judgments across different scroll phases,
languages, and case types to demonstrate the scroll-aware judgment generation system.
"""

import os
import sys
import re
import csv
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Legal terminology categories
LEGAL_TERMS = {
    "criminal": [
        "assault", "defendant", "prosecution", "evidence", "testimony", "witness",
        "guilty", "innocent", "sentence", "conviction", "plea", "bail", "parole"
    ],
    "civil": [
        "plaintiff", "defendant", "contract", "breach", "damages", "liability",
        "negligence", "compensation", "settlement", "agreement", "obligation"
    ],
    "family": [
        "custody", "visitation", "child support", "alimony", "divorce", "petitioner",
        "respondent", "parental rights", "guardianship", "adoption", "paternity"
    ],
    "general": [
        "court", "judge", "jurisdiction", "hearing", "trial", "appeal", "verdict",
        "ruling", "order", "motion", "petition", "complaint", "answer", "discovery"
    ]
}

# Scroll phase tone indicators
SCROLL_TONE_INDICATORS = {
    "dawn": [
        "hope", "renewal", "beginning", "light", "promise", "opportunity", "fresh",
        "awakening", "potential", "dawn", "morning", "sunrise", "birth", "rebirth"
    ],
    "noon": [
        "clarity", "certainty", "authority", "power", "strength", "confidence",
        "determination", "resolve", "noon", "day", "sun", "bright", "clear", "direct"
    ],
    "dusk": [
        "reflection", "consideration", "balance", "wisdom", "experience", "transition",
        "change", "dusk", "evening", "sunset", "twilight", "contemplation", "meditation"
    ],
    "night": [
        "mystery", "caution", "restraint", "prudence", "care", "protection", "safety",
        "night", "darkness", "shadow", "moon", "stars", "silence", "quiet"
    ]
}

# Divine titles and their associations
DIVINE_TITLES = {
    "criminal": {
        "dawn": "The Dawnbringer of Justice",
        "noon": "The Solar Arbiter",
        "dusk": "The Twilight Judge",
        "night": "The Nocturnal Guardian"
    },
    "civil": {
        "dawn": "The Dawn of Equity",
        "noon": "The Solar Mediator",
        "dusk": "The Twilight Balancer",
        "night": "The Nocturnal Protector"
    },
    "family": {
        "dawn": "The Dawn of Harmony",
        "noon": "The Solar Guardian",
        "dusk": "The Twilight Counselor",
        "night": "The Nocturnal Shepherd"
    }
}

def load_judgments(examples_dir):
    """Load all generated judgments from the examples directory."""
    judgments = []
    
    for filename in os.listdir(examples_dir):
        if filename.endswith(".txt"):
            # Parse filename to extract case_id, phase, and language
            parts = filename.replace(".txt", "").split("_")
            if len(parts) >= 3:
                case_id = parts[0]
                phase = parts[1]
                language = parts[2]
                
                # Read the judgment text
                with open(os.path.join(examples_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                
                judgments.append({
                    "case_id": case_id,
                    "phase": phase,
                    "language": language,
                    "text": text
                })
    
    return judgments

def analyze_legal_terminology(text, category=None):
    """Analyze the density of legal terminology in the judgment text."""
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count legal terms by category
    term_counts = {}
    total_terms = 0
    
    for cat, terms in LEGAL_TERMS.items():
        if category is None or cat == category:
            count = sum(1 for term in terms if term.lower() in text_lower)
            term_counts[cat] = count
            total_terms += count
    
    return term_counts, total_terms

def analyze_scroll_tone(text, phase):
    """Analyze the tone indicators for the given scroll phase in the text."""
    text_lower = text.lower()
    
    # Count tone indicators for the given phase
    tone_count = sum(1 for term in SCROLL_TONE_INDICATORS[phase] if term.lower() in text_lower)
    
    # Calculate tone density (terms per 1000 words)
    word_count = len(word_tokenize(text))
    tone_density = (tone_count / word_count) * 1000 if word_count > 0 else 0
    
    return tone_count, tone_density

def analyze_sentiment(text):
    """Analyze the sentiment of the judgment text."""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def analyze_structure(text):
    """Analyze the structure of the judgment text."""
    # Check for common judgment sections
    sections = {
        "introduction": bool(re.search(r"introduction|background|case\s+summary", text.lower())),
        "facts": bool(re.search(r"facts|evidence|testimony|allegations", text.lower())),
        "analysis": bool(re.search(r"analysis|consideration|evaluation|assessment", text.lower())),
        "conclusion": bool(re.search(r"conclusion|verdict|decision|ruling", text.lower())),
        "divine_title": bool(re.search(r"the\s+[a-z]+\s+[a-z]+", text.lower()))
    }
    
    # Count the number of sections present
    section_count = sum(1 for present in sections.values() if present)
    
    return sections, section_count

def analyze_judgments(judgments):
    """Analyze all judgments and generate comparison metrics."""
    results = []
    
    for judgment in judgments:
        case_id = judgment["case_id"]
        phase = judgment["phase"]
        language = judgment["language"]
        text = judgment["text"]
        
        # Determine case category based on case_id
        category = None
        if "CASE-2023-001" in case_id:
            category = "criminal"
        elif "CASE-2023-002" in case_id:
            category = "civil"
        elif "CASE-2023-003" in case_id:
            category = "family"
        
        # Analyze legal terminology
        term_counts, total_terms = analyze_legal_terminology(text, category)
        
        # Analyze scroll tone
        tone_count, tone_density = analyze_scroll_tone(text, phase)
        
        # Analyze sentiment
        sentiment_polarity, sentiment_subjectivity = analyze_sentiment(text)
        
        # Analyze structure
        sections, section_count = analyze_structure(text)
        
        # Calculate word count
        word_count = len(word_tokenize(text))
        
        # Store results
        result = {
            "case_id": case_id,
            "phase": phase,
            "language": language,
            "category": category,
            "word_count": word_count,
            "total_legal_terms": total_terms,
            "legal_term_density": (total_terms / word_count) * 1000 if word_count > 0 else 0,
            "tone_count": tone_count,
            "tone_density": tone_density,
            "sentiment_polarity": sentiment_polarity,
            "sentiment_subjectivity": sentiment_subjectivity,
            "section_count": section_count,
            "has_introduction": sections["introduction"],
            "has_facts": sections["facts"],
            "has_analysis": sections["analysis"],
            "has_conclusion": sections["conclusion"],
            "has_divine_title": sections["divine_title"]
        }
        
        # Add category-specific term counts
        for cat, count in term_counts.items():
            result[f"{cat}_terms"] = count
        
        results.append(result)
    
    return results

def generate_csv_report(results, output_file):
    """Generate a CSV report of the analysis results."""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"CSV report generated: {output_file}")

def generate_markdown_report(results, output_file):
    """Generate a Markdown report of the analysis results."""
    df = pd.DataFrame(results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Scroll Judgment Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total judgments analyzed: {len(results)}\n")
        f.write(f"- Average word count: {df['word_count'].mean():.2f}\n")
        f.write(f"- Average legal term density: {df['legal_term_density'].mean():.2f} terms per 1000 words\n")
        f.write(f"- Average tone density: {df['tone_density'].mean():.2f} terms per 1000 words\n")
        f.write(f"- Average sentiment polarity: {df['sentiment_polarity'].mean():.2f}\n")
        f.write(f"- Average section count: {df['section_count'].mean():.2f}\n\n")
        
        # Analysis by scroll phase
        f.write("## Analysis by Scroll Phase\n\n")
        for phase in ["dawn", "noon", "dusk", "night"]:
            phase_df = df[df["phase"] == phase]
            if not phase_df.empty:
                f.write(f"### {phase.capitalize()} Phase\n\n")
                f.write(f"- Average word count: {phase_df['word_count'].mean():.2f}\n")
                f.write(f"- Average legal term density: {phase_df['legal_term_density'].mean():.2f} terms per 1000 words\n")
                f.write(f"- Average tone density: {phase_df['tone_density'].mean():.2f} terms per 1000 words\n")
                f.write(f"- Average sentiment polarity: {phase_df['sentiment_polarity'].mean():.2f}\n")
                f.write(f"- Average section count: {phase_df['section_count'].mean():.2f}\n\n")
        
        # Analysis by case category
        f.write("## Analysis by Case Category\n\n")
        for category in ["criminal", "civil", "family"]:
            category_df = df[df["category"] == category]
            if not category_df.empty:
                f.write(f"### {category.capitalize()} Cases\n\n")
                f.write(f"- Average word count: {category_df['word_count'].mean():.2f}\n")
                f.write(f"- Average legal term density: {category_df['legal_term_density'].mean():.2f} terms per 1000 words\n")
                f.write(f"- Average tone density: {category_df['tone_density'].mean():.2f} terms per 1000 words\n")
                f.write(f"- Average sentiment polarity: {category_df['sentiment_polarity'].mean():.2f}\n")
                f.write(f"- Average section count: {category_df['section_count'].mean():.2f}\n\n")
        
        # Analysis by language
        f.write("## Analysis by Language\n\n")
        for language in df["language"].unique():
            lang_df = df[df["language"] == language]
            if not lang_df.empty:
                f.write(f"### {language.capitalize()}\n\n")
                f.write(f"- Average word count: {lang_df['word_count'].mean():.2f}\n")
                f.write(f"- Average legal term density: {lang_df['legal_term_density'].mean():.2f} terms per 1000 words\n")
                f.write(f"- Average tone density: {lang_df['tone_density'].mean():.2f} terms per 1000 words\n")
                f.write(f"- Average sentiment polarity: {lang_df['sentiment_polarity'].mean():.2f}\n")
                f.write(f"- Average section count: {lang_df['section_count'].mean():.2f}\n\n")
        
        # Detailed comparison table
        f.write("## Detailed Comparison\n\n")
        f.write("| Case ID | Phase | Language | Word Count | Legal Terms | Tone Density | Sentiment | Sections |\n")
        f.write("|---------|-------|----------|------------|-------------|--------------|-----------|----------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['case_id']} | {row['phase']} | {row['language']} | {row['word_count']} | {row['total_legal_terms']} | {row['tone_density']:.2f} | {row['sentiment_polarity']:.2f} | {row['section_count']} |\n")
    
    print(f"Markdown report generated: {output_file}")

def generate_visualizations(results, output_dir):
    """Generate visualizations of the analysis results."""
    df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style for all plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Legal term density by phase
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="phase", y="legal_term_density", data=df)
    plt.title("Legal Term Density by Scroll Phase")
    plt.xlabel("Scroll Phase")
    plt.ylabel("Legal Terms per 1000 Words")
    plt.savefig(os.path.join(output_dir, "legal_term_density_by_phase.png"))
    plt.close()
    
    # 2. Tone density by phase
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="phase", y="tone_density", data=df)
    plt.title("Scroll Tone Density by Phase")
    plt.xlabel("Scroll Phase")
    plt.ylabel("Tone Terms per 1000 Words")
    plt.savefig(os.path.join(output_dir, "tone_density_by_phase.png"))
    plt.close()
    
    # 3. Sentiment by phase
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="phase", y="sentiment_polarity", data=df)
    plt.title("Sentiment Polarity by Scroll Phase")
    plt.xlabel("Scroll Phase")
    plt.ylabel("Sentiment Polarity")
    plt.savefig(os.path.join(output_dir, "sentiment_by_phase.png"))
    plt.close()
    
    # 4. Section count by phase
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="phase", y="section_count", data=df)
    plt.title("Section Count by Scroll Phase")
    plt.xlabel("Scroll Phase")
    plt.ylabel("Number of Sections")
    plt.savefig(os.path.join(output_dir, "section_count_by_phase.png"))
    plt.close()
    
    # 5. Legal term density by category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="category", y="legal_term_density", data=df)
    plt.title("Legal Term Density by Case Category")
    plt.xlabel("Case Category")
    plt.ylabel("Legal Terms per 1000 Words")
    plt.savefig(os.path.join(output_dir, "legal_term_density_by_category.png"))
    plt.close()
    
    # 6. Heatmap of legal terms by category and phase
    pivot_table = df.pivot_table(
        values=["criminal_terms", "civil_terms", "family_terms", "general_terms"],
        index="phase",
        aggfunc="mean"
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Legal Terms by Category and Scroll Phase")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "legal_terms_heatmap.png"))
    plt.close()
    
    print(f"Visualizations generated in {output_dir}")

def main():
    """Main function to run the scroll judgment comparator."""
    # Get the examples directory
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    
    # Create output directory for reports
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load judgments
    print("Loading judgments...")
    judgments = load_judgments(examples_dir)
    print(f"Loaded {len(judgments)} judgments")
    
    # Analyze judgments
    print("Analyzing judgments...")
    results = analyze_judgments(judgments)
    
    # Generate reports
    print("Generating reports...")
    generate_csv_report(results, os.path.join(output_dir, "scroll_judgment_analysis.csv"))
    generate_markdown_report(results, os.path.join(output_dir, "scroll_judgment_analysis.md"))
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(results, os.path.join(output_dir, "visualizations"))
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 
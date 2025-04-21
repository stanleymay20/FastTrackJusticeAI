# FastTrackJusticeAI Scripts

This directory contains utility scripts for the FastTrackJusticeAI project, particularly focused on the scroll-aware judgment generation system.

## Available Scripts

### 1. `generate_example_judgments.py`

Generates example judgments with varying scroll contexts for demonstration purposes.

**Features:**
- Creates judgments for different case types (criminal, civil, family)
- Tests all scroll phases (dawn, noon, dusk, night)
- Supports multiple languages (English, Spanish, French, German)
- Saves judgments to the `examples/` directory

**Usage:**
```bash
python scripts/generate_example_judgments.py
```

### 2. `scroll_judgment_comparator.py`

Analyzes and compares judgments across different scroll phases, languages, and case types.

**Features:**
- Analyzes legal terminology density
- Measures scroll tone indicators
- Evaluates sentiment and structure
- Generates CSV and Markdown reports
- Creates visualizations of the analysis results

**Usage:**
```bash
python scripts/scroll_judgment_comparator.py
```

### 3. `run_scroll_demo.py`

Runs the complete scroll judgment demonstration suite, generating judgments and analyzing them in sequence.

**Features:**
- Executes both generation and analysis scripts
- Creates detailed logs of the process
- Provides a summary of the results

**Usage:**
```bash
python scripts/run_scroll_demo.py
```

## Output Directories

- **`examples/`**: Contains the generated example judgments
- **`reports/`**: Contains analysis reports and visualizations
- **`logs/`**: Contains log files from the demonstration runs

## Dependencies

These scripts require the following Python packages:
- pandas
- matplotlib
- seaborn
- nltk
- textblob

You can install these dependencies using:
```bash
pip install -r requirements.txt
```

## Scroll Phase Analysis

The scroll judgment comparator analyzes how different scroll phases influence judgment generation:

- **Dawn Phase**: Judgments tend to be more hopeful and focused on renewal
- **Noon Phase**: Judgments are more authoritative and direct
- **Dusk Phase**: Judgments show more reflection and balance
- **Night Phase**: Judgments emphasize caution and prudence

The analysis includes visualizations that demonstrate these differences across phases, case types, and languages. 
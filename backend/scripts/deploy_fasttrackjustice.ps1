# FastTrackJustice Deployment Script
# This script automates the deployment of the FastTrackJustice Precedent Influence Graph Explorer

# Set error action preference
$ErrorActionPreference = "Stop"

# Function for colorful output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to check Python version
function Test-PythonVersion {
    try {
        $pythonVersion = python --version
        Write-ColorOutput Green "Python version: $pythonVersion"
        return $true
    }
    catch {
        Write-ColorOutput Red "Python is not installed or not in PATH"
        return $false
    }
}

# Function to create and activate virtual environment
function Initialize-VirtualEnvironment {
    param (
        [string]$venvPath = "..\venv"
    )
    
    if (-not (Test-Path $venvPath)) {
        Write-ColorOutput Yellow "Creating virtual environment at $venvPath..."
        python -m venv $venvPath
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput Red "Failed to create virtual environment"
            exit 1
        }
    }
    
    # Activate virtual environment
    Write-ColorOutput Green "Activating virtual environment..."
    & "$venvPath\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Failed to activate virtual environment"
        exit 1
    }
}

# Function to install dependencies
function Install-Dependencies {
    Write-ColorOutput Green "Installing dependencies..."
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install required packages
    $packages = @(
        "streamlit",
        "pandas",
        "networkx",
        "plotly",
        "scikit-learn",
        "sentence-transformers",
        "faiss-cpu",
        "python-dotenv",
        "requests",
        "beautifulsoup4",
        "nltk",
        "spacy",
        "torch",
        "transformers"
    )
    
    foreach ($package in $packages) {
        Write-ColorOutput Yellow "Installing $package..."
        pip install $package
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput Red "Failed to install $package"
            exit 1
        }
    }
    
    # Download spaCy model
    Write-ColorOutput Yellow "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
    
    # Download NLTK data
    Write-ColorOutput Yellow "Downloading NLTK data..."
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
}

# Function to create necessary directories
function Initialize-Directories {
    Write-ColorOutput Green "Creating necessary directories..."
    
    $directories = @(
        "..\data",
        "..\data\cases",
        "..\data\principles",
        "..\data\memory",
        "..\logs",
        "..\exports"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir | Out-Null
            Write-ColorOutput Yellow "Created directory: $dir"
        }
    }
}

# Function to create configuration file
function Initialize-Configuration {
    Write-ColorOutput Green "Creating configuration file..."
    
    $configPath = "..\config.json"
    if (-not (Test-Path $configPath)) {
        $config = @{
            "data_source"     = "public_domain"
            "api_limit"       = 1000
            "api_usage"       = 0
            "judicial_mode"   = $true
            "sanctified_mode" = $false
            "memory_path"     = "..\data\memory\scroll_memory.json"
            "cache_dir"       = "..\data\cache"
            "log_level"       = "INFO"
            "export_dir"      = "..\exports"
        } | ConvertTo-Json -Depth 10
        
        $config | Out-File -FilePath $configPath -Encoding UTF8
        Write-ColorOutput Yellow "Created configuration file: $configPath"
    }
}

# Function to create environment file
function Initialize-EnvironmentFile {
    Write-ColorOutput Green "Creating environment file..."
    
    $envPath = "..\.env"
    if (-not (Test-Path $envPath)) {
        $envContent = @"
# FastTrackJustice Environment Variables
DATA_SOURCE=public_domain
API_LIMIT=1000
API_USAGE=0
JUDICIAL_MODE=true
SANCTIFIED_MODE=false
MEMORY_PATH=../data/memory/scroll_memory.json
CACHE_DIR=../data/cache
LOG_LEVEL=INFO
EXPORT_DIR=../exports

# Add your API keys below (if using licensed data)
# LEXISNEXIS_API_KEY=your_key_here
# WESTLAW_API_KEY=your_key_here
"@
        
        $envContent | Out-File -FilePath $envPath -Encoding UTF8
        Write-ColorOutput Yellow "Created environment file: $envPath"
    }
}

# Function to create README file
function Initialize-Readme {
    Write-ColorOutput Green "Creating README file..."
    
    $readmePath = "..\README.md"
    if (-not (Test-Path $readmePath)) {
        $readmeContent = @"
# FastTrackJustice Precedent Influence Graph Explorer

A revolutionary legal intelligence platform that transforms how legal precedents are discovered, analyzed, and applied.

## Features

- Interactive Graph Visualization
- Principle Evolution Tracking
- Cross-Jurisdictional Analysis
- Judicial Mode with Enhanced Transparency
- Scroll Memory Intelligence
- Faith-Safe Protocol

## Installation

1. Ensure Python 3.8+ is installed
2. Run the deployment script: `.\deploy_fasttrackjustice.ps1`
3. Follow the prompts to complete setup

## Usage

1. Activate the virtual environment: `..\venv\Scripts\Activate.ps1`
2. Run the Streamlit app: `streamlit run backend\app\monitoring\precedent_graph_explorer.py`
3. Access the app at http://localhost:8501

## Configuration

Edit the `.env` file to customize:
- Data source (public_domain or licensed_database)
- API limits and usage tracking
- Judicial and Sanctified modes
- File paths and logging

## License

MIT License - See LICENSE file for details

## Contact

[contact@fasttrackjustice.org](mailto:contact@fasttrackjustice.org)
"@
        
        $readmeContent | Out-File -FilePath $readmePath -Encoding UTF8
        Write-ColorOutput Yellow "Created README file: $readmePath"
    }
}

# Function to create license file
function Initialize-License {
    Write-ColorOutput Green "Creating license file..."
    
    $licensePath = "..\LICENSE"
    if (-not (Test-Path $licensePath)) {
        $licenseContent = @"
MIT License

Copyright (c) 2023 FastTrackJustice

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"@
        
        $licenseContent | Out-File -FilePath $licensePath -Encoding UTF8
        Write-ColorOutput Yellow "Created license file: $licensePath"
    }
}

# Function to create .gitignore file
function Initialize-Gitignore {
    Write-ColorOutput Green "Creating .gitignore file..."
    
    $gitignorePath = "..\.gitignore"
    if (-not (Test-Path $gitignorePath)) {
        $gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
env/
.env

# Distribution / packaging
dist/
build/
*.egg-info/

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Jupyter Notebook
.ipynb_checkpoints

# VS Code
.vscode/

# PyCharm
.idea/

# FastTrackJustice specific
data/cases/*
data/principles/*
data/memory/*
logs/*
exports/*
!data/cases/.gitkeep
!data/principles/.gitkeep
!data/memory/.gitkeep
!logs/.gitkeep
!exports/.gitkeep
"@
        
        $gitignoreContent | Out-File -FilePath $gitignorePath -Encoding UTF8
        Write-ColorOutput Yellow "Created .gitignore file: $gitignorePath"
    }
}

# Function to create .gitkeep files
function Initialize-GitkeepFiles {
    Write-ColorOutput Green "Creating .gitkeep files..."
    
    $directories = @(
        "..\data\cases",
        "..\data\principles",
        "..\data\memory",
        "..\logs",
        "..\exports"
    )
    
    foreach ($dir in $directories) {
        $gitkeepPath = "$dir\.gitkeep"
        if (-not (Test-Path $gitkeepPath)) {
            "" | Out-File -FilePath $gitkeepPath -Encoding UTF8
            Write-ColorOutput Yellow "Created .gitkeep file: $gitkeepPath"
        }
    }
}

# Function to create sample data
function Initialize-SampleData {
    Write-ColorOutput Green "Creating sample data..."
    
    # Create sample case
    $sampleCasePath = "..\data\cases\sample_case.json"
    if (-not (Test-Path $sampleCasePath)) {
        $sampleCase = @{
            "id"         = "sample_case_001"
            "title"      = "Sample v. Example (2023)"
            "year"       = 2023
            "court"      = "Supreme Court"
            "text"       = "This is a sample case text for demonstration purposes. It contains legal principles that can be extracted and analyzed."
            "principles" = @(
                @{
                    "text"       = "Legal principles must be clearly stated"
                    "confidence" = 0.95
                },
                @{
                    "text"       = "Precedent should be followed when applicable"
                    "confidence" = 0.85
                }
            )
        } | ConvertTo-Json -Depth 10
        
        $sampleCase | Out-File -FilePath $sampleCasePath -Encoding UTF8
        Write-ColorOutput Yellow "Created sample case: $sampleCasePath"
    }
    
    # Create sample principle
    $samplePrinciplePath = "..\data\principles\sample_principle.json"
    if (-not (Test-Path $samplePrinciplePath)) {
        $samplePrinciple = @{
            "id"                 = "sample_principle_001"
            "text"               = "Legal principles must be clearly stated"
            "category"           = "procedural"
            "related_principles" = @("precedent", "clarity")
            "cases"              = @("sample_case_001")
        } | ConvertTo-Json -Depth 10
        
        $samplePrinciple | Out-File -FilePath $samplePrinciplePath -Encoding UTF8
        Write-ColorOutput Yellow "Created sample principle: $samplePrinciplePath"
    }
    
    # Create sample memory
    $sampleMemoryPath = "..\data\memory\sample_memory.json"
    if (-not (Test-Path $sampleMemoryPath)) {
        $sampleMemory = @{
            "entries" = @(
                @{
                    "id"                = "memory_001"
                    "case_id"           = "sample_case_001"
                    "principle_id"      = "sample_principle_001"
                    "scroll_alignment"  = "Justice shall be administered fairly and efficiently"
                    "prophetic_insight" = "This principle will guide the evolution of procedural fairness"
                    "confidence"        = 0.9
                    "tags"              = @("fairness", "efficiency", "procedure")
                    "timestamp"         = "2023-01-01T00:00:00Z"
                }
            )
        } | ConvertTo-Json -Depth 10
        
        $sampleMemory | Out-File -FilePath $sampleMemoryPath -Encoding UTF8
        Write-ColorOutput Yellow "Created sample memory: $sampleMemoryPath"
    }
}

# Function to create deployment options
function Show-DeploymentOptions {
    Write-ColorOutput Green "`nFastTrackJustice Deployment Options:`n"
    Write-ColorOutput Yellow "1. Full Deployment (Recommended)"
    Write-ColorOutput Yellow "2. Minimal Deployment (Core features only)"
    Write-ColorOutput Yellow "3. Custom Deployment (Select components)"
    Write-ColorOutput Yellow "4. Exit`n"
    
    $choice = Read-Host "Enter your choice (1-4)"
    
    switch ($choice) {
        "1" { 
            Write-ColorOutput Green "`nPerforming Full Deployment...`n"
            Initialize-VirtualEnvironment
            Install-Dependencies
            Initialize-Directories
            Initialize-Configuration
            Initialize-EnvironmentFile
            Initialize-Readme
            Initialize-License
            Initialize-Gitignore
            Initialize-GitkeepFiles
            Initialize-SampleData
        }
        "2" { 
            Write-ColorOutput Green "`nPerforming Minimal Deployment...`n"
            Initialize-VirtualEnvironment
            Install-Dependencies
            Initialize-Directories
            Initialize-Configuration
            Initialize-EnvironmentFile
        }
        "3" { 
            Write-ColorOutput Green "`nCustom Deployment Options:`n"
            Write-ColorOutput Yellow "a. Virtual Environment"
            Write-ColorOutput Yellow "b. Dependencies"
            Write-ColorOutput Yellow "c. Directories"
            Write-ColorOutput Yellow "d. Configuration"
            Write-ColorOutput Yellow "e. Environment File"
            Write-ColorOutput Yellow "f. README"
            Write-ColorOutput Yellow "g. License"
            Write-ColorOutput Yellow "h. .gitignore"
            Write-ColorOutput Yellow "i. .gitkeep Files"
            Write-ColorOutput Yellow "j. Sample Data"
            Write-ColorOutput Yellow "k. Exit`n"
            
            $customChoice = Read-Host "Enter your choices (comma-separated, e.g., a,b,c)"
            $choices = $customChoice -split ','
            
            foreach ($c in $choices) {
                switch ($c.Trim()) {
                    "a" { Initialize-VirtualEnvironment }
                    "b" { Install-Dependencies }
                    "c" { Initialize-Directories }
                    "d" { Initialize-Configuration }
                    "e" { Initialize-EnvironmentFile }
                    "f" { Initialize-Readme }
                    "g" { Initialize-License }
                    "h" { Initialize-Gitignore }
                    "i" { Initialize-GitkeepFiles }
                    "j" { Initialize-SampleData }
                    "k" { exit }
                    default { Write-ColorOutput Red "Invalid choice: $c" }
                }
            }
        }
        "4" { exit }
        default { 
            Write-ColorOutput Red "Invalid choice. Please try again."
            Show-DeploymentOptions
        }
    }
}

# Main execution
Write-ColorOutput Green "`n=== FastTrackJustice Deployment Script ===`n"

# Check if Python is installed
if (-not (Test-PythonVersion)) {
    Write-ColorOutput Red "Python is required but not found. Please install Python 3.8+ and try again."
    exit 1
}

# Show deployment options
Show-DeploymentOptions

# Final instructions
Write-ColorOutput Green "`nDeployment completed successfully!`n"
Write-ColorOutput Yellow "To run the FastTrackJustice Precedent Influence Graph Explorer:"
Write-ColorOutput Yellow "1. Activate the virtual environment: ..\venv\Scripts\Activate.ps1"
Write-ColorOutput Yellow "2. Run the Streamlit app: streamlit run backend\app\monitoring\precedent_graph_explorer.py"
Write-ColorOutput Yellow "3. Access the app at http://localhost:8501`n"
Write-ColorOutput Green "Thank you for deploying FastTrackJustice!" 
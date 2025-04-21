# PowerShell script to run the Precedent Influence Graph Explorer
# This script activates the virtual environment and launches the Streamlit app

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to write colorful messages
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# Function to check if a command exists
function Test-Command($cmd) {
    return [bool](Get-Command -Name $cmd -ErrorAction SilentlyContinue)
}

# Check if Python is installed
if (-not (Test-Command python)) {
    Write-ColorOutput Red "Python is not installed or not in PATH. Please install Python 3.8+ and try again."
    exit 1
}

# Check if virtual environment exists
$venvPath = "..\venv"
if (-not (Test-Path $venvPath)) {
    Write-ColorOutput Yellow "Virtual environment not found. Creating one..."
    python -m venv $venvPath
    if (-not $?) {
        Write-ColorOutput Red "Failed to create virtual environment."
        exit 1
    }
}

# Activate virtual environment
Write-ColorOutput Cyan "Activating virtual environment..."
& "$venvPath\Scripts\Activate.ps1"
if (-not $?) {
    Write-ColorOutput Red "Failed to activate virtual environment."
    exit 1
}

# Check if required packages are installed
Write-ColorOutput Cyan "Checking required packages..."
$requiredPackages = @("streamlit", "pandas", "networkx", "plotly")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    if (-not (Test-Command $package)) {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-ColorOutput Yellow "Installing missing packages: $($missingPackages -join ', ')"
    pip install $missingPackages
    if (-not $?) {
        Write-ColorOutput Red "Failed to install required packages."
        exit 1
    }
}

# Run the Streamlit app
Write-ColorOutput Green "Starting Precedent Influence Graph Explorer..."
Write-ColorOutput Cyan "The app will be available at http://localhost:8501"
Write-ColorOutput Cyan "Press Ctrl+C to stop the app"

# Run the Python script
python run_graph_explorer.py

# Deactivate virtual environment
deactivate 
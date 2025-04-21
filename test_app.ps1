# PowerShell script to test FastTrackJusticeAI

# Set the Python virtual environment path
$venvPath = ".\venv311\Scripts\activate"

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& $venvPath

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Run tests
Write-Host "Running tests..." -ForegroundColor Green
cd backend
python -m pytest tests/ -v

# Check if tests passed
if ($LASTEXITCODE -eq 0) {
    Write-Host "All tests passed!" -ForegroundColor Green
}
else {
    Write-Host "Some tests failed. Please fix the issues before pushing to GitHub." -ForegroundColor Red
    exit 1
}

# Run the app
Write-Host "Starting the app..." -ForegroundColor Green
cd ..
python backend/app.py

Write-Host "Done!" -ForegroundColor Green 
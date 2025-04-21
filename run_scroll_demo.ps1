# PowerShell script to run the scroll judgment demonstration suite

# Set the Python virtual environment path
$venvPath = ".\venv311\Scripts\activate"

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& $venvPath

# Install dependencies if needed
Write-Host "Checking dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Green
$directories = @("examples", "reports", "logs")
foreach ($dir in $directories) {
    if (-not (Test-Path -Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Yellow
    }
}

# Run the scroll judgment demonstration
Write-Host "Running scroll judgment demonstration..." -ForegroundColor Green
python scripts/run_scroll_demo.py

# Check if the demonstration completed successfully
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nScroll judgment demonstration completed successfully!" -ForegroundColor Green
    Write-Host "`nGenerated files:" -ForegroundColor Cyan
    
    # List generated judgments
    $judgmentCount = (Get-ChildItem -Path "examples" -Filter "*.txt").Count
    Write-Host "- $judgmentCount judgments in the 'examples' directory" -ForegroundColor Cyan
    
    # List generated reports
    if (Test-Path -Path "reports\scroll_judgment_analysis.md") {
        Write-Host "- Analysis report: reports\scroll_judgment_analysis.md" -ForegroundColor Cyan
    }
    
    if (Test-Path -Path "reports\scroll_judgment_analysis.csv") {
        Write-Host "- Analysis data: reports\scroll_judgment_analysis.csv" -ForegroundColor Cyan
    }
    
    # List generated visualizations
    $visualizationCount = (Get-ChildItem -Path "reports\visualizations" -Filter "*.png").Count
    if ($visualizationCount -gt 0) {
        Write-Host "- $visualizationCount visualizations in the 'reports\visualizations' directory" -ForegroundColor Cyan
    }
    
    # List log files
    $logCount = (Get-ChildItem -Path "logs" -Filter "*.log").Count
    if ($logCount -gt 0) {
        Write-Host "- $logCount log files in the 'logs' directory" -ForegroundColor Cyan
    }
    
    Write-Host "`nYou can now push these results to GitHub using push_to_github.ps1" -ForegroundColor Green
}
else {
    Write-Host "`nScroll judgment demonstration failed with exit code $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Check the logs for more information." -ForegroundColor Red
}

Write-Host "`nDone!" -ForegroundColor Green 
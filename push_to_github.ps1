# PowerShell script to push the FastTrackJusticeAI project to GitHub

# Configuration
$repoName = "FastTrackJusticeAI"
$repoDescription = "A scroll-aware legal AI system that generates judgments aligned with the sacred scroll phases"
$repoUrl = "https://github.com/stanleymay20/$repoName.git"

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "Git is installed: $gitVersion" -ForegroundColor Green
}
catch {
    Write-Host "Git is not installed. Please install Git and try again." -ForegroundColor Red
    exit 1
}

# Check if Git user identity is configured
$userName = git config --get user.name
$userEmail = git config --get user.email

if (-not $userName -or -not $userEmail) {
    Write-Host "Git user identity not configured. Please enter your information:" -ForegroundColor Yellow
    
    $userName = Read-Host "Enter your name (e.g., John Doe)"
    $userEmail = Read-Host "Enter your email (e.g., john.doe@example.com)"
    
    if ($userName -and $userEmail) {
        git config --global user.name "$userName"
        git config --global user.email "$userEmail"
        Write-Host "Git user identity configured successfully." -ForegroundColor Green
    }
    else {
        Write-Host "Git user identity configuration failed. Please try again." -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "Git user identity already configured: $userName <$userEmail>" -ForegroundColor Green
}

# Configure Git to handle line endings
Write-Host "Configuring Git to handle line endings..." -ForegroundColor Green
git config --global core.autocrlf true

# Remove existing .git directory if it exists
if (Test-Path -Path ".git") {
    Write-Host "Removing existing Git repository..." -ForegroundColor Yellow
    Remove-Item -Path ".git" -Recurse -Force
}

# Initialize fresh Git repository
Write-Host "Initializing fresh Git repository..." -ForegroundColor Green
git init

# Create .gitignore file
Write-Host "Creating .gitignore file..." -ForegroundColor Green
@"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
venv311/
ENV/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
logs/
*.log

# Local configuration
.env
.env.local

# Generated files (optional, comment out if you want to include these)
# examples/
# reports/
"@ | Out-File -FilePath ".gitignore" -Encoding utf8

# Add files individually to avoid nested directory issues
Write-Host "Adding project files..." -ForegroundColor Green

# List of files and directories to add
$filesToAdd = @(
    ".gitignore",
    "README.md",
    "requirements.txt",
    "push_to_github.ps1",
    "run_scroll_demo.ps1",
    "test_app.ps1"
)

# List of directories to add
$dirsToAdd = @(
    "app",
    "backend",
    "scripts"
)

# Add individual files
foreach ($file in $filesToAdd) {
    if (Test-Path -Path $file) {
        Write-Host "Adding file: $file" -ForegroundColor Cyan
        git add $file
    }
}

# Add directories
foreach ($dir in $dirsToAdd) {
    if (Test-Path -Path $dir) {
        Write-Host "Adding directory: $dir" -ForegroundColor Cyan
        git add $dir
    }
}

# Commit changes
Write-Host "Committing changes..." -ForegroundColor Green
$commitMessage = "Initial commit: FastTrackJusticeAI - A scroll-aware legal AI system"
git commit -m $commitMessage

# Check if commit was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to commit changes. Please check the error message above." -ForegroundColor Red
    exit 1
}

# Remove existing origin if it exists
git remote remove origin

# Add remote repository
Write-Host "Adding remote repository..." -ForegroundColor Green
git remote add origin $repoUrl

# Force push to master branch
Write-Host "Force pushing to GitHub..." -ForegroundColor Green
Write-Host "Note: You may need to enter your GitHub credentials when prompted." -ForegroundColor Yellow
git push -f origin master

# Check if push was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSuccessfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
    
    # Open the repository in the browser
    Write-Host "Opening repository in browser..." -ForegroundColor Green
    Start-Process $repoUrl
}
else {
    Write-Host "`nFailed to push to GitHub. Please check the error message above." -ForegroundColor Red
    Write-Host "You may need to:" -ForegroundColor Yellow
    Write-Host "1. Create the repository manually on GitHub first" -ForegroundColor Yellow
    Write-Host "2. Ensure you have the correct permissions" -ForegroundColor Yellow
    Write-Host "3. Check your GitHub credentials" -ForegroundColor Yellow
}

Write-Host "`nDone!" -ForegroundColor Green 
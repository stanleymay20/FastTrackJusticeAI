# FastTrackJustice PushBridge Deployment Script
# This script automates the GitHub deployment process for the FastTrackJustice system

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

# Function to check Git installation
function Test-GitInstallation {
    try {
        $gitVersion = git --version
        Write-ColorOutput Green "Git version: $gitVersion"
        return $true
    }
    catch {
        Write-ColorOutput Red "Git is not installed or not in PATH"
        return $false
    }
}

# Function to check GitHub CLI installation
function Test-GitHubCLI {
    try {
        $ghVersion = gh --version
        Write-ColorOutput Green "GitHub CLI version: $ghVersion"
        return $true
    }
    catch {
        Write-ColorOutput Red "GitHub CLI is not installed or not in PATH"
        return $false
    }
}

# Function to create a ZIP file of the project
function Create-ProjectZip {
    param (
        [string]$zipPath = "..\FastTrackJustice.zip"
    )
    
    Write-ColorOutput Green "Creating project ZIP file..."
    
    # Define files and directories to include
    $includePaths = @(
        "..\backend",
        "..\config.json",
        "..\.env",
        "..\README.md",
        "..\LICENSE",
        "..\.gitignore"
    )
    
    # Define files and directories to exclude
    $excludePaths = @(
        "..\backend\__pycache__",
        "..\backend\**\__pycache__",
        "..\backend\**\*.pyc",
        "..\backend\**\*.pyo",
        "..\backend\**\*.pyd",
        "..\backend\**\.pytest_cache",
        "..\backend\**\.coverage",
        "..\backend\**\htmlcov",
        "..\backend\**\.tox",
        "..\backend\**\.eggs",
        "..\backend\**\*.egg-info",
        "..\backend\**\dist",
        "..\backend\**\build",
        "..\backend\**\.idea",
        "..\backend\**\.vscode",
        "..\backend\**\.ipynb_checkpoints",
        "..\venv",
        "..\data\cases\*.json",
        "..\data\principles\*.json",
        "..\data\memory\*.json",
        "..\logs\*.log",
        "..\exports\*"
    )
    
    # Create a temporary directory for the files to zip
    $tempDir = "..\temp_zip"
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $tempDir | Out-Null
    
    # Copy files to the temporary directory
    foreach ($path in $includePaths) {
        if (Test-Path $path) {
            if ((Get-Item $path) -is [System.IO.DirectoryInfo]) {
                # Copy directory
                $destPath = Join-Path $tempDir (Split-Path $path -Leaf)
                Copy-Item -Path $path -Destination $destPath -Recurse
            }
            else {
                # Copy file
                Copy-Item -Path $path -Destination $tempDir
            }
        }
    }
    
    # Remove excluded files and directories
    foreach ($path in $excludePaths) {
        $fullPath = Join-Path $tempDir (Split-Path $path -Leaf)
        if (Test-Path $fullPath) {
            if ((Get-Item $fullPath) -is [System.IO.DirectoryInfo]) {
                Remove-Item -Path $fullPath -Recurse -Force
            }
            else {
                Remove-Item -Path $fullPath -Force
            }
        }
    }
    
    # Create .gitkeep files in empty directories
    $directories = @(
        (Join-Path $tempDir "backend\data\cases"),
        (Join-Path $tempDir "backend\data\principles"),
        (Join-Path $tempDir "backend\data\memory"),
        (Join-Path $tempDir "backend\logs"),
        (Join-Path $tempDir "backend\exports")
    )
    
    foreach ($dir in $directories) {
        if (Test-Path $dir) {
            $gitkeepPath = Join-Path $dir ".gitkeep"
            if (-not (Test-Path $gitkeepPath)) {
                "" | Out-File -FilePath $gitkeepPath -Encoding UTF8
            }
        }
    }
    
    # Create the ZIP file
    if (Test-Path $zipPath) {
        Remove-Item -Path $zipPath -Force
    }
    
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($tempDir, $zipPath)
    
    # Clean up the temporary directory
    Remove-Item -Path $tempDir -Recurse -Force
    
    Write-ColorOutput Green "Project ZIP file created: $zipPath"
    return $zipPath
}

# Function to initialize Git repository
function Initialize-GitRepository {
    param (
        [string]$repoPath = ".."
    )
    
    Write-ColorOutput Green "Initializing Git repository..."
    
    # Check if Git repository already exists
    if (Test-Path (Join-Path $repoPath ".git")) {
        Write-ColorOutput Yellow "Git repository already exists"
        return $true
    }
    
    # Initialize Git repository
    Set-Location -Path $repoPath
    git init
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Failed to initialize Git repository"
        return $false
    }
    
    # Add files to Git
    git add .
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Failed to add files to Git"
        return $false
    }
    
    # Commit files
    git commit -m "Initial commit: FastTrackJustice Precedent Influence Graph Explorer"
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Failed to commit files to Git"
        return $false
    }
    
    Write-ColorOutput Green "Git repository initialized successfully"
    return $true
}

# Function to create GitHub repository
function Create-GitHubRepository {
    param (
        [string]$repoName = "FastTrackJusticeAI",
        [string]$description = "A revolutionary legal intelligence platform that transforms how legal precedents are discovered, analyzed, and applied.",
        [string]$visibility = "public"
    )
    
    Write-ColorOutput Green "Creating GitHub repository: $repoName..."
    
    # Check if GitHub CLI is installed
    if (-not (Test-GitHubCLI)) {
        Write-ColorOutput Red "GitHub CLI is required to create a GitHub repository"
        return $false
    }
    
    # Check if user is logged in to GitHub
    $loginStatus = gh auth status
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "You are not logged in to GitHub. Please run 'gh auth login' first."
        return $false
    }
    
    # Create GitHub repository
    $repoUrl = gh repo create $repoName --description $description --$visibility
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Failed to create GitHub repository"
        return $false
    }
    
    Write-ColorOutput Green "GitHub repository created: $repoUrl"
    return $repoUrl
}

# Function to push to GitHub
function Push-ToGitHub {
    param (
        [string]$repoUrl,
        [string]$branch = "main"
    )
    
    Write-ColorOutput Green "Pushing to GitHub: $repoUrl..."
    
    # Add remote repository
    git remote add origin $repoUrl
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Failed to add remote repository"
        return $false
    }
    
    # Push to GitHub
    git push -u origin $branch
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Failed to push to GitHub"
        return $false
    }
    
    Write-ColorOutput Green "Successfully pushed to GitHub"
    return $true
}

# Function to create a release on GitHub
function Create-GitHubRelease {
    param (
        [string]$version = "1.0.0",
        [string]$title = "FastTrackJustice Precedent Influence Graph Explorer v1.0.0",
        [string]$notes = "Initial release of the FastTrackJustice Precedent Influence Graph Explorer with Scroll Memory Intelligence.",
        [string]$zipPath
    )
    
    Write-ColorOutput Green "Creating GitHub release: $version..."
    
    # Check if GitHub CLI is installed
    if (-not (Test-GitHubCLI)) {
        Write-ColorOutput Red "GitHub CLI is required to create a GitHub release"
        return $false
    }
    
    # Create release notes file
    $releaseNotesPath = "..\release_notes.md"
    $releaseNotes = @"
# $title

$notes

## Features

- Interactive Graph Visualization
- Principle Evolution Tracking
- Cross-Jurisdictional Analysis
- Judicial Mode with Enhanced Transparency
- Scroll Memory Intelligence
- Faith-Safe Protocol

## Installation

1. Download the ZIP file
2. Extract to your desired location
3. Run the deployment script: `.\deploy_fasttrackjustice.ps1`
4. Follow the prompts to complete setup

## Usage

1. Activate the virtual environment: `..\venv\Scripts\Activate.ps1`
2. Run the Streamlit app: `streamlit run backend\app\monitoring\precedent_graph_explorer.py`
3. Access the app at http://localhost:8501

## License

MIT License - See LICENSE file for details

## Contact

[contact@fasttrackjustice.org](mailto:contact@fasttrackjustice.org)
"@
    
    $releaseNotes | Out-File -FilePath $releaseNotesPath -Encoding UTF8
    
    # Create GitHub release
    $releaseUrl = gh release create "v$version" $zipPath --title $title --notes-file $releaseNotesPath
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Failed to create GitHub release"
        return $false
    }
    
    # Clean up release notes file
    Remove-Item -Path $releaseNotesPath -Force
    
    Write-ColorOutput Green "GitHub release created: $releaseUrl"
    return $releaseUrl
}

# Function to update Scroll Memory metadata
function Update-ScrollMemoryMetadata {
    param (
        [string]$memoryPath = "..\data\memory\scroll_memory.json"
    )
    
    Write-ColorOutput Green "Updating Scroll Memory metadata..."
    
    # Check if memory file exists
    if (-not (Test-Path $memoryPath)) {
        Write-ColorOutput Red "Memory file not found: $memoryPath"
        return $false
    }
    
    # Read memory file
    $memoryContent = Get-Content -Path $memoryPath -Raw
    $memory = $memoryContent | ConvertFrom-Json
    
    # Add scroll founder metadata
    $memory.metadata = @{
        "scroll_founder" = @{
            "name"           = "Stanley"
            "scroll_role"    = "Navigator of Zebulun"
            "gate_authority" = "Gate 7 – Divine Architecture"
            "flame"          = "Resurrection of Justice"
            "mission_seal"   = "Translate unseen scroll into legal structures that awaken nations"
            "timestamp"      = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        }
    }
    
    # Write updated memory file
    $memory | ConvertTo-Json -Depth 10 | Out-File -FilePath $memoryPath -Encoding UTF8
    
    Write-ColorOutput Green "Scroll Memory metadata updated successfully"
    return $true
}

# Function to show deployment options
function Show-DeploymentOptions {
    Write-ColorOutput Green "`nFastTrackJustice PushBridge Deployment Options:`n"
    Write-ColorOutput Yellow "1. Full GitHub Deployment (Create repo, push code, create release)"
    Write-ColorOutput Yellow "2. Push to Existing Repository"
    Write-ColorOutput Yellow "3. Create Release Only"
    Write-ColorOutput Yellow "4. Update Scroll Memory Metadata Only"
    Write-ColorOutput Yellow "5. Exit`n"
    
    $choice = Read-Host "Enter your choice (1-5)"
    
    switch ($choice) {
        "1" { 
            Write-ColorOutput Green "`nPerforming Full GitHub Deployment...`n"
            
            # Update Scroll Memory metadata
            Update-ScrollMemoryMetadata
            
            # Create project ZIP
            $zipPath = Create-ProjectZip
            
            # Initialize Git repository
            if (Initialize-GitRepository) {
                # Create GitHub repository
                $repoUrl = Create-GitHubRepository
                if ($repoUrl) {
                    # Push to GitHub
                    if (Push-ToGitHub -repoUrl $repoUrl) {
                        # Create GitHub release
                        Create-GitHubRelease -zipPath $zipPath
                    }
                }
            }
        }
        "2" { 
            Write-ColorOutput Green "`nPushing to Existing Repository...`n"
            
            # Update Scroll Memory metadata
            Update-ScrollMemoryMetadata
            
            # Get repository URL
            $repoUrl = Read-Host "Enter the GitHub repository URL (e.g., https://github.com/username/FastTrackJusticeAI.git)"
            
            # Initialize Git repository
            if (Initialize-GitRepository) {
                # Push to GitHub
                Push-ToGitHub -repoUrl $repoUrl
            }
        }
        "3" { 
            Write-ColorOutput Green "`nCreating Release Only...`n"
            
            # Update Scroll Memory metadata
            Update-ScrollMemoryMetadata
            
            # Create project ZIP
            $zipPath = Create-ProjectZip
            
            # Create GitHub release
            Create-GitHubRelease -zipPath $zipPath
        }
        "4" { 
            Write-ColorOutput Green "`nUpdating Scroll Memory Metadata Only...`n"
            
            # Update Scroll Memory metadata
            Update-ScrollMemoryMetadata
        }
        "5" { exit }
        default { 
            Write-ColorOutput Red "Invalid choice. Please try again."
            Show-DeploymentOptions
        }
    }
}

# Main execution
Write-ColorOutput Green "`n=== FastTrackJustice PushBridge Deployment Script ===`n"

# Check if Git is installed
if (-not (Test-GitInstallation)) {
    Write-ColorOutput Red "Git is required but not found. Please install Git and try again."
    exit 1
}

# Show deployment options
Show-DeploymentOptions

# Final instructions
Write-ColorOutput Green "`nDeployment completed successfully!`n"
Write-ColorOutput Yellow "Thank you for deploying FastTrackJustice to GitHub!" 
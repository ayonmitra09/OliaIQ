# Check if Poetry is installed
if (!(Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Poetry..."
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
}

# Initialize Poetry if pyproject.toml doesn't exist
if (!(Test-Path pyproject.toml)) {
    Write-Host "Initializing Poetry project..."
    poetry init --no-interaction
}

# Install dependencies
Write-Host "Installing dependencies..."
poetry install

# Create virtual environment if it doesn't exist
if (!(Test-Path .venv)) {
    Write-Host "Creating virtual environment..."
    poetry env use python
}

Write-Host "Setup complete! You can now run:"
Write-Host "poetry shell - to activate the virtual environment"
Write-Host "poetry run python src/test_temperature_analysis.py - to run the temperature analysis" 
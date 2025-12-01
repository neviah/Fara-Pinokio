Write-Host "[Fara Install] Starting installation..."
$ErrorActionPreference = "Stop"

function Fail($msg) {
  Write-Host "[Fara Install] ERROR: $msg" -ForegroundColor Red
  exit 1
}

# 1. Create virtual environment if missing
if (!(Test-Path -Path "env")) {
  Write-Host "[Fara Install] Creating virtual environment..."
  python -m venv env || Fail "Failed to create virtual environment"
} else {
  Write-Host "[Fara Install] Virtual environment already exists. Skipping."
}

$venvPython = "env/Scripts/python.exe"
$venvPip = "env/Scripts/pip.exe"
$playwright = "env/Scripts/playwright.cmd"

if (!(Test-Path $venvPython)) { Fail "Virtual environment python not found" }

# 2. Upgrade pip
Write-Host "[Fara Install] Upgrading pip..."
& $venvPython -m pip install --upgrade pip || Fail "Failed to upgrade pip"

# 3. Base requirements
Write-Host "[Fara Install] Installing requirements.txt (if present)..."
if (Test-Path -Path "requirements.txt") {
  & $venvPip install -r requirements.txt || Fail "Failed to install requirements.txt"
} else {
  Write-Host "[Fara Install] requirements.txt not found, skipping."
}

# 4. Extra packages (ensure playwright)
Write-Host "[Fara Install] Installing playwright + openai + pillow + tenacity + numpy + requests"
& $venvPip install playwright openai pillow tenacity numpy requests || Fail "Failed to install core packages"

# 5. Clone Fara repo if missing
if (!(Test-Path -Path "fara_repo")) {
  Write-Host "[Fara Install] Cloning microsoft/fara..."
  git clone https://github.com/microsoft/fara.git fara_repo || Fail "Failed to clone Fara repo"
} else {
  Write-Host "[Fara Install] fara_repo already present. Skipping clone."
}

# 6. Editable install of Fara
Write-Host "[Fara Install] Installing Fara (editable)..."
& $venvPip install -e ./fara_repo || Fail "Failed to install Fara editable"

# 7. Install Playwright browsers
Write-Host "[Fara Install] Installing Playwright browsers..."
& $playwright install || Fail "Failed to install Playwright browsers"

Write-Host "[Fara Install] Completed successfully." -ForegroundColor Green

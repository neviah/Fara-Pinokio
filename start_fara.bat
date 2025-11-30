@echo off
echo Starting Fara-7B Gradio Interface...
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Please run install.js first.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate

:: Check if Fara is installed
python -c "import fara" 2>nul
if errorlevel 1 (
    echo Fara package not found. Please run install.js first.
    pause
    exit /b 1
)

:: Start the Gradio interface
echo.
echo 🚀 Starting Fara-7B Computer Use Agent...
echo 🌐 Interface will be available at: http://localhost:7860
echo 💡 Make sure you have a model server running or Azure Foundry configured
echo.

python gradio_interface.py

pause
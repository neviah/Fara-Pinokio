#!/bin/bash
echo "Starting Fara-7B Gradio Interface..."
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run install.js first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Fara is installed
python3 -c "import fara" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Fara package not found. Please run install.js first."
    exit 1
fi

# Start the Gradio interface
echo ""
echo "🚀 Starting Fara-7B Computer Use Agent..."
echo "🌐 Interface will be available at: http://localhost:7860"
echo "💡 Make sure you have a model server running or Azure Foundry configured"
echo ""

python3 gradio_interface.py
# 🤖 Fara-7B Computer Use Agent - Gradio Interface

A beautiful web interface for Microsoft's Fara-7B, the efficient 7B parameter agentic model designed for computer use and web automation.

![Fara-7B Interface](https://img.shields.io/badge/Fara--7B-Computer%20Use%20Agent-blue) ![Gradio](https://img.shields.io/badge/Gradio-Interface-orange) ![Pinokio](https://img.shields.io/badge/Pinokio-Compatible-green)

## 🌟 Features

- 🎯 **Easy-to-use Gradio Interface**: Beautiful web UI for interacting with Fara-7B
- 🤖 **Microsoft's Fara-7B**: State-of-the-art 7B parameter computer use agent
- 🔧 **Flexible Hosting**: Supports both local VLLM and Azure Foundry deployment
- 📊 **Task Monitoring**: Real-time progress tracking and history
- 🎨 **Modern Design**: Responsive interface with dark/light theme support
- 🚀 **Pinokio Compatible**: One-click installation and deployment
- 🔒 **Privacy-First**: Run locally for complete data privacy

## 🎬 What Fara-7B Can Do

Fara-7B can automate a wide variety of web tasks:

- 🔍 **Search & Research**: Find information and summarize results
- 📝 **Form Filling**: Complete forms and manage accounts
- 🎫 **Booking Services**: Book travel, hotels, movie tickets, restaurants
- 🛒 **Shopping**: Compare prices, add items to cart, checkout
- 💼 **Job Hunting**: Find job postings and real estate listings
- 📊 **Data Collection**: Extract information from websites
- 🗂️ **Account Management**: Login, navigate, and manage web accounts

## 🚀 Quick Start with Pinokio

### Method 1: Direct Pinokio Installation (Recommended)

1. **Install Pinokio** from [pinokio.computer](https://pinokio.computer)

2. **Add this repository** to Pinokio:
   ```
   https://github.com/your-username/fara-gradio-interface
   ```

3. **Click Install** - Pinokio will automatically:
   - Clone the Microsoft Fara repository
   - Set up Python virtual environment
   - Install all dependencies
   - Download Playwright browsers
   - Create the Gradio interface

4. **Click Run** to start the interface

5. **Configure** your model endpoint and start automating!

### Method 2: Manual Installation

If you prefer manual setup:

```bash
# Clone this repository
git clone https://github.com/your-username/fara-gradio-interface
cd fara-gradio-interface

# Install using the provided script
node install.js

# Start the interface
python gradio_interface.py
```

## ⚙️ Configuration

### LM Studio Setup (Easiest for Local Testing) ⭐

**LM Studio** provides the easiest way to run Fara-7B locally with a user-friendly GUI!

1. **Download and Install** [LM Studio](https://lmstudio.ai/)

2. **Download Fara-7B**:
   - Open LM Studio
   - Go to the "Discover" tab
   - Search for `microsoft/Fara-7B`
   - Click Download (choose FP16 for best quality, or INT4/INT8 for lower VRAM)

3. **Start the Local Server**:
   - Go to the "Local Server" tab in LM Studio
   - Select the Fara-7B model you downloaded
   - Click "Start Server"
   - Default port is `1234` (OpenAI-compatible endpoint)

4. **Configure in the Gradio interface**:
   - Model Endpoint: `http://localhost:1234/v1`
   - API Key: `lm-studio` (or any value, it's not validated locally)
   - Click "Save Configuration"

5. **Start automating!** The interface will use your local LM Studio server.

**Benefits of LM Studio:**
- ✅ Easy GUI for model management
- ✅ No command-line needed
- ✅ Built-in model quantization options
- ✅ OpenAI-compatible API (works with existing code)
- ✅ GPU acceleration automatically configured

### Local VLLM Setup (Advanced Users)

1. **Install VLLM**:
   ```bash
   pip install vllm
   ```

2. **Start the model server**:
   ```bash
   vllm serve "microsoft/Fara-7B" --port 5000 --dtype auto
   ```

3. **Configure in the interface**:
   - Model Name: `microsoft/Fara-7B`
   - Base URL: `http://localhost:5000/v1`
   - API Key: leave empty

### Azure Foundry Setup (Recommended for Production)

1. **Deploy Fara-7B** on [Azure Foundry](https://ai.azure.com/explore/models/Fara-7B/version/2/registry/azureml-msr)

2. **Get your endpoint details** from the deployment

3. **Configure in the interface**:
   - Model Name: `Fara-7B`
   - Base URL: `https://your-endpoint.inference.ml.azure.com/`
   - API Key: `your-api-key-here`

### BrowserBase Setup (Optional)

For cloud browser management, set these environment variables:
```bash
export BROWSERBASE_API_KEY="your-api-key"
export BROWSERBASE_PROJECT_ID="your-project-id"
```

## 🖥️ Interface Guide

### Main Interface Tabs

1. **🚀 Run Tasks**: Main interface for running automation tasks
   - Enter task description
   - Monitor real-time progress
   - View results and actions taken

2. **⚙️ Configuration**: Set up model endpoints
   - Configure local VLLM or Azure Foundry
   - Test connection
   - View setup instructions

3. **📊 History**: View previous tasks
   - See task results and timestamps
   - Review successful and failed attempts

4. **ℹ️ About**: Learn about Fara-7B and the interface

### Example Tasks

Try these example tasks to get started:

- `"How many pages does Wikipedia have?"`
- `"Search for the weather in New York City"`
- `"Find the latest iPhone price on Apple's website"`
- `"Search for job openings for Python developers in Seattle"`
- `"Find a hotel in Paris for next month"`

## 🎯 How It Works

1. **Visual Understanding**: Fara-7B takes screenshots of web pages and understands the visual layout

2. **Action Planning**: The model decides what actions to take (click, type, scroll, navigate)

3. **Coordinate Prediction**: Actions are executed at precise coordinates on the page

4. **Task Completion**: The agent continues until the task is completed or the maximum rounds are reached

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM
- **Storage**: 5GB free space for dependencies + 14GB for model weights
- **Internet**: For model download and web browsing

### GPU Requirements (for Local Model Hosting)

#### **Minimum VRAM** (4-6 GB)
- Supports quantized inference (INT8/INT4)
- Runs slower, may require CPU offloading
- Example GPUs: RTX 3050, GTX 1660 Ti, RTX 2060

#### **Recommended VRAM** (8-12 GB) ⭐
- Full FP16 precision
- Smooth inference performance
- Example GPUs: RTX 3060 12GB, RTX 4060 Ti, RTX 3080 10GB

#### **Optimal VRAM** (16+ GB)
- Fast inference with large context windows
- Room for batch processing
- Example GPUs: RTX 4080, RTX 4090, RTX A5000, A6000

#### **CPU Fallback** (No GPU)
- Automatically uses CPU-only PyTorch if no NVIDIA GPU detected
- Runs 10-30x slower than GPU
- Functional for testing and light usage
- Requires 16GB+ system RAM recommended

### Other Recommended Specs
- **Memory**: 16GB+ RAM (32GB for CPU-only mode)
- **CPU**: Multi-core processor (8+ cores recommended for CPU mode)
- **Internet**: Stable broadband connection

## 🛠️ Troubleshooting

### Common Issues

**"Agent not initialized"**
- Make sure you've configured a valid model endpoint
- Check that your VLLM server is running (for local setup)
- Verify your API credentials (for Azure Foundry)

**"Failed to install Playwright browsers"**
- Run manually: `python -m playwright install`
- Check your internet connection
- Try running with admin/sudo privileges

**"Model timeout errors"**
- Increase timeout in configuration
- Check model server load
- Ensure sufficient GPU memory

**"Browser automation errors"**
- Some websites have anti-automation protection
- Try different starting pages
- Check if the website is accessible

### Getting Help

- 📖 [Official Fara Documentation](https://github.com/microsoft/fara)
- 💬 [Pinokio Discord](https://discord.gg/pinokio)
- 🐛 [Report Issues](https://github.com/microsoft/fara/issues)

## 🔧 Advanced Configuration

### Custom Model Endpoints

You can use any OpenAI-compatible API endpoint:

```json
{
  "model": "your-model-name",
  "base_url": "https://your-api-endpoint.com/v1",
  "api_key": "your-api-key"
}
```

### Environment Variables

Configure via environment variables:

```bash
export FARA_MODEL_ENDPOINT="http://localhost:5000/v1"
export FARA_API_KEY="your-key"
export FARA_MAX_ROUNDS="50"
export FARA_HEADLESS="true"
```

### Browser Settings

Customize browser behavior by editing the browser configuration in `gradio_interface.py`:

```python
browser_manager = BrowserBB(
    headless=True,           # Run headless
    viewport_height=900,     # Browser height
    viewport_width=1440,     # Browser width
    animate_actions=False,   # Animate clicks
    single_tab_mode=True,    # Use single tab
)
```

## 🤝 Contributing

Contributions are welcome! This interface builds on Microsoft's excellent Fara project.

1. Fork this repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The underlying Fara-7B model is licensed under Microsoft's terms. See the [official repository](https://github.com/microsoft/fara) for details.

## 🙏 Acknowledgments

- **Microsoft Research** for developing the amazing Fara-7B model
- **Gradio Team** for the excellent web interface framework
- **Pinokio Community** for the one-click deployment platform
- **Open Source Contributors** who make projects like this possible

## 📊 Performance Notes

- **Speed**: Fara-7B averages ~16 steps per task vs ~41 for comparable models
- **Efficiency**: 7B parameters vs much larger competing agents
- **Accuracy**: State-of-the-art performance in its size class
- **Privacy**: Can run entirely on-device with local hosting

## 🔮 Future Plans

- [ ] Multi-language support
- [ ] Custom task templates
- [ ] Batch task processing
- [ ] Advanced scheduling
- [ ] Integration with more model providers
- [ ] Mobile-responsive improvements

---

**Ready to automate your web tasks? Get started with Pinokio today!** 🚀
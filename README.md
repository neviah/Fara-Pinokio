# 🤖 Fara-7B Computer Use Agent - Gradio Interface

A beautiful web interface for Microsoft's Fara-7B, the efficient 7B parameter agentic model designed for computer use and web automation.

![Fara-7B Interface](https://img.shields.io/badge/Fara--7B-Computer%20Use%20Agent-blue) ![Gradio](https://img.shields.io/badge/Gradio-Interface-orange) ![Pinokio](https://img.shields.io/badge/Pinokio-Compatible-green)

## ⚠️ **IMPORTANT: You Must Run Fara-7B Locally**

**This interface requires the Fara-7B model to work.** You cannot use OpenAI's ChatGPT or other cloud LLMs.

**You have 3 options to run Fara-7B:**

1. **LM Studio** (Easiest - Recommended) ⭐ - User-friendly GUI, no command line needed
2. **VLLM** (Advanced) - Command line setup for power users
3. **Azure Foundry** (Cloud) - Deploy Fara-7B on Azure (paid)

See the **[Configuration](#️-configuration)** section below for detailed setup instructions.

---

## 🌟 Features

- 🎯 **Easy-to-use Gradio Interface**: Beautiful web UI for interacting with Fara-7B
- 🤖 **Microsoft's Fara-7B**: State-of-the-art 7B parameter computer use agent  
- 🏠 **Run Locally**: Complete privacy with on-device model hosting
- 📊 **Task Monitoring**: Real-time progress tracking
- 🎨 **Modern Design**: Responsive interface with dark/light theme support
- 🚀 **Pinokio Compatible**: One-click installation and deployment
- 🔒 **Privacy-First**: All processing happens on your machine

## 🎬 What Fara-7B Can Do

Fara-7B can automate a wide variety of web tasks:

- 🔍 **Search & Research**: Find information and summarize results
- 📝 **Form Filling**: Complete forms and manage accounts
- 🎫 **Booking Services**: Book travel, hotels, movie tickets, restaurants
- 🛒 **Shopping**: Compare prices, add items to cart, checkout
- 💼 **Job Hunting**: Find job postings and real estate listings
- 📊 **Data Collection**: Extract information from websites
- 🗂️ **Account Management**: Login, navigate, and manage web accounts

---

## 🚀 Quick Start with Pinokio

### Step 1: Install the Interface

1. **Install Pinokio** from [pinokio.computer](https://pinokio.computer)

2. **Add this repository** to Pinokio:
   ```
   https://github.com/neviah/Fara-Pinokio
   ```

3. **Click Install** - Pinokio will automatically:
   - Set up Python virtual environment
   - Install all dependencies (Gradio, Playwright, etc.)
   - Download required browser automation tools
   - Create the Gradio interface

4. **Click Start** to launch the interface

### Step 2: Set Up Fara-7B Model Server

**⚠️ REQUIRED**: The interface alone won't work. You must also run the Fara-7B model using one of the methods below.

---

## ⚙️ Configuration

### Option 1: LM Studio (Recommended - Easiest) ⭐

**LM Studio is the EASIEST way to run Fara-7B locally!** No command line knowledge needed.

#### Why LM Studio?
- ✅ **User-friendly GUI** - No terminal commands required
- ✅ **One-click model download** - Browse and download models easily
- ✅ **Auto GPU detection** - Automatically uses your GPU if available
- ✅ **Built-in quantization** - Choose FP16, INT8, or INT4 for your VRAM
- ✅ **OpenAI-compatible API** - Works seamlessly with this interface

#### Setup Steps:

1. **Download and Install [LM Studio](https://lmstudio.ai/)**

2. **Download Fara-7B Model**:
   - Open LM Studio
   - Click the "🔍 Discover" tab (search icon)
   - Search for: `microsoft/Fara-7B` or `Fara-7B`
   - Choose a version based on your GPU VRAM:
     - **FP16** - Best quality, needs 12-14GB VRAM
     - **INT8** (Q8) - Good quality, needs 8GB VRAM
     - **INT4** (Q4_K_M) - Lower quality, needs 4-6GB VRAM
   - Click **Download**

3. **Start the Local Server**:
   - Click the "💻 Local Server" tab in LM Studio
   - Select the `Fara-7B` model you just downloaded
   - Click **"Start Server"**
   - Server starts on port `1234` (default)
   - ⚠️ **Keep LM Studio running** while using the interface

4. **Configure This Interface**:
   - In the Gradio interface, click the **"⚙️ Configuration"** tab
   - Enter:
     - **Model Endpoint**: `http://localhost:1234/v1`
     - **API Key**: `lm-studio` (any value works for local)
   - Click **"💾 Save Configuration"**

5. **Start Automating!** ✅
   - Go to the **"🚀 Run Tasks"** tab
   - Enter a task like: "Find an Xbox controller on Amazon"
   - Click **"Run Task"**
   - Watch Fara-7B work!

---

### Option 2: VLLM (Advanced - Command Line)

**For advanced users comfortable with the command line.**

#### Setup Steps:

1. **Install VLLM**:
   ```bash
   pip install vllm
   ```

2. **Start the Fara-7B Server**:
   ```bash
   vllm serve microsoft/Fara-7B --port 5000 --dtype auto
   ```
   - First run will download the model (~14GB)
   - Auto-detects GPU and uses FP16 if available
   - ⚠️ **Keep this terminal open** while using the interface

3. **Configure This Interface**:
   - **Model Endpoint**: `http://localhost:5000/v1`
   - **API Key**: (leave empty for local)
   - Click **"💾 Save Configuration"**

---

### Option 3: Azure Foundry (Cloud - Paid)

**Deploy Fara-7B on Microsoft Azure for cloud-based inference.**

#### Setup Steps:

1. **Deploy Fara-7B** on [Azure AI Foundry](https://ai.azure.com/explore/models/Fara-7B/version/2/registry/azureml-msr)

2. **Get your endpoint details** from the Azure deployment page

3. **Configure This Interface**:
   - **Model Endpoint**: `https://your-endpoint.inference.ml.azure.com/v1`
   - **API Key**: Your Azure API key
   - Click **"💾 Save Configuration"**

---

## ❌ What WON'T Work

**This interface will NOT work with:**
- ❌ OpenAI ChatGPT API (`https://api.openai.com`)
- ❌ Anthropic Claude API
- ❌ Google Gemini API
- ❌ Any other cloud LLM service

**Why?** Fara-7B is a specialized model for computer use and web automation. It requires specific training and architecture that generic chat models don't have.

---

## 🖥️ Interface Guide

### Main Interface Tabs

1. **🚀 Run Tasks**: Main interface for running automation tasks
   - Enter task description
   - Monitor real-time progress
   - View results and actions taken

2. **⚙️ Configuration**: Set up your Fara-7B model server
   - Configure LM Studio, VLLM, or Azure endpoint
   - Test connection
   - View setup instructions

3. **ℹ️ About**: Learn about Fara-7B and the interface

### Example Tasks

Try these example tasks to get started:

- `"How many pages does Wikipedia have?"`
- `"Search for the weather in New York City"`
- `"Find the latest iPhone price on Apple's website"`
- `"Search for job openings for Python developers in Seattle"`
- `"Find a hotel in Paris for next month"`

---

## 🎯 How It Works

1. **Visual Understanding**: Fara-7B takes screenshots of web pages and understands the visual layout

2. **Action Planning**: The model decides what actions to take (click, type, scroll, navigate)

3. **Coordinate Prediction**: Actions are executed at precise coordinates on the page

4. **Task Completion**: The agent continues until the task is completed or the maximum rounds are reached

---

## 📋 System Requirements

### For the Gradio Interface:
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum
- **Storage**: 5GB free space for dependencies
- **Internet**: For downloading dependencies and web browsing

### For Running Fara-7B Locally (LM Studio or VLLM):

#### **Minimum VRAM** (4-6 GB)
- Supports quantized inference (INT4)
- Slower performance, good for testing
- **Example GPUs**: GTX 1660 Ti, RTX 3050, RTX 2060

#### **Recommended VRAM** (8-12 GB) ⭐
- INT8 or FP16 precision
- Smooth inference performance
- **Example GPUs**: RTX 3060 12GB, RTX 4060 Ti, RTX 3080 10GB

#### **Optimal VRAM** (16+ GB)
- Full FP16 precision with large contexts
- Fast inference
- **Example GPUs**: RTX 4080, RTX 4090, RTX A5000, A6000

#### **CPU Fallback** (No GPU)
- LM Studio and VLLM can run on CPU only
- **Very slow** (10-30x slower than GPU)
- Requires 16GB+ system RAM
- Functional for testing but not recommended for regular use

### Other Recommended Specs:
- **Memory**: 16GB+ RAM (32GB for CPU-only mode)
- **CPU**: Multi-core processor (8+ cores recommended)
- **Storage**: 20GB free (5GB interface + 14GB model weights)
- **Internet**: Stable broadband for model download and web browsing

---

## 🛠️ Troubleshooting

### "Configuration Error - Please configure your model endpoint"
- **Cause**: You haven't set up a Fara-7B model server yet
- **Solution**: Follow the [Configuration](#️-configuration) section above to set up LM Studio, VLLM, or Azure

### "Connection Error - Could not connect to endpoint"
- **Cause**: The model server isn't running
- **Solution**:
  - **LM Studio**: Make sure you clicked "Start Server" and it shows "Running"
  - **VLLM**: Check that the `vllm serve` command is still running in your terminal
  - **Azure**: Verify your endpoint URL and API key are correct

### "HTTP Error 400 - invalid model ID"
- **Cause**: You're using an OpenAI/ChatGPT endpoint instead of Fara-7B
- **Solution**: You MUST use Fara-7B. See the [Configuration](#️-configuration) section to set up LM Studio or VLLM

### "Failed to install Playwright browsers"
- Run manually: `python -m playwright install`
- Check your internet connection
- Try running with admin/sudo privileges

### Model is very slow
- **Check GPU usage**: Make sure LM Studio/VLLM is using your GPU
- **Try quantization**: Use INT8 or INT4 models in LM Studio for faster inference
- **Check VRAM**: If you're running out of VRAM, use a smaller quantized model

### Getting Help

- 📖 [Official Fara Documentation](https://github.com/microsoft/fara)
- 💬 [Pinokio Discord](https://discord.gg/pinokio)
- 🐛 [Report Issues](https://github.com/neviah/Fara-Pinokio/issues)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The underlying Fara-7B model is licensed under Microsoft's terms. See the [official repository](https://github.com/microsoft/fara) for details.

## 🙏 Acknowledgments

- **Microsoft Research** for developing the amazing Fara-7B model
- **Gradio Team** for the excellent web interface framework
- **Pinokio Community** for the one-click deployment platform
- **LM Studio** for making local LLM hosting accessible to everyone

## 📊 Performance Notes

- **Speed**: Fara-7B averages ~16 steps per task vs ~41 for comparable models
- **Efficiency**: 7B parameters vs much larger competing agents
- **Accuracy**: State-of-the-art performance in its size class
- **Privacy**: Can run entirely on-device with local hosting

---

**Ready to automate your web tasks? Install Pinokio and set up LM Studio today!** 🚀

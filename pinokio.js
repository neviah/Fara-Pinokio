{
  "title": "Fara-7B Computer Use Agent",
  "description": "Microsoft's efficient 7B parameter agentic model for automating web tasks with a beautiful Gradio interface",
  "icon": "icon.png",
  "author": "Microsoft Research + Community",
  "version": "1.0.0",
  "license": "MIT",
  "repository": "https://github.com/microsoft/fara",
  "tags": ["AI", "Agent", "Computer Vision", "Web Automation", "Microsoft", "Gradio"],
  "requirements": {
    "python": ">=3.8",
    "node": ">=14.0.0",
    "gpu": "recommended",
    "memory": "8GB+"
  },
  "menu": [
    {
      "text": "🏠 Home",
      "href": "index.html"
    },
    {
      "text": "🚀 Launch Interface",
      "href": "javascript:startFara()"
    },
    {
      "text": "📖 Documentation", 
      "href": "README.md"
    },
    {
      "text": "⚙️ Configuration",
      "href": "config.html"
    }
  ],
  "install": [
    {
      "method": "shell.run",
      "params": {
        "message": "Installing Fara-7B dependencies...",
        "venv": "env",
        "path": "install.js"
      }
    }
  ],
  "run": [
    {
      "method": "shell.run",
      "params": {
        "daemon": true,
        "venv": "env", 
        "path": "gradio_interface.py",
        "message": "Starting Fara-7B Gradio Interface...",
        "on": [
          {
            "event": "/http:\\/\\/\\S+/",
            "done": true
          }
        ]
      }
    },
    {
      "method": "local.set",
      "params": {
        "url": "http://localhost:7860"
      }
    },
    {
      "method": "browser.open",
      "params": {
        "uri": "{{local.url}}"
      }
    }
  ],
  "api": [
    {
      "method": "shell.run", 
      "params": {
        "daemon": true,
        "venv": "env",
        "path": "gradio_interface.py",
        "message": "Starting Fara-7B API server..."
      }
    }
  ],
  "pre": {
    "install": [
      {
        "method": "notify",
        "params": {
          "html": "Installing <b>Fara-7B Computer Use Agent</b>...<br><br>This will:<br>• Clone the official Microsoft Fara repository<br>• Set up Python environment with dependencies<br>• Install Playwright browsers<br>• Create Gradio web interface<br><br>⏱️ Estimated time: 5-10 minutes"
        }
      }
    ],
    "run": [
      {
        "method": "notify", 
        "params": {
          "html": "🚀 <b>Starting Fara-7B Agent</b><br><br>💡 <b>Important:</b> For local inference, make sure you have:<br>• VLLM server running: <code>vllm serve microsoft/Fara-7B --port 5000</code><br>• Or configure Azure Foundry endpoint in the interface<br><br>🌐 The interface will open at <b>http://localhost:7860</b>"
        }
      }
    ]
  },
  "post": {
    "install": [
      {
        "method": "notify",
        "params": {
          "html": "✅ <b>Fara-7B Installation Complete!</b><br><br>🎯 <b>What's installed:</b><br>• Microsoft Fara-7B agent framework<br>• Gradio web interface<br>• All Python dependencies<br>• Playwright browsers<br><br>🚀 <b>Next steps:</b><br>1. Click 'Run' to start the interface<br>2. Configure your model endpoint<br>3. Start automating web tasks!<br><br>📚 Check the Documentation tab for detailed usage instructions."
        }
      }
    ],
    "run": [
      {
        "method": "notify",
        "params": {
          "html": "🎉 <b>Fara-7B is Ready!</b><br><br>The interface is now running at:<br><a href='http://localhost:7860' target='_blank'>http://localhost:7860</a><br><br>💡 <b>Quick Start:</b><br>1. Configure your model endpoint in the 'Configuration' tab<br>2. Try example tasks like 'Search for latest iPhone price'<br>3. Watch Fara automate web browsing for you!<br><br>⚠️ <b>Note:</b> Ensure you have proper model hosting setup (local VLLM or Azure Foundry)"
        }
      }
    ]
  },
  "params": [
    {
      "id": "model_endpoint",
      "title": "Model Endpoint",
      "description": "Base URL for the Fara-7B model API",
      "placeholder": "http://localhost:5000/v1",
      "value": "http://localhost:5000/v1"
    },
    {
      "id": "api_key",
      "title": "API Key", 
      "description": "API key for model access (leave empty for local VLLM)",
      "placeholder": "your-api-key-here",
      "value": ""
    },
    {
      "id": "max_rounds",
      "title": "Max Rounds",
      "description": "Maximum number of actions the agent can take per task",
      "placeholder": "50",
      "value": "50"
    },
    {
      "id": "headless",
      "title": "Headless Mode",
      "description": "Run browser in headless mode (no visible window)",
      "placeholder": "true",
      "value": "true"
    }
  ],
  "env": {
    "FARA_MODEL_ENDPOINT": "{{params.model_endpoint}}",
    "FARA_API_KEY": "{{params.api_key}}",
    "FARA_MAX_ROUNDS": "{{params.max_rounds}}",
    "FARA_HEADLESS": "{{params.headless}}"
  },
  "features": [
    "🤖 Microsoft's state-of-the-art 7B parameter computer use agent",
    "🎯 Automates complex web tasks through visual understanding",
    "💻 Beautiful Gradio web interface for easy interaction",
    "🔧 Supports both local VLLM and Azure Foundry deployment",
    "📊 Task history and real-time progress monitoring", 
    "🎨 Modern, responsive UI with dark/light theme support",
    "🚀 One-click installation and setup",
    "📝 Comprehensive documentation and examples"
  ],
  "tutorials": [
    {
      "title": "Getting Started",
      "description": "Learn how to set up and run your first task with Fara-7B",
      "uri": "tutorials/getting-started.md"
    },
    {
      "title": "Model Hosting Options",
      "description": "Compare local VLLM vs Azure Foundry deployment",
      "uri": "tutorials/hosting-options.md" 
    },
    {
      "title": "Advanced Configuration",
      "description": "Customize Fara-7B for your specific use cases",
      "uri": "tutorials/advanced-config.md"
    }
  ],
  "help": {
    "discord": "https://discord.gg/pinokio",
    "github": "https://github.com/microsoft/fara",
    "docs": "https://github.com/microsoft/fara/blob/main/README.md"
  }
}
import gradio as gr
import asyncio
import os
import tempfile
import logging
import json
import requests
from pathlib import Path
from datetime import datetime

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_fara_task(task_description):
    """Call the configured LLM endpoint with the task description"""
    try:
        if not task_description or not task_description.strip():
            return (
                "❌ Error",
                "",
                "Please enter a task description"
            )
        
        # Load current config
        config = read_config()
        base_url = config.get("base_url", "").strip()
        api_key = config.get("api_key", "").strip()
        
        if not base_url:
            return (
                "❌ Configuration Error",
                "",
                "Please configure your model endpoint in the Configuration tab"
            )
        
        # Prepare OpenAI-compatible API request
        headers = {
            "Content-Type": "application/json",
        }
        if api_key and api_key != "lm-studio":
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Construct the prompt for Fara-7B
        payload = {
            "model": config.get("model", "microsoft/Fara-7B"),
            "messages": [
                {
                    "role": "system",
                    "content": "You are Fara-7B, a helpful AI assistant specialized in web automation and computer use tasks. Explain how you would approach the task step by step."
                },
                {
                    "role": "user",
                    "content": f"Task: {task_description}\n\nPlease explain your approach to completing this task."
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        # Make the API call
        endpoint = base_url.rstrip("/") + "/chat/completions"
        logger.info(f"Calling endpoint: {endpoint}")
        
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Extract the response
        if "choices" in data and len(data["choices"]) > 0:
            result = data["choices"][0]["message"]["content"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return (
                f"✅ Completed at {timestamp}",
                f"Task: {task_description}",
                result
            )
        else:
            return (
                "❌ Unexpected Response",
                f"Task: {task_description}",
                "The API returned an unexpected response format"
            )
            
    except requests.exceptions.Timeout:
        return (
            "❌ Timeout Error",
            f"Task: {task_description}",
            "The request timed out. Please check if your model server is running and responding."
        )
    except requests.exceptions.ConnectionError:
        return (
            "❌ Connection Error",
            f"Task: {task_description}",
            f"Could not connect to {config.get('base_url', 'the endpoint')}. Make sure your model server is running."
        )
    except requests.exceptions.HTTPError as e:
        return (
            f"❌ HTTP Error {e.response.status_code}",
            f"Task: {task_description}",
            f"Server returned an error: {e.response.text[:200]}"
        )
    except Exception as e:
        logger.exception("Error running task")
        return (
            f"❌ Error: {type(e).__name__}",
            f"Task: {task_description}",
            f"Details: {str(e)}"
        )

# Load/save simple endpoint config (supports LM Studio by default)
CONFIG_PATH = Path(__file__).parent / "endpoint_config.json"

DEFAULT_CONFIG = {
    "model": "microsoft/Fara-7B",
    # LM Studio default OpenAI-compatible endpoint
    "base_url": "http://localhost:1234/v1",
    # LM Studio accepts any key; default label for clarity
    "api_key": "lm-studio"
}

def read_config():
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {**DEFAULT_CONFIG, **data}
    except Exception as e:
        logger.warning(f"Failed to read config: {e}")
    return DEFAULT_CONFIG.copy()

def write_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True, "✅ Configuration saved"
    except Exception as e:
        return False, f"❌ Failed to save configuration: {e}"

app_config = read_config()

# Create the Gradio interface
with gr.Blocks(title="Fara-7B Computer Use Agent") as demo:
    gr.Markdown("""
    # 🤖 Fara-7B Computer Use Agent
    
    Microsoft's efficient 7B parameter model for automating web tasks.
    
    **Note**: This is the interface shell. To activate full functionality, configure your model endpoint below.
    """)
    
    with gr.Tab("🚀 Run Tasks"):
        with gr.Row():
            task_input = gr.Textbox(
                label="Task Description",
                placeholder="Example: Search for the latest iPhone price on Apple's website",
                lines=3
            )
            
        run_button = gr.Button("🏃 Run Task", variant="primary")
        
        gr.Examples(
            examples=[
                ["How many pages does Wikipedia have?"],
                ["Search for the weather in New York City"],
                ["Find the latest iPhone price on Apple's website"],
            ],
            inputs=task_input
        )
        
        with gr.Row():
            result_output = gr.Textbox(label="🎯 Result", lines=3, interactive=False)
            task_output = gr.Textbox(label="📝 Task Info", lines=3, interactive=False) 
            notes_output = gr.Textbox(label="📋 Notes", lines=3, interactive=False)
    
    with gr.Tab("⚙️ Configuration"):
        gr.Markdown("### Model Endpoint Setup")
        
        model_endpoint = gr.Textbox(
            label="Model Endpoint",
            value=app_config.get("base_url", DEFAULT_CONFIG["base_url"]),
            placeholder="http://localhost:1234/v1"
        )
        
        api_key = gr.Textbox(
            label="API Key", 
            placeholder="your-api-key-here",
            type="password"
        )

        api_key.value = app_config.get("api_key", DEFAULT_CONFIG["api_key"])  # set default after creation
        
        config_button = gr.Button("💾 Save Configuration")
        config_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("""
        ### Setup Instructions
        
        **For LM Studio (recommended for easy local use):**
        1. Install [LM Studio](https://lmstudio.ai/) and start a local server (OpenAI compatible)
        2. Use endpoint `http://localhost:1234/v1` and any API key (e.g. `lm-studio`)
        
        **For Local VLLM:**
        ```bash
        pip install vllm
        vllm serve "microsoft/Fara-7B" --port 5000 --dtype auto
        ```
        
        **For Azure Foundry:**
        - Deploy Fara-7B on Azure Foundry
        - Use your endpoint URL and API key above
        """)
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ### About Fara-7B
        
        Fara-7B is Microsoft's first agentic small language model designed for computer use.
        
        **Features:**
        - 🎯 Only 7B parameters vs much larger competing models
        - 🖥️ Visual webpage understanding and coordinate-based actions
        - 🏠 Can run locally for privacy and low latency
        - 📈 Efficient: ~16 steps per task vs ~41 for comparable models
        
        **Links:**
        - [GitHub Repository](https://github.com/microsoft/fara)
        - [Hugging Face Model](https://huggingface.co/microsoft/Fara-7b)
        - [Azure Foundry](https://aka.ms/foundry-fara-7b)
        """)
    
    # Event handlers
    def save_config(endpoint, key):
        cfg = read_config()
        cfg["base_url"] = endpoint.strip() or cfg.get("base_url")
        cfg["api_key"] = key.strip() or cfg.get("api_key")
        ok, msg = write_config(cfg)
        return msg
    
    run_button.click(
        fn=run_fara_task,
        inputs=task_input,
        outputs=[result_output, task_output, notes_output]
    )
    
    config_button.click(
        fn=save_config,
        inputs=[model_endpoint, api_key],
        outputs=config_status
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
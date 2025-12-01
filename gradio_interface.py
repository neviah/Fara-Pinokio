import gradio as gr
import asyncio
import os
import tempfile
import logging
import json
from pathlib import Path
from datetime import datetime

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_fara_task(task_description):
    """Simple function to simulate running a Fara task"""
    try:
        # This is a placeholder - in real implementation, this would use the Fara agent
        result = f"Task completed: {task_description}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return (
            f"✅ Completed at {timestamp}",
            f"Task: {task_description}",
            "This is a demo interface. To use real Fara functionality, configure your model endpoint below."
        )
    except Exception as e:
        return (
            f"❌ Error: {str(e)}", 
            "", 
            "Please check your configuration and try again."
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
import gradio as gr
import asyncio
import json
import os
from pathlib import Path
from fara import FaraAgent
from fara.browser.browser_bb import BrowserBB
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Global variables
agent = None
browser_manager = None
current_task_running = False
current_task = None  # Track the asyncio task for cancellation

# Paths
ENDPOINT_CONFIG_PATH = "endpoint_config.json"
DOWNLOADS_FOLDER = "fara_downloads"
SCREENSHOTS_FOLDER = os.path.join(DOWNLOADS_FOLDER, "screenshots")

# Create folders
os.makedirs(DOWNLOADS_FOLDER, exist_ok=True)
os.makedirs(SCREENSHOTS_FOLDER, exist_ok=True)

DEFAULT_ENDPOINT_CONFIG = {
    "model": "microsoft/Fara-7B",
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
}


def load_config():
    """Load endpoint configuration from JSON file."""
    if os.path.exists(ENDPOINT_CONFIG_PATH):
        with open(ENDPOINT_CONFIG_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_ENDPOINT_CONFIG.copy()


def save_config(model_name, model_endpoint, api_key):
    """Save endpoint configuration to JSON file."""
    # Strip /chat/completions if user included it
    if model_endpoint.endswith("/chat/completions"):
        model_endpoint = model_endpoint[: -len("/chat/completions")]

    config = {
        "model": model_name,
        "base_url": model_endpoint,
        "api_key": api_key,
    }

    with open(ENDPOINT_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    return "✅ Configuration saved successfully!"


async def initialize_browser_and_agent():
    """Initialize the browser manager and Fara agent."""
    global browser_manager, agent

    if agent is not None:
        return agent, browser_manager

    # Load endpoint config
    config = load_config()

    # Initialize browser manager
    logger.info("Initializing Browser...")
    browser_manager = BrowserBB(
        headless=False,  # Always use headed mode on Windows to avoid xvfb issues
        viewport_height=900,
        viewport_width=1440,
        page_script_path=None,
        browser_channel="chromium",  # Use chromium instead of firefox for better Windows compatibility
        browser_data_dir=None,
        downloads_folder=DOWNLOADS_FOLDER,
        to_resize_viewport=True,
        single_tab_mode=True,
        animate_actions=True,  # Enable human-like animations to avoid bot detection
        use_browser_base=False,
        logger=logger,
    )
    logger.info("Browser Running... Starting Fara Agent...")

    # Initialize Fara agent
    agent = FaraAgent(
        browser_manager=browser_manager,
        client_config=config,
        start_page="https://www.bing.com/",
        downloads_folder=DOWNLOADS_FOLDER,
        save_screenshots=True,
        max_rounds=100,
    )

    await agent.initialize()
    logger.info("✅ Fara Agent Ready!")

    return agent, browser_manager


async def run_task_async(task_description, progress=gr.Progress()):
    """Run a Fara task with full browser automation."""
    global current_task_running, current_task

    if not task_description or task_description.strip() == "":
        return (
            "⚠️ Please enter a task description",
            None,
            "No task provided",
            "",
        )

    if current_task_running:
        return (
            "⚠️ A task is already running. Please wait for it to complete or cancel it.",
            None,
            "Task already running",
            "",
        )

    current_task_running = True

    try:
        # Initialize agent if needed
        progress(0.1, desc="Initializing browser and agent...")
        agent_instance, _ = await initialize_browser_and_agent()

        progress(0.2, desc=f"Running task: {task_description}")
        
        # Run the task
        logger.info("##########################################")
        logger.info(f"Task: {task_description}")
        logger.info("##########################################")

        final_answer, all_actions, all_observations = await agent_instance.run(
            task_description
        )

        progress(1.0, desc="Task completed!")

        # Format results
        result_text = f"## ✅ Task Completed\n\n**Final Answer:**\n{final_answer}\n\n"

        # Create action history
        action_history = []
        for i, (action, obs) in enumerate(zip(all_actions, all_observations), 1):
            action_str = f"**Step {i}:** {action.get('action', 'Unknown')}"
            if action.get("action") == "computer":
                action_str += f"\n- **Operation:** {action.get('op', 'N/A')}"
                if action.get("coordinate"):
                    action_str += f"\n- **Coordinates:** {action['coordinate']}"
                if action.get("text"):
                    action_str += f"\n- **Text:** {action['text']}"

            action_history.append(action_str)

        # Get latest screenshot
        latest_screenshot = None
        screenshot_files = sorted(Path(SCREENSHOTS_FOLDER).glob("*.png"))
        if screenshot_files:
            latest_screenshot = str(screenshot_files[-1])

        notes = f"Completed in {len(all_actions)} steps"

        return result_text, latest_screenshot, notes, "\n\n".join(action_history)

    except asyncio.CancelledError:
        logger.info("Task was cancelled by user")
        return (
            "## 🛑 Task Cancelled\n\nThe task was stopped by user request.",
            None,
            "Task cancelled",
            "",
        )
    except Exception as e:
        logger.error(f"Error running task: {e}", exc_info=True)
        return (
            f"## ❌ Error\n\n{str(e)}\n\n**Tip:** Use the Reset button to clear the error and try again.",
            None,
            f"Error: {str(e)}",
            "",
        )
    finally:
        current_task_running = False
        current_task = None


def run_task_sync(task_description):
    """Synchronous wrapper for run_task_async."""
    global current_task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        current_task = loop.create_task(run_task_async(task_description))
        return loop.run_until_complete(current_task)
    finally:
        loop.close()


def cancel_task():
    """Cancel the currently running task."""
    global current_task, current_task_running
    
    if current_task and current_task_running:
        current_task.cancel()
        current_task_running = False
        return "✅ Task cancellation requested. Please wait..."
    return "ℹ️ No task is currently running"


def reset_agent():
    """Reset the agent and browser without shutting down."""
    global agent, browser_manager, current_task_running, current_task
    
    async def cleanup_and_reset():
        global agent, browser_manager, current_task_running, current_task
        if agent:
            try:
                await agent.close()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
        agent = None
        browser_manager = None
        current_task_running = False
        current_task = None
    
    try:
        asyncio.run(cleanup_and_reset())
        return "✅ Agent reset successfully. You can start a new task."
    except Exception as e:
        return f"⚠️ Reset attempted with warnings: {str(e)}"


def shutdown_agent():
    """Shutdown the agent and browser."""
    global agent, browser_manager

    async def cleanup():
        global agent, browser_manager
        if agent:
            await agent.close()
            agent = None
        browser_manager = None

    if agent:
        asyncio.run(cleanup())
        return "✅ Agent shut down successfully"
    return "ℹ️ No agent running"


# Load initial config
initial_config = load_config()

# Create Gradio interface
with gr.Blocks(title="Fara-7B Computer Use Agent") as demo:
    gr.Markdown(
        """
    # 🤖 Fara-7B Computer Use Agent
    
    Microsoft's efficient 7B parameter model for automating web tasks.
    
    **Note:** This is the REAL Fara agent with browser automation! 🌐
    """
    )

    with gr.Tabs():
        # ========== RUN TASKS TAB ==========
        with gr.Tab("🚀 Run Tasks"):
            gr.Markdown(
                """
            ### Enter a task for Fara to automate
            
            Fara will open a browser, take screenshots, and perform actions to complete your task.
            
            **Examples:**
            - "How many pages does Wikipedia have?"
            - "Find an Xbox controller on Amazon"
            - "Search for the weather in New York City"
            - "Find job openings for Python developers in Seattle"
            """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    task_input = gr.Textbox(
                        label="Task Description",
                        placeholder="find me an xbox controller on amazon.",
                        lines=3,
                    )

                    with gr.Row():
                        run_button = gr.Button("▶️ Run Task", variant="primary", size="lg")
                        cancel_button = gr.Button("🛑 Cancel Task", variant="stop", size="lg")
                        reset_button = gr.Button("🔄 Reset Agent", variant="secondary", size="lg")

            with gr.Row():
                with gr.Column(scale=2):
                    result_output = gr.Markdown(label="Result")

                with gr.Column(scale=1):
                    screenshot_output = gr.Image(
                        label="Latest Screenshot", type="filepath"
                    )

            with gr.Row():
                task_info_output = gr.Textbox(label="📊 Task Info", lines=2)
                control_status_output = gr.Textbox(label="🎛️ Control Status", lines=2)

            with gr.Row():
                action_history_output = gr.Markdown(label="📜 Action History")

            gr.Markdown("### Example Tasks")
            with gr.Row():
                example1_btn = gr.Button("How many pages does Wikipedia have?")
                example2_btn = gr.Button("Search for the weather in New York City")
                example3_btn = gr.Button(
                    "Find the latest iPhone price on Apple's website"
                )

            # Button actions
            run_button.click(
                fn=run_task_sync,
                inputs=[task_input],
                outputs=[
                    result_output,
                    screenshot_output,
                    task_info_output,
                    action_history_output,
                ],
            )
            
            cancel_button.click(
                fn=cancel_task,
                outputs=[control_status_output],
            )
            
            reset_button.click(
                fn=reset_agent,
                outputs=[control_status_output],
            )

            # Example buttons
            example1_btn.click(
                fn=lambda: "How many pages does Wikipedia have?",
                outputs=[task_input],
            )
            example2_btn.click(
                fn=lambda: "Search for the weather in New York City",
                outputs=[task_input],
            )
            example3_btn.click(
                fn=lambda: "Find the latest iPhone price on Apple's website",
                outputs=[task_input],
            )

        # ========== CONFIGURATION TAB ==========
        with gr.Tab("⚙️ Configuration"):
            gr.Markdown(
                """
            ### Configure Your Model Endpoint
            
            ⚠️ **IMPORTANT:** You must run Fara-7B locally using LM Studio, VLLM, or Azure.
            
            #### Quick Setup with LM Studio (Recommended):
            1. Download [LM Studio](https://lmstudio.ai/)
            2. Search for and download `microsoft/Fara-7B` (or quantized versions)
            3. Click "Start Server" (port 1234)
            4. Use the default settings below
            
            **💡 Quantized Model Support:**
            - **4-bit (Q4_K_M)**: ~4GB VRAM - Good balance of speed and quality
            - **5-bit (Q5_K_M)**: ~5GB VRAM - Better quality
            - **8-bit (Q8_0)**: ~8GB VRAM - Near-original quality
            - **FP16 (full)**: ~14GB VRAM - Best quality
            
            Simply load any quantized GGUF version in LM Studio - no config changes needed!
            
            #### Or use VLLM:
            ```bash
            vllm serve microsoft/Fara-7B --port 5000
            ```
            Then set endpoint to: `http://localhost:5000/v1`
            """
            )

            model_name_input = gr.Textbox(
                label="Model Name",
                value=initial_config.get("model", "microsoft/Fara-7B"),
                placeholder="microsoft/Fara-7B",
            )

            model_endpoint_input = gr.Textbox(
                label="Model Endpoint (Base URL)",
                value=initial_config.get("base_url", "http://localhost:1234/v1"),
                placeholder="http://localhost:1234/v1",
            )

            api_key_input = gr.Textbox(
                label="API Key (optional for local)",
                value=initial_config.get("api_key", "lm-studio"),
                type="password",
                placeholder="lm-studio",
            )

            save_config_button = gr.Button("💾 Save Configuration", variant="primary")
            config_status = gr.Textbox(label="Status", interactive=False)

            save_config_button.click(
                fn=save_config,
                inputs=[model_name_input, model_endpoint_input, api_key_input],
                outputs=[config_status],
            )

            gr.Markdown(
                """
            ### ❌ What WON'T Work
            - OpenAI ChatGPT API
            - Claude API  
            - Any other cloud LLM
            
            **Why?** Fara-7B is a specialized computer use model. Generic chat models don't have the required capabilities.
            """
            )

        # ========== ABOUT TAB ==========
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
            ## About Fara-7B
            
            **Fara-7B** is Microsoft's efficient 7 billion parameter model designed specifically for computer use and web automation tasks.
            
            ### Key Features:
            - 🎯 **Visual Understanding**: Takes screenshots and understands page layout
            - 🤖 **Action Planning**: Decides what actions to take (click, type, scroll)
            - 📍 **Coordinate Prediction**: Executes actions at precise locations
            - ⚡ **Efficient**: ~16 steps per task vs ~41 for comparable models
            - 🔒 **Private**: Runs entirely on your machine
            
            ### What Fara Can Do:
            - Search and research tasks
            - Form filling and account management
            - Booking services (travel, hotels, restaurants)
            - Shopping and price comparison
            - Job hunting and real estate searches
            - Data collection from websites
            
            ### System Requirements:
            - **Minimum:** 4-6GB VRAM (INT4 quantized)
            - **Recommended:** 8-12GB VRAM (FP16)
            - **Optimal:** 16+ GB VRAM
            - **CPU Fallback:** Possible but very slow (16GB+ RAM)
            
            ### Links:
            - [Official Fara Repository](https://github.com/microsoft/fara)
            - [LM Studio](https://lmstudio.ai/)
            - [Report Issues](https://github.com/neviah/Fara-Pinokio/issues)
            
            ---
            
            **Version:** 1.0.0 with Full Browser Automation  
            **License:** MIT  
            **Made with:** Microsoft Fara, Gradio, Playwright
            """
            )

            shutdown_button = gr.Button("🛑 Shutdown Agent", variant="stop")
            shutdown_status = gr.Textbox(label="Shutdown Status", interactive=False)

            shutdown_button.click(
                fn=shutdown_agent,
                outputs=[shutdown_status],
            )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
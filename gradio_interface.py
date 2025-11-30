import gradio as gr
import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import threading
from datetime import datetime

# Set up logging to capture Fara agent output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FaraGradioInterface:
    """Gradio interface for the Fara-7B computer use agent"""
    
    def __init__(self):
        self.agent = None
        self.browser_manager = None
        self.is_initialized = False
        self.current_task = None
        self.task_history = []
        
    async def initialize_agent(self, endpoint_config: Dict[str, str]) -> bool:
        """Initialize the Fara agent with the given configuration"""
        try:
            from fara import FaraAgent
            from fara.browser.browser_bb import BrowserBB
            
            # Create downloads folder
            downloads_folder = os.path.join(tempfile.gettempdir(), "fara_downloads")
            os.makedirs(downloads_folder, exist_ok=True)
            
            # Initialize browser manager
            self.browser_manager = BrowserBB(
                headless=True,
                viewport_height=900,
                viewport_width=1440,
                page_script_path=None,
                browser_channel="firefox",
                browser_data_dir=None,
                downloads_folder=downloads_folder,
                to_resize_viewport=True,
                single_tab_mode=True,
                animate_actions=False,
                use_browser_base=False,
                logger=logger,
            )
            
            # Initialize Fara agent
            self.agent = FaraAgent(
                browser_manager=self.browser_manager,
                client_config=endpoint_config,
                start_page="https://www.bing.com/",
                downloads_folder=downloads_folder,
                save_screenshots=True,
                max_rounds=50,
            )
            
            await self.agent.initialize()
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            return False
    
    async def run_task_async(self, task: str, progress_callback=None) -> Tuple[str, list, list]:
        """Run a task asynchronously with the Fara agent"""
        if not self.is_initialized:
            raise Exception("Agent not initialized. Please configure endpoint first.")
        
        if progress_callback:
            progress_callback(0.1, "Starting task...")
        
        try:
            self.current_task = task
            
            if progress_callback:
                progress_callback(0.3, "Running Fara agent...")
            
            # Run the agent
            final_answer, actions, observations = await self.agent.run(task)
            
            if progress_callback:
                progress_callback(0.9, "Task completed!")
            
            # Store in history
            result = {
                "task": task,
                "timestamp": datetime.now().isoformat(),
                "final_answer": final_answer,
                "actions": actions,
                "observations": observations,
                "status": "completed"
            }
            self.task_history.append(result)
            
            if progress_callback:
                progress_callback(1.0, "Done!")
            
            return final_answer, actions, observations
            
        except Exception as e:
            error_msg = f"Error running task: {str(e)}"
            logger.error(error_msg)
            
            # Store error in history
            result = {
                "task": task,
                "timestamp": datetime.now().isoformat(),
                "final_answer": error_msg,
                "actions": [],
                "observations": [],
                "status": "error"
            }
            self.task_history.append(result)
            
            raise Exception(error_msg)
    
    def run_task_sync(self, task: str, progress=gr.Progress()) -> Tuple[str, str, str]:
        """Synchronous wrapper for running tasks"""
        def progress_callback(value, desc):
            progress(value, desc=desc)
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_answer, actions, observations = loop.run_until_complete(
                self.run_task_async(task, progress_callback)
            )
            loop.close()
            
            # Format results for display
            actions_text = "\n".join([f"{i+1}. {action}" for i, action in enumerate(actions)])
            observations_text = "\n".join([f"{i+1}. {obs}" for i, obs in enumerate(observations)])
            
            return final_answer, actions_text, observations_text
            
        except Exception as e:
            return str(e), "", ""
    
    def configure_endpoint(self, model_name: str, base_url: str, api_key: str) -> str:
        """Configure the model endpoint"""
        try:
            endpoint_config = {
                "model": model_name,
                "base_url": base_url,
                "api_key": api_key or "not-needed"
            }
            
            # Test the configuration by initializing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(self.initialize_agent(endpoint_config))
            loop.close()
            
            if success:
                return "✅ Agent initialized successfully! Ready to run tasks."
            else:
                return "❌ Failed to initialize agent. Please check your configuration."
                
        except Exception as e:
            return f"❌ Configuration error: {str(e)}"
    
    def get_task_history(self) -> str:
        """Get formatted task history"""
        if not self.task_history:
            return "No tasks completed yet."
        
        history_text = "## Task History\n\n"
        for i, task in enumerate(self.task_history[-5:], 1):  # Show last 5 tasks
            status_icon = "✅" if task["status"] == "completed" else "❌"
            history_text += f"**{i}. {status_icon} {task['task']}**\n"
            history_text += f"*Time: {task['timestamp']}*\n"
            history_text += f"*Result: {task['final_answer'][:100]}...*\n\n"
        
        return history_text

# Create the interface instance
fara_interface = FaraGradioInterface()

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Fara-7B Computer Use Agent", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 Fara-7B Computer Use Agent
        
        Microsoft's efficient 7B parameter agentic model for automating web tasks.
        
        **Capabilities:**
        - 🔍 Searching for information and summarizing results
        - 📝 Filling out forms and managing accounts  
        - 🎫 Booking travel, movie tickets, and restaurant reservations
        - 🛒 Shopping and comparing prices across retailers
        - 💼 Finding job postings and real estate listings
        
        **Setup Instructions:**
        1. Configure your model endpoint below
        2. Enter a task for the agent to perform
        3. Watch as Fara automates the web browsing for you!
        """)
        
        with gr.Tab("🚀 Run Tasks"):
            with gr.Row():
                with gr.Column(scale=2):
                    task_input = gr.Textbox(
                        label="Task Description",
                        placeholder="Example: Search for the latest iPhone price on Apple's website",
                        lines=3
                    )
                    
                    run_button = gr.Button("🏃 Run Task", variant="primary", size="lg")
                    
                    with gr.Row():
                        clear_button = gr.Button("🗑️ Clear", size="sm")
                        example_button = gr.Button("💡 Load Example", size="sm")
                
                with gr.Column(scale=1):
                    status_box = gr.Textbox(
                        label="Agent Status", 
                        value="Not initialized - Configure endpoint first",
                        interactive=False
                    )
            
            gr.Examples(
                examples=[
                    ["How many pages does Wikipedia have?"],
                    ["Search for the weather in New York City"],
                    ["Find the latest iPhone price on Apple's website"],
                    ["Search for job openings for Python developers in Seattle"],
                    ["Find a hotel in Paris for next month"],
                ],
                inputs=task_input,
                label="Example Tasks"
            )
            
            with gr.Row():
                with gr.Column():
                    final_answer = gr.Textbox(
                        label="🎯 Final Result",
                        lines=4,
                        interactive=False
                    )
                
                with gr.Column():
                    actions_taken = gr.Textbox(
                        label="🔧 Actions Taken",
                        lines=6,
                        interactive=False
                    )
                
                with gr.Column():
                    observations = gr.Textbox(
                        label="👁️ Observations", 
                        lines=6,
                        interactive=False
                    )
        
        with gr.Tab("⚙️ Configuration"):
            gr.Markdown("### Model Endpoint Configuration")
            
            with gr.Row():
                with gr.Column():
                    model_name = gr.Textbox(
                        label="Model Name",
                        value="microsoft/Fara-7B",
                        placeholder="microsoft/Fara-7B"
                    )
                    
                    base_url = gr.Textbox(
                        label="Base URL", 
                        value="http://localhost:5000/v1",
                        placeholder="http://localhost:5000/v1"
                    )
                    
                    api_key = gr.Textbox(
                        label="API Key (optional for local)",
                        placeholder="your-api-key-here",
                        type="password"
                    )
                    
                    config_button = gr.Button("💾 Configure Agent", variant="primary")
                    config_status = gr.Textbox(label="Configuration Status", interactive=False)
            
            with gr.Column():
                gr.Markdown("""
                ### Setup Options
                
                **Local VLLM (Recommended for testing):**
                ```bash
                pip install vllm
                vllm serve "microsoft/Fara-7B" --port 5000 --dtype auto
                ```
                
                **Azure Foundry (Recommended for production):**
                - Deploy Fara-7B on Azure Foundry
                - Use your endpoint URL and API key
                
                **Requirements:**
                - Python 3.8+
                - GPU with sufficient VRAM (for local hosting)
                - Playwright browsers installed
                """)
        
        with gr.Tab("📊 History"):
            history_display = gr.Markdown("No tasks completed yet.")
            refresh_history = gr.Button("🔄 Refresh History")
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### About Fara-7B
            
            Fara-7B is Microsoft's first agentic small language model (SLM) designed specifically for computer use. 
            With only 7 billion parameters, it achieves state-of-the-art performance in web automation tasks.
            
            **Key Features:**
            - 🎯 **Efficient**: Only 7B parameters vs much larger competing models
            - 🖥️ **Visual**: Perceives webpages and takes actions via coordinates
            - 🏠 **Local**: Can run on-device for privacy and low latency
            - 📈 **Fast**: Averages only ~16 steps per task vs ~41 for comparable models
            
            **Research Paper & Code:**
            - [GitHub Repository](https://github.com/microsoft/fara)
            - [Hugging Face Model](https://huggingface.co/microsoft/Fara-7b)
            - [Azure Foundry Deployment](https://aka.ms/foundry-fara-7b)
            
            **Built with:**
            - Gradio for the web interface
            - Playwright for browser automation
            - OpenAI-compatible API for model inference
            
            *This interface provides an easy way to interact with Fara-7B for web automation tasks.*
            """)
        
        # Event handlers
        def load_example():
            return "Search for the latest news about artificial intelligence"
        
        def clear_inputs():
            return "", "", "", ""
        
        # Configure button action
        config_button.click(
            fn=fara_interface.configure_endpoint,
            inputs=[model_name, base_url, api_key],
            outputs=config_status
        )
        
        # Run task button action  
        run_button.click(
            fn=fara_interface.run_task_sync,
            inputs=task_input,
            outputs=[final_answer, actions_taken, observations]
        )
        
        # Helper button actions
        example_button.click(fn=load_example, outputs=task_input)
        clear_button.click(fn=clear_inputs, outputs=[task_input, final_answer, actions_taken, observations])
        refresh_history.click(fn=fara_interface.get_task_history, outputs=history_display)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
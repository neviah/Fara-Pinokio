import asyncio
import argparse
import os
from fara import FaraAgent
from fara.browser.browser_bb import BrowserBB
import logging
from typing import Dict
from pathlib import Path
import json


# Configure logging to only show logs from fara.fara_agent
logging.basicConfig(
    level=logging.CRITICAL,  
    format="%(message)s",  
)

# Enable INFO level only for fara.fara_agent
fara_agent_logger = logging.getLogger("fara.fara_agent")
fara_agent_logger.setLevel(logging.INFO)

# Add a handler to ensure fara_agent logs are shown
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(message)s"))
fara_agent_logger.addHandler(handler)
fara_agent_logger.propagate = False  # Don't propagate to root logger

logger = logging.getLogger(__name__)


DEFAULT_ENDPOINT_CONFIG = {
    "model": "microsoft/Fara-7B",
    "base_url": "http://localhost:5000/v1",
    "api_key": "not-needed",
}


async def run_fara_agent(
    initial_task: str = None,
    endpoint_config: Dict[str, str] = None,
    start_page: str = "https://www.bing.com/",
    headless: bool = True,
    downloads_folder: str = None,
    save_screenshots: bool = True,
    max_rounds: int = 100,
    use_browser_base: bool = False,
):
    # Initialize browser manager
    print("Initializing Browser...")
    browser_manager = BrowserBB(
        headless=headless,
        viewport_height=900,
        viewport_width=1440,
        page_script_path=None,
        browser_channel="firefox",
        browser_data_dir=None,
        downloads_folder=downloads_folder,
        to_resize_viewport=True,
        single_tab_mode=True,
        animate_actions=False,
        use_browser_base=use_browser_base,
        logger=logger,
    )
    print("Browser Running... Starting Fara Agent...")

    agent = FaraAgent(
        browser_manager=browser_manager,
        client_config=endpoint_config,
        start_page=start_page,
        downloads_folder=downloads_folder,
        save_screenshots=save_screenshots,
        max_rounds=max_rounds,
    )

    try:
        await agent.initialize()

        # Interactive loop
        task = initial_task
        first_round = True

        while True:
            if task is None:
                if first_round:
                    task = input("Enter task: ").strip()
                else:
                    task = input(
                        "\nEnter another task (or press Enter to exit): "
                    ).strip()

                if not task:
                    print("Exiting...")
                    break

            print("##########################################")
            print(f"Task: {task}")
            print("##########################################")

            try:
                print("Running Fara...\n")
                final_answer, all_actions, all_observations = await agent.run(task)
                print(f"\nFinal Answer: {final_answer}")
            except Exception as e:
                print(f"Error occurred: {e}")
            task = None
            first_round = False

    finally:
        # Close the agent and browser
        await agent.close()


def main():
    """CLI entry point for fara command."""
    parser = argparse.ArgumentParser(description="Run FARA agent interactively")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="Initial task for the FARA agent (optional)",
    )
    parser.add_argument(
        "--start_page",
        type=str,
        default="https://www.bing.com/",
        help="The starting page",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run the browser in headful mode (show GUI, default is headless)",
    )
    parser.add_argument(
        "--downloads_folder",
        type=str,
        default=None,
        help="Folder to save screenshots and downloads",
    )
    parser.add_argument(
        "--save_screenshots",
        action="store_true",
        help="Whether to save screenshots during the agent's operation",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=100,
        help="Maximum number of rounds for the agent to run",
    )
    parser.add_argument(
        "--browserbase",
        action="store_true",
        help="Whether to use BrowserBase for browser management",
    )
    parser.add_argument(
        "--endpoint_config",
        type=Path,
        default=None,
        help="Path to the endpoint configuration JSON file. By default, tries local vllm on 5000 port",
    )

    args = parser.parse_args()

    if args.browserbase:
        assert os.environ.get("BROWSERBASE_API_KEY"), (
            "BROWSERBASE_API_KEY environment variable must be set to use browserbase"
        )
        assert os.environ.get("BROWSERBASE_PROJECT_ID"), (
            "BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables must be set to use browserbase"
        )

    endpoint_config = DEFAULT_ENDPOINT_CONFIG
    if args.endpoint_config:
        with open(args.endpoint_config, "r") as f:
            endpoint_config = json.load(f)

    asyncio.run(
        run_fara_agent(
            initial_task=args.task,
            endpoint_config=endpoint_config,
            start_page=args.start_page,
            headless=not args.headful,
            downloads_folder=args.downloads_folder,
            save_screenshots=args.save_screenshots,
            max_rounds=args.max_rounds,
            use_browser_base=args.browserbase,
        )
    )


if __name__ == "__main__":
    main()

from pathlib import Path
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import os
from collections import defaultdict
from autogen_core.components.models import RequestUsage
from autogen_ext.models._openai._openai_client import _add_usage


@dataclass
class FinalAnswer:
    final_answer: str = "<no_answer>"

    """
    env_state_json: Parsed JSON representation of the final webpage environment state retrieved from the <pre> element at /finish.
    env_state_raw: Raw text captured from the <pre> element before JSON parsing, used for debugging or fallback when parsing fails.
    """
    env_state_json: str = "<no_answer>"
    env_state_raw: str = "<no_answer>"

    screenshots: List[str] = field(default_factory=list)
    is_aborted: bool = False
    is_rel_paths: bool = True   # True as default, but missing value is interpreted
    token_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize token_usage as defaultdict if not already set
        if not isinstance(self.token_usage, defaultdict):
            token_usage_dict = self.token_usage if self.token_usage else {}
            self.token_usage = defaultdict(lambda: RequestUsage(prompt_tokens=0, completion_tokens=0))
            # Restore any existing token usage data
            for k, v in token_usage_dict.items():
                self.set_token_usage(k, v)

    def set_token_usage(self, key: str, token_usage: RequestUsage | Dict[str, int]):
        if isinstance(token_usage, RequestUsage):
            self.token_usage[key] = token_usage
        else:
            self.token_usage[key] = RequestUsage(**token_usage)

    def add_token_usage(self, key: str, token_usage: RequestUsage | Dict[str, int]):
        if isinstance(token_usage, RequestUsage):
            self.token_usage[key] = _add_usage(self.token_usage[key], token_usage)
        else:
            self.token_usage[key] = _add_usage(self.token_usage[key], RequestUsage(**token_usage))

    def to_dict(self):
        result = asdict(self)
        # Convert RequestUsage objects to dicts for JSON serialization
        if 'token_usage' in result:
            result['token_usage'] = {k: asdict(v) if isinstance(v, RequestUsage) else v for k, v in self.token_usage.items()}
        return result

    def save(self, path: os.PathLike):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: os.PathLike):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return FinalAnswer(**data)  

def remap_action_names(action_name: str) -> str:
    """
        this is needed in the case of gpt_solver, where the action names are not consistent with those of trained models
    """
    if action_name == 'stop_execution':
        return 'terminate'
    elif action_name == 'stop_and_answer_question':
        return 'terminate'
    else:
        return action_name  # Return as is if no remapping is needed

def parse_text_based_event(event: Dict) -> Dict | None:
    """
    Parse events where thoughts and actions are embedded as text in the message field.

    Expected format:
    "Thought #X: <thought text>
    Action #X: executing tool '<tool_name>' with arguments {<json>}"

    Returns a dict with 'action' and 'arguments' fields, or None if not a thought/action event.
    """
    import re

    message = event.get('message', '')

    # Check if this is a thought/action message
    if 'Thought #' not in message or 'Action #' not in message:
        return None

    try:
        # Extract thought
        thought_match = re.search(r'Thought #\d+:\s*(.+?)(?=\nAction #)', message, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Extract action arguments JSON
        # Pattern: executing tool '<tool_name>' with arguments {json}
        action_match = re.search(r'with arguments\s+(\{.+\})', message, re.DOTALL)
        if not action_match:
            return None

        arguments = json.loads(action_match.group(1))

        # Add thoughts to arguments
        arguments['thoughts'] = thought

        # Get action name from arguments
        action_name = arguments.get('action', 'unknown')

        return {
            'action': action_name,
            'arguments': arguments
        }
    except (json.JSONDecodeError, AttributeError):
        # Failed to parse - return None
        return None

class Trajectory:
    def __init__(self, path, gpt_solver = False, skip_web_surfer_log = False):
        self.path = Path(path)
        
        # Handle web_surfer.log (may not exist for specialized trajectory types)
        if skip_web_surfer_log:
            self.events = []
        else:
            with open(self.path / 'web_surfer.log') as f:
                self.events = [json.loads(l) for l in f.readlines()]
        self.latest_screenshot = self.path / 'screenshot_scaled.png'
        answer_files = list(self.path.glob('*_answer.json'))
        if len(answer_files) != 1:
            raise ValueError(f"Expected exactly one answer file in {self.path}, found {len(answer_files)}")
        self.answer = FinalAnswer.load(answer_files[0])
        if self.answer and self.answer.is_rel_paths:
            self.answer.screenshots = [self.path / f for f in self.answer.screenshots]
        self.screenshots = self.answer.screenshots
        
        # Load metadata for instruction following compatibility (is_action field)
        self.is_action = False
        metadata_path = self.path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.is_action = metadata.get("is_action", False)
        
        # Check if events are text-based (no 'action' field in any event)
        has_structured_events = any(e.get('action') is not None for e in self.events)

        if not has_structured_events and self.events:
            # Parse text-based events
            parsed_events = []
            for event in self.events:
                parsed = parse_text_based_event(event)
                if parsed:
                    parsed_events.append(parsed)
            self.events = parsed_events

        if gpt_solver:
            #  remove non-WebSurfer events e.g. WebSurfer-SummarizedAction and other miscellaneous comments from solving pipeline
            self.events = [e for e in self.events if (e.get('source', None) == "WebSurfer" and e.get('action', None) is not None)]
            # For gpt_solver, normalize events to have action in arguments for compatibility
            for event in self.events:
                if event.get('action') and 'arguments' in event and isinstance(event['arguments'], dict):
                    # Move top-level action into arguments for consistency with other solvers
                    event['arguments']['action'] = remap_action_names(event['action'])
            self.actions = []
            self.thoughts = []
        else:
            self.actions = [e['arguments'] for e in self.events if e.get('action', None) is not None]
            self.thoughts = [a['thoughts'] for a in self.actions]
            self.actions = [json.dumps({k: v for k, v in a.items() if k != 'thoughts'}) for a in self.actions]
    @property
    def is_aborted(self):
        return self.answer.is_aborted

    @staticmethod
    def from_folder(path, gpt_solver = False, skip_web_surfer_log = False):
        try:
            return Trajectory(path, gpt_solver=gpt_solver, skip_web_surfer_log=skip_web_surfer_log)
        except Exception as e:
            return None
        
    def __repr__(self):
        return f'Trajectory("{self.path.name}: {len(self.screenshots)} screenshots, {len(self.actions)} actions")'



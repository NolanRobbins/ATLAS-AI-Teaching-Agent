import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta
from core.state import AcademicState
from config.llm_config import YourLLM

# Assuming AcademicState is imported from core.state
# from core.state import AcademicState # No, this should be passed as argument
# Assuming YourLLM is imported from config.llm_config
# from config.llm_config import YourLLM # No, this should be passed as argument

class AgentAction(BaseModel):
    action: str
    thought: str
    tool: Optional[str] = None
    action_input: Optional[Dict] = None

class AgentOutput(BaseModel):
    observation: str
    output: Dict

class ReActAgent:
    def __init__(self, llm_instance: Any): # Use Any for llm_instance type hinting for now
        self.llm = llm_instance
        self.few_shot_examples = []
        self.tools = {
            "search_calendar": self.search_calendar,
            "analyze_tasks": self.analyze_tasks,
            "check_learning_style": self.check_learning_style,
            "check_performance": self.check_performance
        }

    async def search_calendar(self, state: Dict) -> List[Dict]: # Use Dict for state type hinting for now
        events = state['calendar'].get("events", [])
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=7)
        return [e for e in events if datetime.fromisoformat(e['start']['dateTime']) > now]

    async def analyze_tasks(self, state: Dict) -> List[Dict]:
        return state['tasks'].get("tasks", [])

    async def check_learning_style(self, state: Dict) -> Dict:
        profile = state["profile"]
        learning_data = {
            "style": profile.get("learning_preferences", {}).get("learning_style", {}),
            "patterns": profile.get("learning_preferences", {}).get("study_patterns", {})
        }
        # Instead of modifying state in-place, return the update
        return {"results": {"learning_analysis": learning_data}}

    async def check_performance(self, state: Dict) -> Dict:
        profile = state["profile"]
        courses = profile.get("academic_info", {}).get("current_courses", [])
        # Instead of modifying state in-place, return the update
        return {"results": {"performance_analysis": {"courses": courses}}}
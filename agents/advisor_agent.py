import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

from core.react_agent import ReActAgent
from core.state import AcademicState
from config.llm_config import YourLLM

class AdvisorAgent(ReActAgent):
    def __init__(self, llm_instance: Any):
        super().__init__(llm_instance)
        self.llm = llm_instance
        self.few_shot_examples = self._initialize_fewshots()
        self.workflow = self.create_subgraph()

    def _initialize_fewshots(self):
        return [
            # ... (Your existing few-shot examples) ...
            {
                "request": "Managing multiple deadlines with limited time",
                "profile": {
                    "learning_style": "visual",
                    "workload": "heavy",
                    "time_constraints": ["2 hackathons", "project", "exam"]
                },
                "advice": """PRIORITY-BASED SCHEDULE:

                1. IMMEDIATE ACTIONS
                   • Create visual timeline of all deadlines
                   • Break each task into 45-min chunks
                   • Schedule high-focus work in mornings

                2. WORKLOAD MANAGEMENT
                   • Hackathons: Form team early, set clear roles
                   • Project: Daily 2-hour focused sessions
                   • Exam: Interleaved practice with breaks

                3. ENERGY OPTIMIZATION
                   • Use Pomodoro (25/5) for intensive tasks
                   • Physical activity between study blocks
                   • Regular progress tracking

                4. EMERGENCY PROTOCOLS
                   • If overwhelmed: Take 10min reset break
                   • If stuck: Switch tasks or environments
                   • If tired: Quick power nap, then review"""
            }
        ]

    def create_subgraph(self) -> StateGraph:
        subgraph = StateGraph(Dict) # Using Dict for AcademicState
        subgraph.add_node("advisor_analyze", self.analyze_situation)
        subgraph.add_node("advisor_generate", self.generate_guidance)
        subgraph.add_edge("advisor_analyze", "advisor_generate")
        subgraph.set_entry_point("advisor_analyze")
        subgraph.add_edge("advisor_generate", END)
        return subgraph.compile()

    async def analyze_situation(self, state: Dict) -> Dict:
        profile = state["profile"]
        learning_prefs = profile.get("learning_preferences", {})
        prompt = f"""Analyze student situation and determine guidance approach:

        CONTEXT:
        - Profile: {json.dumps(profile, indent=2)}
        - Learning Preferences: {json.dumps(learning_prefs, indent=2)}
        - Request: {state['messages'][-1].content}

        ANALYZE:
        1. Current challenges
        2. Learning style compatibility
        3. Time management needs
        4. Stress management requirements
        """
        response = await self.llm.agenerate([{"role": "system", "content": prompt}])
        return {"results": {"situation_analysis": {"analysis": response}}}

    async def generate_guidance(self, state: Dict) -> Dict:
        analysis = state["results"].get("situation_analysis", {}).get("analysis", "") # Adjusted path
        prompt = f"""Generate personalized academic guidance based on analysis:

        ANALYSIS: {analysis}
        EXAMPLES: {json.dumps(self.few_shot_examples, indent=2)}

        FORMAT:
        1. Immediate Action Steps
        2. Schedule Optimization
        3. Energy Management
        4. Support Strategies
        5. Emergency Protocols
        """
        response = await self.llm.agenerate([{"role": "system", "content": prompt}])
        return {"results": {"guidance": {"advice": response}}}

    async def __call__(self, state: Dict) -> Dict:
        final_state_advisor = await self.workflow.ainvoke(state)
        return final_state_advisor.get("results", {})
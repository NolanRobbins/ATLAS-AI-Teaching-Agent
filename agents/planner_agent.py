# agents/planner_agent.py
import json
from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta
from langgraph.graph import StateGraph, END, START # Import START here if it's used in subgraphs

from core.react_agent import ReActAgent
from core.state import AcademicState # Will be passed as argument
from config.llm_config import YourLLM # Will be passed as argument

class PlannerAgent(ReActAgent):
    def __init__(self, llm_instance: Any):
        super().__init__(llm_instance)
        self.llm = llm_instance
        self.few_shot_examples = self._initialize_fewshots()
        self.workflow = self.create_subgraph()

    def _initialize_fewshots(self):
        return [
            # ... (Your existing few-shot examples) ...
            {
                "input": "Help with exam prep while managing ADHD and football",
                "thought": "Need to check calendar conflicts and energy patterns",
                "action": "search_calendar",
                "observation": "Football match at 6PM, exam tomorrow 9AM",
                "plan": """ADHD-OPTIMIZED SCHEDULE:
                    PRE-FOOTBALL (2PM-5PM):
                    - 3x20min study sprints
                    - Movement breaks
                    - Quick rewards after each sprint

                    FOOTBALL MATCH (6PM-8PM):
                    - Use as dopamine reset
                    - Formula review during breaks

                    POST-MATCH (9PM-12AM):
                    - Environment: Café noise
                    - 15/5 study/break cycles
                    - Location changes hourly

                    EMERGENCY PROTOCOLS:
                    - Focus lost → jumping jacks
                    - Overwhelmed → room change
                    - Brain fog → cold shower"""
            },
            {
                "input": "Struggling with multiple deadlines",
                "thought": "Check task priorities and performance issues",
                "action": "analyze_tasks",
                "observation": "3 assignments due, lowest grade in Calculus",
                "plan": """PRIORITY SCHEDULE:
                    HIGH-FOCUS SLOTS:
                    - Morning: Calculus practice
                    - Post-workout: Assignments
                    - Night: Quick reviews

                    ADHD MANAGEMENT:
                    - Task timer challenges
                    - Reward system per completion
                    - Study buddy accountability"""
            }
        ]

    def create_subgraph(self) -> StateGraph:
        # Pass AcademicState directly or use Dict here if it's not imported at module level
        subgraph = StateGraph(Dict) # Using Dict as a placeholder for AcademicState
        subgraph.add_node("calendar_analyzer", self.calendar_analyzer)
        subgraph.add_node("task_analyzer", self.task_analyzer)
        subgraph.add_node("plan_generator", self.plan_generator)
        subgraph.add_edge("calendar_analyzer", "task_analyzer")
        subgraph.add_edge("task_analyzer", "plan_generator")
        subgraph.set_entry_point("calendar_analyzer")
        subgraph.add_edge("plan_generator", END) # Make sure this ends somewhere
        return subgraph.compile()

    async def calendar_analyzer(self, state: Dict) -> Dict:
        events = state["calendar"].get("events", [])
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=7)
        filtered_events = [
            event for event in events if now <= datetime.fromisoformat(event['start']["dateTime"]) <= future
        ]
        prompt = """Analyze calendar events and identify:
        Events: {events}

        Focus on:
        - Available time blocks
        - Energy impact of activities
        - Potential conflicts
        - Recovery periods
        - Study opportunity windows
        - Activity patterns
        - Schedule optimization
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(filtered_events)}
        ]
        response = await self.llm.agenerate(messages)
        return {"results": {"calendar_analysis": {"analysis": response}}}

    async def task_analyzer(self, state: Dict) -> Dict:
        tasks = state["tasks"].get("tasks", [])
        prompt = """Analyze tasks and create priority structure:
        Tasks: {tasks}

        Consider:
        - Urgency levels
        - Task complexity
        - Energy requirements
        - Dependencies
        - Required focus levels
        - Time estimations
        - Learning objectives
        - Success criteria
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(tasks)}
        ]
        response = await self.llm.agenerate(messages)
        return {"results": {"task_analysis": {"analysis": response}}}

    async def plan_generator(self, state: Dict) -> Dict:
        # Ensure these keys exist from previous steps in the subgraph
        profile_analysis = state["results"].get("profile_analysis", {}) # This might come from profile_analyzer node outside this subgraph
        calendar_analysis = state["results"].get("calendar_analysis", {})
        task_analysis = state["results"].get("task_analysis", {})

        prompt = f"""AI Planning Assistant: Create focused study plan using ReACT framework.

          INPUT CONTEXT:
          - Profile Analysis: {profile_analysis}
          - Calendar Analysis: {calendar_analysis}
          - Task Analysis: {task_analysis}

          EXAMPLES:
          {json.dumps(self.few_shot_examples, indent=2)}

          INSTRUCTIONS:
          1. Follow ReACT pattern:
            Thought: Analyze situation and needs
            Action: Consider all analyses
            Observation: Synthesize findings
            Plan: Create structured plan

          2. Address:
            - ADHD management strategies
            - Energy level optimization
            - Task chunking methods
            - Focus period scheduling
            - Environment switching tactics
            - Recovery period planning
            - Social/sport activity balance

          3. Include:
            - Emergency protocols
            - Backup strategies
            - Quick wins
            - Reward system
            - Progress tracking
            - Adjustment triggers

          Pls act as an intelligent tool to help the students reach their goals or overcome struggles and answer with informal words.

          FORMAT:
          Thought: [reasoning and situation analysis]
          Action: [synthesis approach]
          Observation: [key findings]
          Plan: [actionable steps and structural schedule]
          """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": state["messages"][-1].content}
        ]
        response = await self.llm.agenerate(messages, temperature=0.5)

        return {"results": {"final_plan": {"plan": response}}}

    async def __call__(self, state: Dict) -> Dict: # This is the entry point from the main graph
        # This will invoke the internal workflow of the PlannerAgent
        final_state_planner = await self.workflow.ainvoke(state)
        # Extract the relevant output from the final state of the subgraph
        # and merge it back into the main state's results
        return final_state_planner.get("results", {}) # Return just the results part
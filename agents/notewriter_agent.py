# agents/notewriter_agent.py
import json
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from core.react_agent import ReActAgent

class NoteWriterAgent(ReActAgent):
    def __init__(self, llm_instance: Any):
        super().__init__(llm_instance)
        self.llm = llm_instance
        self.few_shot_examples = self._initialize_fewshots()
        self.workflow = self.create_subgraph()

    def _initialize_fewshots(self):
        return [
            # ... (Your existing few-shot examples) ...
            {
                "input": "Need to cram Calculus III for tomorrow",
                "template": "Quick Review",
                "notes": """CALCULUS III ESSENTIALS:

                1. CORE CONCEPTS (80/20 Rule):
                   • Multiple Integrals → volume/area
                   • Vector Calculus → flow/force/rotation
                   • KEY FORMULAS:
                     - Triple integrals in cylindrical/spherical coords
                     - Curl, divergence, gradient relationships

                2. COMMON EXAM PATTERNS:
                   • Find critical points
                   • Calculate flux/work
                   • Optimize with constraints

                3. QUICKSTART GUIDE:
                   • Always draw 3D diagrams
                   • Check units match
                   • Use symmetry to simplify

                4. EMERGENCY TIPS:
                   • If stuck, try converting coordinates
                   • Check boundary conditions
                   • Look for special patterns"""
            }
        ]

    def create_subgraph(self) -> StateGraph:
        subgraph = StateGraph(Dict) # Using Dict for AcademicState
        subgraph.add_node("notewriter_analyze", self.analyze_learning_style)
        subgraph.add_node("notewriter_generate", self.generate_notes)
        subgraph.add_edge("notewriter_analyze", "notewriter_generate")
        subgraph.set_entry_point("notewriter_analyze")
        subgraph.add_edge("notewriter_generate", END)
        return subgraph.compile()

    async def analyze_learning_style(self, state: Dict) -> Dict:
        profile = state["profile"]
        learning_style = profile.get("learning_preferences", {}).get("learning_style", {})
        prompt = f"""Analyze content requirements and determine optimal note structure:

        STUDENT PROFILE:
        - Learning Style: {json.dumps(learning_style, indent=2)}
        - Request: {state['messages'][-1].content}

        FORMAT:
        1. Key Topics (80/20 principle)
        2. Learning Style Adaptations
        3. Time Management Strategy
        4. Quick Reference Format

        FOCUS ON:
        - Essential concepts that give maximum understanding
        - Visual and interactive elements
        - Time-optimized study methods
        """
        response = await self.llm.agenerate([{"role": "system", "content": prompt}])
        return {"results": {"learning_analysis": {"analysis": response}}}

    async def generate_notes(self, state: Dict) -> Dict:
        analysis = state["results"].get("learning_analysis", {}).get("analysis", "") # Adjusted path
        learning_style = state["profile"].get("learning_preferences", {}).get("learning_style", {}) # Adjusted path

        prompt = f"""Create concise, high-impact study materials based on analysis:

        ANALYSIS: {analysis}
        LEARNING STYLE: {json.dumps(learning_style, indent=2)}
        REQUEST: {state['messages'][-1].content}

        EXAMPLES:
        {json.dumps(self.few_shot_examples, indent=2)}

        FORMAT:
        **THREE-WEEK INTENSIVE STUDY PLANNER**

        [Generate structured notes with:]
        1. Weekly breakdown
        2. Daily focus areas
        3. Core concepts
        4. Emergency tips
        """
        response = await self.llm.agenerate([{"role": "system", "content": prompt}])
        return {"results": {"generated_notes": {"notes": response}}}

    async def __call__(self, state: Dict) -> Dict:
        final_state_notewriter = await self.workflow.ainvoke(state)
        return final_state_notewriter.get("results", {})
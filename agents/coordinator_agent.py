# agents/coordinator_agent.py
import json
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage # If AcademicState uses this type directly
from core.state import AcademicState
from config.llm_config import YourLLM

# Assuming AcademicState and YourLLM are passed in context or imported locally
# from core.state import AcademicState
# from config.llm_config import YourLLM

COORDINATOR_PROMPT = """You are a Coordinator Agent using ReACT framework to orchestrate multiple academic support agents.

        AVAILABLE AGENTS:
        • PLANNER: Handles scheduling and time management
        • NOTEWRITER: Creates study materials and content summaries
        • ADVISOR: Provides personalized academic guidance

        PARALLEL EXECUTION RULES:
        1. Group compatible agents that can run concurrently
        2. Maintain dependencies between agent executions
        3. Coordinate results from parallel executions

        REACT PATTERN:
        Thought: [Analyze request complexity and required support types]
        Action: [Select optimal agent combination]
        Observation: [Evaluate selected agents' capabilities]
        Decision: [Finalize agent deployment plan]

        ANALYSIS POINTS:
        1. Task Complexity and Scope
        2. Time Constraints
        3. Resource Requirements
        4. Learning Style Alignment
        5. Support Type Needed

        CONTEXT:
        Request: {request}
        Student Context: {context}

        FORMAT RESPONSE AS:
        Thought: [Analysis of academic needs and context]
        Action: [Agent selection and grouping strategy]
        Observation: [Expected workflow and dependencies]
        Decision: [Final agent deployment plan with rationale]
        """

async def analyze_context(state: Dict) -> Dict: # Use Dict for state type hinting
    profile = state.get("profile", {})
    calendar = state.get("calendar", {})
    tasks = state.get("tasks", {})

    courses = profile.get("academic_info", {}).get("current_courses", [])
    current_course = None
    request = state["messages"][-1].content.lower()

    for course in courses:
        if course["name"].lower() in request:
            current_course = course
            break

    return {
        "student": {
            "major": profile.get("personal_info", {}).get("major", "Unknown"),
            "year": profile.get("personal_info", {}).get("academic_year"),
            "learning_style": profile.get("learning_preferences", {}).get("learning_style", {})
        },
        "course": current_course,
        "upcoming_events": len(calendar.get("events", [])),
        "active_tasks": len(tasks.get("tasks", [])),
        "study_patterns": profile.get("learning_preferences", {}).get("study_patterns", {})
    }

def parse_coordinator_response(response: str) -> Dict:
    try:
        analysis = {
            "required_agents": ["PLANNER"],
            "priority": {"PLANNER": 1},
            "concurrent_groups": [["PLANNER"]],
            "reasoning": "Default coordination"
        }

        if "Thought:" in response and "Decision:" in response:
            if "NoteWriter" in response or "note" in response.lower():
                analysis["required_agents"].append("NOTEWRITER")
                analysis["priority"]["NOTEWRITER"] = 2
                analysis["concurrent_groups"] = [["PLANNER", "NOTEWRITER"]]

            if "Advisor" in response or "guidance" in response.lower():
                analysis["required_agents"].append("ADVISOR")
                analysis["priority"]["ADVISOR"] = 3
                # ADVISOR typically runs after initial planning, so it might be a separate group or later in the sequence

            thought_section_match = response.split("Thought:")[1].split("Action:")[0].strip() if "Thought:" in response and "Action:" in response else None
            analysis["reasoning"] = thought_section_match if thought_section_match else analysis["reasoning"]

        return analysis
    except Exception as e:
        print(f'Parse error in coordinator response: {str(e)}')
        return {
            "required_agents": ["PLANNER"],
            "priority": {"PLANNER": 1},
            "concurrent_groups": [["PLANNER"]],
            "reasoning": "Fallback due to parse error"
        }

async def coordinator_agent(state: Dict, llm_instance: Any) -> Dict: # Pass llm_instance as argument
    try:
        context = await analyze_context(state)
        query = state['messages'][-1].content

        prompt = COORDINATOR_PROMPT

        response = await llm_instance.agenerate([
            {"role": "system", "content": prompt.format(
                request = query,
                context = json.dumps(context, indent=2)
            )}
        ])

        analysis = parse_coordinator_response(response)
        return {
            "results": {
                "coordinator_analysis": {
                    "required_agents": analysis.get("required_agents", ["PLANNER"]),
                    "priority": analysis.get("priority", {"PLANNER": 1}),
                    "concurrent_groups": analysis.get("concurrent_groups", [["PLANNER"]]),
                    "response": response
                }
            }
        }
    except Exception as e:
        print(f"Coordinator error: {e}")
        return {
            "results": {
                "coordinator_analysis": {
                    "required_agents": ["PLANNER"],
                    "priority": {"PLANNER": 1},
                    "concurrent_groups": [["PLANNER"]],
                    "reasoning": "Error in coordination. Falling back to planner."
                }
            }
        }
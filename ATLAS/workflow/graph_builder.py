from langgraph.graph import StateGraph, END, START
from typing import List, Dict, Any, Union, Literal

# Import components
from core.state import AcademicState, dict_reducer
from config.llm_config import YourLLM
from data.data_manager import DataManager
from agents.coordinator_agent import coordinator_agent, analyze_context, parse_coordinator_response
from agents.planner_agent import PlannerAgent
from agents.notewriter_agent import NoteWriterAgent
from agents.advisor_agent import AdvisorAgent
from executor.agent_executor import AgentExecutor

def create_agents_graph(llm_instance: YourLLM) -> StateGraph:
    workflow = StateGraph(AcademicState)

    planner_agent = PlannerAgent(llm_instance)
    notewriter_agent = NoteWriterAgent(llm_instance)
    advisor_agent = AdvisorAgent(llm_instance)
    executor = AgentExecutor(llm_instance)

    # MAIN WORKFLOW NODES
    workflow.add_node("coordinator", lambda state: coordinator_agent(state, llm_instance)) # Pass llm_instance
    # Assuming profile_analyzer function is defined in coordinator_agent.py or a utilities file
    # For now, let's move it to a common utility or an agent itself if it needs LLM
    # If profile_analyzer is a standalone function not using LLM, it could stay simple
    # For now, let's assume profile_analyzer needs an LLM or specific data,
    # it's usually better encapsulated in an Agent if it's complex.
    # Let's assume profile_analyzer needs to be a callable directly.
    # If it's the function from the notebook:
    # from agents.profile_analyzer import profile_analyzer # You need to create this file
    # For simplicity, assuming profile_analyzer is a simple function that takes state and returns results
    async def simple_profile_analyzer_node(state: Dict) -> Dict:
        # This is a placeholder for your original profile_analyzer logic
        # It's better if this is encapsulated in an agent if it uses LLM
        # For now, let's make it reflect the analysis done in the notebook's profile_analyzer
        profile = state.get("profile", {})
        analysis_summary = {
            "learning_style": profile.get("learning_preferences", {}).get("learning_style", {}),
            "study_patterns": profile.get("learning_preferences", {}).get("study_patterns", {})
            # Add more profile analysis data as needed
        }
        return {"results": {"profile_analysis": {"analysis": analysis_summary}}}

    workflow.add_node("profile_analyzer", simple_profile_analyzer_node) # Using the placeholder
    workflow.add_node("execute", executor.execute)

    # Add agent-specific entry points if they are standalone nodes in the main graph
    # For example, if you want to explicitly call planner_agent.__call__ as a node
    workflow.add_node("planner_entry", planner_agent.__call__)
    workflow.add_node("notewriter_entry", notewriter_agent.__call__)
    workflow.add_node("advisor_entry", advisor_agent.__call__)


    # Parallel Execution Routing
    def route_to_parallel_agents(state: AcademicState) -> List[str]:
        analysis = state["results"].get("coordinator_analysis", {})
        required_agents = analysis.get("required_agents", [])
        next_nodes = []

        if "PLANNER" in required_agents:
            next_nodes.append("planner_entry") # Point to the agent's main entry node
        if "NOTEWRITER" in required_agents:
            next_nodes.append("notewriter_entry")
        if "ADVISOR" in required_agents:
            next_nodes.append("advisor_entry")

        return next_nodes if next_nodes else ["planner_entry"] # Default to planner

    # Workflow Connections
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "profile_analyzer")

    workflow.add_conditional_edges(
        "profile_analyzer",
        route_to_parallel_agents,
        {
            "planner_entry": "planner_entry",
            "notewriter_entry": "notewriter_entry",
            "advisor_entry": "advisor_entry"
        }
    )

    # All agent entry nodes will lead to the 'execute' node
    # Since agent __call__ methods return results that get merged into the state,
    # you can have multiple paths converge to 'execute'.
    workflow.add_edge("planner_entry", "execute")
    workflow.add_edge("notewriter_entry", "execute")
    workflow.add_edge("advisor_entry", "execute")

    # Workflow Completion Checking
    def should_end(state: AcademicState) -> Union[Literal["coordinator"], Literal[END]]:
        analysis = state["results"].get("coordinator_analysis", {})
        # `agent_outputs` is where AgentExecutor puts its combined results.
        # This will contain keys like 'planner', 'notewriter', 'advisor'
        executed = set(state["results"].get("agent_outputs", {}).keys())
        required = set(a.lower() for a in analysis.get("required_agents", []))

        # Check if all required agents have at least one output
        if required.issubset(executed) and all(state["results"]["agent_outputs"].get(a) for a in required):
            return END
        else:
            # If not all required agents have run or produced output, loop back to coordinator?
            # This logic needs careful consideration. If agents run in parallel and return,
            # 'execute' is the point where we gather. If we need to loop, coordinator
            # might re-evaluate. This is a complex part of multi-agent orchestration.
            # For simplicity, for now, we end if execute has run.
            # Or you can re-introduce a 'Router' node after 'execute' to decide next steps.
            return END # For simplicity, let's end after execute for now.

    # Modify the should_end logic or where it routes to.
    # The original notebook had `execute` leading to `should_end` which
    # could loop back to "coordinator". This implies a multi-turn, iterative process.
    # If it's a single pass:
    workflow.add_edge("execute", END)

    # If you want the original iterative behavior:
    # workflow.add_conditional_edges(
    #     "execute",
    #     should_end,
    #     {
    #         "coordinator": "coordinator", # Loop back if more work needed
    #         END: END
    #     }
    # )

    return workflow.compile()
import asyncio
from typing import Dict, Any
from agents.planner_agent import PlannerAgent
from agents.notewriter_agent import NoteWriterAgent
from agents.advisor_agent import AdvisorAgent

class AgentExecutor:
    def __init__(self, llm_instance: Any):
        self.llm = llm_instance
        self.agents = {
            "PLANNER": PlannerAgent(llm_instance),
            "NOTEWRITER": NoteWriterAgent(llm_instance),
            "ADVISOR": AdvisorAgent(llm_instance)
        }

    async def execute(self, state: Dict) -> Dict:
        try:
            analysis = state["results"].get("coordinator_analysis", {})
            required_agents = analysis.get("required_agents", ["PLANNER"])
            concurrent_groups = analysis.get("concurrent_groups", [])

            results = {}

            for group in concurrent_groups:
                tasks = []
                for agent_name in group:
                    if agent_name in required_agents and agent_name in self.agents:
                        # Pass the current state to the agent's __call__ method
                        # which will invoke its internal subgraph
                        tasks.append(self.agents[agent_name](state))

                if tasks:
                    group_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for agent_name_in_group, result in zip(group, group_results):
                        if not isinstance(result, Exception):
                            results[agent_name_in_group.lower()] = result
                        else:
                            print(f"Error executing {agent_name_in_group}: {result}")

            if not results and "PLANNER" in self.agents:
                # If no agents ran or all failed, try the planner as a fallback
                planner_result = await self.agents["PLANNER"](state)
                results["planner"] = planner_result

            return {
                "results": {
                    "agent_outputs": results
                }
            }

        except Exception as e:
            print(f"Execution error in AgentExecutor: {e}")
            return {
                "results": {
                    "agent_outputs": {
                        "planner": {
                            "plan": "Emergency fallback plan: Please try again or contact support."
                        }
                    }
                }
            }
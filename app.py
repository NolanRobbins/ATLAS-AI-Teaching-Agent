# app.py
# Triggering CI for the third time
import streamlit as st
import asyncio
from datetime import datetime
from typing import Dict

# Import all necessary modules from your new structure
from config.llm_config import YourLLM, get_openai_key
from core.state import AcademicState
from data.data_manager import DataManager
from workflow.graph_builder import create_agents_graph
from langchain_core.messages import HumanMessage # Needed for HumanMessage

# --- Streamlit UI Components for Data Input ---

def get_profile_input():
    st.header("Student Profile Information")
    profile = {}
    st.subheader("Personal Info")
    profile["id"] = st.text_input("Student ID", value="student_123")
    profile["personal_info"] = {
        "major": st.text_input("Major", value="Computer Science"),
        "academic_year": st.selectbox("Academic Year", ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"], index=2)
    }

    st.subheader("Learning Preferences")
    profile["learning_preferences"] = {
        "learning_style": {
            "visual": st.checkbox("Visual Learner", value=True),
            "auditory": st.checkbox("Auditory Learner", value=False),
            "kinesthetic": st.checkbox("Kinesthetic Learner", value=False)
        },
        "study_patterns": {
            "peak_time": st.selectbox("Peak Study Time", ["morning", "afternoon", "evening", "night"], index=0),
            "focus_duration": st.text_input("Typical Focus Duration (e.g., '45 minutes')", value="45 minutes")
        }
    }

    st.subheader("Academic Info")
    num_courses = st.number_input("Number of Current Courses", min_value=0, value=1)
    current_courses = []
    for i in range(num_courses):
        st.write(f"Course {i+1}")
        course_name = st.text_input(f"Course Name {i+1}", key=f"course_name_{i}")
        course_grade = st.text_input(f"Course Grade {i+1}", key=f"course_grade_{i}")
        if course_name: # Only add if name is provided
            current_courses.append({"name": course_name, "grade": course_grade})
    profile["academic_info"] = {"current_courses": current_courses}

    return {"profiles": [profile]} # Wrap in "profiles" list as DataManager expects

def get_calendar_input():
    st.header("Calendar Events")
    events = []
    st.markdown("Add upcoming events. Datetime format: `YYYY-MM-DDTHH:MM:SSZ` (e.g., 2025-06-08T18:00:00Z)")
    num_events = st.number_input("Number of Events", min_value=0, value=1)

    for i in range(num_events):
        st.write(f"Event {i+1}")
        summary = st.text_input(f"Event Summary {i+1}", key=f"event_summary_{i}")
        start_date = st.date_input(f"Start Date {i+1}", key=f"event_start_date_{i}", value=datetime.now().date())
        start_time = st.time_input(f"Start Time {i+1}", key=f"event_start_time_{i}", value=datetime.now().time())

        if summary and start_date and start_time:
            start_datetime_str = f"{start_date}T{start_time}:00Z"
            events.append({
                "summary": summary,
                "start": {"dateTime": start_datetime_str}
            })
    return {"events": events}

def get_task_input():
    st.header("Academic Tasks / To-Do List")
    tasks = []
    st.markdown("Add active tasks. Due date format: `YYYY-MM-DDTHH:MM:SSZ` (e.g., 2025-06-10T23:59:00Z)")
    num_tasks = st.number_input("Number of Tasks", min_value=0, value=1)

    for i in range(num_tasks):
        st.write(f"Task {i+1}")
        title = st.text_input(f"Task Title {i+1}", key=f"task_title_{i}")
        status = st.selectbox(f"Status {i+1}", ["needsAction", "completed"], key=f"task_status_{i}")
        due_date = st.date_input(f"Due Date {i+1}", key=f"task_due_date_{i}", value=datetime.now().date())
        due_time = st.time_input(f"Due Time {i+1}", key=f"task_due_time_{i}", value=datetime.now().time())

        if title and due_date and due_time:
            due_datetime_str = f"{due_date}T{due_time}:00Z"
            tasks.append({
                "title": title,
                "status": status,
                "due": due_datetime_str
            })
    return {"tasks": tasks}

# --- Main Application Logic (Modified for Streamlit) ---

async def run_all_system_streamlit(profile_data: Dict, calendar_data: Dict, task_data: Dict, user_request: str):
    st.info(f"Processing request: {user_request}")

    llm_instance = YourLLM(get_openai_key()) # Get the LLM instance
    dm = DataManager()
    dm.load_data(profile_data, calendar_data, task_data) # Pass dicts directly

    initial_state = AcademicState(
        messages=[HumanMessage(content=user_request)],
        profile=dm.get_student_profile("student_123"), # Assuming fixed student ID
        calendar={"events": dm.get_upcoming_events()},
        tasks={"tasks": dm.get_active_tasks()},
        results={}
    )

    graph = create_agents_graph(llm_instance)

    st.subheader("Workflow Execution")
    # Streamlit doesn't directly support mermaid PNG display, consider alternatives
    # For now, omit display(Image(graph.get_graph().draw_mermaid_png()))

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    step_num = 0
    total_steps_estimate = 5 # A rough estimate for progress bar

    coordinator_output = None
    final_state = None

    # Use a placeholder for dynamic output
    output_placeholder = st.empty()

    async for step in graph.astream(initial_state):
        step_num += 1
        current_progress = min(step_num / total_steps_estimate, 1.0)
        my_bar.progress(current_progress, text=f"Executing step {step_num}...")

        step_name = list(step.keys())[0] # Get the current node name
        step_value = step[step_name]

        with output_placeholder.container():
            st.markdown(f"**Current Step:** `{step_name}`")
            if "coordinator_analysis" in step_value.get("results", {}):
                coordinator_output = step_value
                analysis = coordinator_output["results"]["coordinator_analysis"]
                st.markdown("**Selected Agents:**")
                for agent in analysis.get("required_agents", []):
                    st.markdown(f"- {agent}")
            elif step_name == "execute":
                final_state = step_value # Capture the state after executor runs

    my_bar.progress(100, text="Execution Complete!")
    st.success("Task Completed!")

    if final_state:
        agent_outputs = final_state.get("results", {}).get("agent_outputs", {}) # Corrected path for agent_outputs

        st.markdown("---")
        st.header("Final Agent Outputs")
        for agent, output_data in agent_outputs.items():
            st.markdown(f"### {agent.upper()} Output:")
            if isinstance(output_data, dict):
                # Iterate through expected keys for each agent's output structure
                if agent == "planner" and "final_plan" in output_data and "plan" in output_data["final_plan"]:
                    st.markdown(output_data["final_plan"]["plan"])
                elif agent == "notewriter" and "generated_notes" in output_data and "notes" in output_data["generated_notes"]:
                    st.markdown(output_data["generated_notes"]["notes"])
                elif agent == "advisor" and "guidance" in output_data and "advice" in output_data["guidance"]:
                    st.markdown(output_data["guidance"]["advice"])
                else: # Fallback for other unexpected structured outputs
                    st.json(output_data)
            else: # Direct string output
                st.markdown(output_data)
    return coordinator_output, final_state


# --- Streamlit App Entry Point ---

def main_app():
    st.set_page_config(page_title="ATLAS Academic Assistant", layout="wide")
    st.title("📚 ATLAS: Academic Task Learning Agent System")

    st.write("Welcome! Enter your academic details and request below to get personalized assistance.")

    # API Key Input
    api_key_configured = get_openai_key()
    if not api_key_configured:
        st.warning("Please enter your OpenAI API key in the sidebar or set it as an environment variable.")
        st.stop() # Stop execution if key isn't provided

    with st.sidebar:
        st.header("Configuration & Data Input")
        st.markdown("Provide details for your student profile, calendar, and tasks.")

        profile_data = get_profile_input()
        calendar_data = get_calendar_input()
        task_data = get_task_input()

    st.header("Your Academic Request")
    user_request = st.text_area(
        "Describe what you need help with (e.g., 'Help me prepare for my Calculus III exam tomorrow while managing my football match tonight and Data Structures assignment due soon.')",
        value="Help me prepare for my Calculus III exam tomorrow while managing my football match tonight and Data Structures assignment due soon.",
        height=150
    )

    st.markdown("---")

    if st.button("Run ATLAS Assistant", type="primary"):
        if user_request and profile_data and calendar_data and task_data:
            # Use a spinner for background processing
            with st.spinner("Initiating agents and processing request..."):
                coordinator_output, final_state = asyncio.run(
                    run_all_system_streamlit(profile_data, calendar_data, task_data, user_request)
                )
            # Results are displayed within run_all_system_streamlit
        else:
            st.error("Please ensure all data input sections are filled and provide a request.")

if __name__ == "__main__":
    main_app()
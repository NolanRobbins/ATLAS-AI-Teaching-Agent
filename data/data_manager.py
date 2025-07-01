import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta

class DataManager:
    def __init__(self):
        self.profile_data = None
        self.calendar_data = None
        self.task_data = None

    def load_data(self, profile_dict: Dict, calendar_dict: Dict, task_dict: Dict):
        # Now expects dictionaries directly, not JSON strings
        self.profile_data = profile_dict
        self.calendar_data = calendar_dict
        self.task_data = task_dict

    def get_student_profile(self, student_id: str) -> Optional[Dict]:
        if self.profile_data and "profiles" in self.profile_data:
            return next((p for p in self.profile_data["profiles"] if p["id"] == student_id), None)
        return None

    def parse_datetime(self, dt_str: str) -> datetime:
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc)
        except ValueError:
            dt = datetime.fromisoformat(dt_str)
            return dt.replace(tzinfo=timezone.utc)

    def get_upcoming_events(self, days: int = 7) -> List[Dict]:
        if not self.calendar_data:
            return []
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=days)
        events = []
        for event in self.calendar_data.get("events", []):
            try:
                start_time = self.parse_datetime(event["start"]["dateTime"])
                if now <= start_time <= future:
                    events.append(event)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not process event due to {str(e)}")
                continue
        return events

    def get_active_tasks(self) -> List[Dict]:
        if not self.task_data:
            return []
        now = datetime.now(timezone.utc)
        active_tasks = []
        for task in self.task_data.get("tasks", []):
            try:
                due_date = self.parse_datetime(task["due"])
                if task["status"] == "needsAction" and due_date > now:
                    task["due_datetime"] = due_date
                    active_tasks.append(task)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not process task due to {str(e)}")
                continue
        return active_tasks
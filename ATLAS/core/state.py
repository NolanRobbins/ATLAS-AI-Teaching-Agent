from typing import Annotated, List, Dict, TypedDict, Any, TypeVar
from operator import add
from langchain_core.messages import BaseMessage

T = TypeVar('T')

def dict_reducer(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = dict_reducer(merged[key], value)
        else:
            merged[key] = value
    return merged

class AcademicState(TypedDict):
    messages: Annotated[List[BaseMessage], add] # Use Any for BaseMessage type here if langchain_core.messages is not imported yet
    profile: Annotated[Dict, dict_reducer]
    calendar: Annotated[Dict, dict_reducer]
    tasks: Annotated[Dict, dict_reducer]
    results: Annotated[Dict[str, Any], dict_reducer]
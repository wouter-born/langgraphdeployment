from typing_extensions import TypedDict
from typing import Optional, TypedDict, List, Dict, Any, Union
from typing import Annotated
import operator

class OverallState(TypedDict):
    ReportQuery: str
    POV: list
    ReportMetadata: Annotated[List[Dict[str, Any]], operator.add]
    JsonLayout: dict
    Components: list
    JsonLayoutWithComponentConfig: Annotated[list, operator.add]
    Lists: list
    JsonLists: Annotated[list, operator.add]

#########################################################
# SPECIALIZED COMPONENT SUBCHART
#########################################################
class SpecializedComponentState(TypedDict):
    """
    State for our subchart that picks the specialized node
    and returns a config structure that updates the layout.
    """
    component: dict
    selected_node: str
    JsonLayoutWithComponentConfig: List[dict]  # same format as your main returns


class ListSubchartState(TypedDict):
    """
    Extend your state to include any fields needed
    for checking, creating, or returning a list.
    """
    List: dict              # The raw definition of a list or instructions for how to create it
    listExists: bool        # Whether the list already exists
    listType: str           # 'dynamic' or 'fixed'
    dimensions: List        # Top-level dimensions for the list 
    ReportMetadata: Annotated[List[Dict[str, Any]], operator.add]
    JsonLists: List[dict]   # The final generated JSON list(s)
    FinalList: dict         # The actual final list data (if you want it separate from JsonLists)

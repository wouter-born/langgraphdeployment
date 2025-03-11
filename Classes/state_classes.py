from typing_extensions import TypedDict
from typing import Optional, TypedDict, List, Dict, Any, Union
from typing import Annotated
import operator

from langchain_core.messages import (
    AIMessage, 
    HumanMessage
)

class OverallState(TypedDict):
    ReportQuery: str
    ConceptualDesign: str
    POV: list
    ReportMetadata: Annotated[List[Dict[str, Any]], operator.add]
    ExistingLists: Annotated[List[dict], operator.add]
    JsonLayout: dict
    JsonLayoutList: List[dict]
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
    List: dict  # The raw definition of a list or instructions for how to create it
    listExists: bool  # Whether the list already exists
    foundListID: str  # Capture the matched list name if found
    listType: str  # 'dynamic' or 'fixed'
    dimensions: List  # Top-level dimensions for the list
    ReportMetadata: Annotated[List[Dict[str, Any]], operator.add]
    JsonLists: List[dict]  # The final generated JSON list(s)
    ExistingLists: Annotated[List[dict], operator.add]  # Any existing lists that could be used as a base



#########################################################
# EDIT REPORT STATE CLASSES
#########################################################
class ModifyReportState(TypedDict):
    instruction: str
    input_json: dict
    instruction_correct: bool
    output_json: dict
    clarification_questions: Optional[str]
    json_patches: Optional[List[dict]]



#########################################################
# STORYBOARD STATE CLASSES
#########################################################

thread_config = {"configurable": {"thread_id": "2"}}

class StoryboardState(TypedDict):
    input_prompt: str
    narrative: Optional[str]
    narrative_modif: Optional[str]
    isaccurate: Optional[bool]
    metadata_feedback: Optional[str]
    POV: Optional[list]
    ReportMetadata: Optional[Annotated[List[Dict[str, Any]], operator.add]]
    ExistingLists: Optional[Annotated[List[dict], operator.add]]
    isvalid: Optional[bool]
    pages: Optional[dict]

#########################################################
# CHATBOT STATE CLASSES
#########################################################

class ChatbotState(TypedDict):
    #messages: Annotated[list[HumanMessage | AIMessage], operator.add]
    messages: Annotated[list[HumanMessage | AIMessage], operator.add]
    generate_pages: Optional[bool]
    pages: List[dict]
    POV: Optional[list]
    ReportMetadata: Optional[Annotated[List[Dict[str, Any]], operator.add]]
    ExistingLists: Optional[Annotated[List[dict], operator.add]]
    JsonLayoutList: Annotated[List[Dict[str, Any]], operator.add]
    final_storyboard_json : List[Dict[str, Any]]
import os
import json
import time
from collections import deque
import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Optional, TypedDict, List, Dict, Any, Union

from langgraph.constants import Send
from langchain_groq import ChatGroq
from langchain_core.messages import (
    AIMessage, 
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from langgraph.graph import StateGraph, START, END

###############################################
# SPECDEC 30k tokens/minute THROTTLING LOGIC
###############################################
MAX_TOKENS_PER_MIN = 30000
token_usage_window_specdec = deque()

def record_specdec_token_usage(token_count):
    now = time.time()
    token_usage_window_specdec.append((now, token_count))
    # Remove any usage older than 60 seconds
    while token_usage_window_specdec and (now - token_usage_window_specdec[0][0]) > 60:
        token_usage_window_specdec.popleft()

def get_specdec_tokens_last_minute():
    now = time.time()
    return sum(t[1] for t in token_usage_window_specdec if (now - t[0]) <= 60)

def approximate_token_count(conversation: List[BaseMessage]) -> int:
    """
    A naive approach to estimate token usage by counting words.
    If total words > 6000, we'll avoid SpecDec (it can handle ~8000 tokens).
    In a production environment, consider a more accurate token counting method.
    """
    total_words = 0
    for msg in conversation:
        total_words += len(msg.content.split())
    return total_words

def safe_invoke_specdec(structured_llm, conversation, **kwargs):
    """
    Invokes SpecDec LLM calls while ensuring we:
     1) Avoid using SpecDec if the input tokens exceed 6000.
     2) Use Versatile if rolling token usage + input + expected output > 28k.
        (Since SpecDec has an ~8k token limit total, and we're accounting for 2k output tokens.)
    """
    # Approximate token usage of the input conversation
    input_tokens_approx = approximate_token_count(conversation)
    
    # Estimate total tokens after the request
    expected_output_tokens = 2000  # Fixed assumption as per logic
    rolling_token_usage = get_specdec_tokens_last_minute()
    estimated_total_tokens = rolling_token_usage + input_tokens_approx + expected_output_tokens

    # Check if we should fallback to modelVers
    if input_tokens_approx > 6000 or estimated_total_tokens > 28000:
        # Fallback to modelVers with the same structured output schema
        fallback_llm = modelVers.with_structured_output(
            structured_llm.output_schema,
            method=structured_llm.method,
            include_raw=structured_llm.include_raw
        )
        output = fallback_llm.invoke(conversation, **kwargs)
        return output

    # Proceed with SpecDec if limits are respected
    output = structured_llm.invoke(conversation, **kwargs)

    # Record the actual tokens used for SpecDec to update rolling token usage
    used_tokens = output.get("response_metadata", {}).get("token_usage", {}).get("total_tokens", 0)
    record_specdec_token_usage(used_tokens)
    return output



os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("CUSTOM_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("CUSTOM_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("CUSTOM_API_KEY", "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")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("CUSTOM_PROJECT", "CFOLytics_reportgenerator")

# Manually set the __file__ variable to the notebook's directory
__file__ = os.path.abspath("notebook_name.ipynb")

def load_xml_instructions(filename: str) -> str:
    """
    Load system instructions from 'XML_instructions/filename' if you keep them externally.
    Otherwise, just inline your prompts as strings.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "XML_instructions", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Prompts we will use
layout_prompt = load_xml_instructions("render_layout.xml")
component_prompt = load_xml_instructions("component_content_gen.xml")

# LLM
modelVers = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key="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",
    disable_streaming=True
)

modelSpec = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-specdec",
    api_key="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",
    disable_streaming=True
)

class Components(BaseModel):
    Components: list[str]

class ComponentConfig(BaseModel):
    config : dict

class OverallState(TypedDict):
    ReportQuery: str
    POV: list
    ReportMetadata: Annotated[List[Dict[str, Any]], operator.add]
    JsonLayout: dict
    Components: list
    JsonLayoutWithComponentConfig: Annotated[list, operator.add]
    Lists: list
    JsonLists: Annotated[list, operator.add]

class NestedNumberFormat(BaseModel):
    scale: str
    decimals: int

from typing import List, Optional, Union
from pydantic import BaseModel, Field

# Define the lowest-level models first
class GridColumns(BaseModel):
    sm: Union[int, str]
    md: Union[int, str]
    lg: Union[int, str]

    class Config:
        extra = "forbid"

class ColSpan(BaseModel):
    sm: Union[int, str]
    md: Union[int, str]
    lg: Union[int, str]

    class Config:
        extra = "forbid"

class NestedNumberFormat(BaseModel):
    scale: str
    decimals: int

    class Config:
        extra = "forbid"

# Define rows and columns next
class Component(BaseModel):
    id: str
    type: str
    title: Optional[str] = None
    AI_Generation_Description: Optional[str] = Field(
        None,
        alias="AI Generation Description"
    )
    noborder: Optional[bool] = None
    height: Optional[int] = None
    numberFormat: Optional[NestedNumberFormat] = None
    config: Optional["Layout"] = None  # Use forward reference for recursive Layout

    class Config:
        populate_by_name = True
        extra = "forbid"

class Column(BaseModel):
    colSpan: ColSpan
    components: List[Component]

    class Config:
        extra = "forbid"

class Row(BaseModel):
    columns: List[Column]

    class Config:
        extra = "forbid"

# Define layout after rows and columns
class Layout(BaseModel):
    gridColumns: Optional[GridColumns] = None
    rows: List[Row]

    class Config:
        extra = "forbid"

# Define the top-level models
class NumberFormat(BaseModel):
    currency: str
    scale: str
    decimals: int

    class Config:
        extra = "forbid"

class ReportConfig(BaseModel):
    reportTitle: str
    numberFormat: NumberFormat
    layout: Layout

    class Config:
        extra = "forbid"

def generate_layout(state: OverallState):
    system_instructions = load_xml_instructions("render_layout.xml")
    system_msg = SystemMessage(content=system_instructions)
    user_msg = HumanMessage(content=state["ReportQuery"])
    report_metadata = state["ReportMetadata"]

    structured_llm = modelSpec.with_structured_output(
        ReportConfig,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]

    # Use safe_invoke_specdec to ensure we don't exceed token limits
    output = safe_invoke_specdec(structured_llm, conversation, stream=False, response_format="json")

    if output["parsed"] is None:
        # Construct a meaningful error message
        error_message = (
            "Parsing failed: The 'parsed' field in the output is None.\n"
            "Raw LLM Output:\n"
            f"{output.get('raw', 'No raw output available')}\n\n"
            "Additional Context:\n"
            f"Model Name: {output.get('response_metadata', {}).get('model_name', 'Unknown')}\n"
            f"Token Usage: {output.get('response_metadata', {}).get('token_usage', 'Unknown')}\n"
            f"Finish Reason: {output.get('response_metadata', {}).get('finish_reason', 'Unknown')}\n"
            "Please check the raw output for errors or unexpected formatting."
        )
        raise Exception(error_message)

    parsed_output = output["parsed"].model_dump()

    # Find components
    components = []
    def walk(obj):
        if isinstance(obj, dict):
            if "components" in obj and isinstance(obj["components"], list):
                for c in obj["components"]:
                    components.append(c)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(parsed_output)
    
    parsed_output["POV"] = state["POV"]

    # Output both JsonLayout and Components
    return {"JsonLayout": parsed_output, "Components": components, "ReportMetadata": report_metadata}


def continue_to_components(state: OverallState):
    return [Send("generate_component", {"component": c}) for c in state["Components"]]

def continue_to_lists(state: OverallState):
    return [Send("generate_list_subchart",{"List": l,"ReportMetadata": state["ReportMetadata"] }) for l in state["Lists"]]

class ComponentState(TypedDict):
    component: dict

def generate_component(state: ComponentState):
    component = state["component"]
    ai_description = component.get("AI_Generation_Description", "None")
    ai_description = ai_description.strip() if isinstance(ai_description, str) else ai_description
    ai_description = None if ai_description == "None" else ai_description

    component_id = component.get("id", "NoId")
    if(ai_description is not None):
        system_instructions = load_xml_instructions("component_content_gen.xml")
        system_msg = SystemMessage(content=system_instructions)
        user_msg = HumanMessage(content=ai_description)
        
        structured_llm = modelSpec.with_structured_output(
            ComponentConfig,
            method="json_mode",
            include_raw=True
        )

        conversation = [system_msg] + [user_msg]
        # This is a Versatile model call, so we keep it as is

        output = safe_invoke_specdec(structured_llm, conversation, stream=False, response_format="json")
        parsed_output = output["parsed"].model_dump()
    else:
        parsed_output = {}

    # Generate configuration
    generated_config = {
        "id": component_id,
        "generatedConfig": parsed_output
    }

    return {"JsonLayoutWithComponentConfig": [generated_config]}  # Return as list

def update_json_layout(state: OverallState):
    components_configs = state["JsonLayoutWithComponentConfig"]
    updated_layout = state["JsonLayout"]

    def walk_and_update(layout, configs):
        if isinstance(layout, dict):
            if "components" in layout and isinstance(layout["components"], list):
                for component in layout["components"]:
                    for config in configs:
                        if config["id"] == component.get("id"):
                            component.update(config["generatedConfig"])
            for v in layout.values():
                walk_and_update(v, configs)
        elif isinstance(layout, list):
            for v in layout:
                walk_and_update(v, configs)

    walk_and_update(updated_layout, components_configs)
    return {"JsonLayout": updated_layout}

def gatheruniquelists(state: OverallState):
    """
    Gathers all unique 'lists' definitions from the components in 'JsonLayoutWithComponentConfig'
    and stores them in the 'Lists' and 'JsonLists' fields of the state.
    """
    new_lists = []
    
    for component_config in state.get("JsonLayoutWithComponentConfig", []):
        generated_config = component_config.get("generatedConfig", {})
        config_obj = generated_config.get("config", {})
        comp_lists = config_obj.get("lists", [])

        # Add only unique list definitions
        for list_def in comp_lists:
            if list_def not in new_lists:
                new_lists.append(list_def)

    state["Lists"] = new_lists

    return {
        "Lists": new_lists
    }

#########################################################
# 1. Extend the TypedDict for your subchart state
#########################################################
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


#########################################################
# 2. Define placeholder nodes
#########################################################

def check_if_list_exists(state: ListSubchartState):
    """
    Placeholder: checks if the list already exists.
    Set 'listExists' in the returned partial state.
    """
    # Example logic (replace with real check):
    list_exists_flag = False  # Placeholder: always assume it doesn't exist

    return {"listExists": list_exists_flag}


def return_existing_list(state: ListSubchartState):
    """
    Placeholder: if the list already exists, retrieve and return it.
    """
    # Example logic (replace with real retrieval):
    existing_list_data = {"someExistingList": True}  # Placeholder

    return {
        "FinalList": existing_list_data,
        "JsonLists": [existing_list_data]
    }

class DynamicOrFixedReply(BaseModel):
    type: str
    dimensions: list

def check_dynamic_or_fixed(state: ListSubchartState):
    """
    Uses an LLM (SpecDec) to decide if the list is 'Fixed' or 'Dynamic' and determine
    which top-level dimensions apply to the list.
    """
    current_list = state["List"]
    report_metadata = state["ReportMetadata"]
    top_level_dims = [
        {
            "name": dim["name"],
            "alias": dim["alias"],
            "members": [member["Name"] for member in dim.get("dimensionContent", [])[:3]]
        }
        for dim in report_metadata
    ]
    
    dynamic_or_fixed_prompt = load_xml_instructions("dynamic_or_fixed_prompt.xml")
    system_msg = SystemMessage(content=dynamic_or_fixed_prompt)

    user_input = {
        "listObject": current_list,
        "topLevelDimensions": top_level_dims
    }

    user_msg = HumanMessage(content=json.dumps(user_input, indent=2))
        
    structured_llm = modelSpec.with_structured_output(
        DynamicOrFixedReply,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]

    # Use safe_invoke_specdec
    output = safe_invoke_specdec(structured_llm, conversation, stream=False)
    parsed_output = output["parsed"].model_dump()

    return {
        "listType": parsed_output["type"],
        "dimensions": parsed_output["dimensions"]
    }

def build_hierarchy_string(filtered_metadata, parent_id=None, indent=0):
    """
    Convert multiple dimensions of `filtered_metadata` into a hierarchical string representation.
    """
    result = ""
    for metadata in filtered_metadata:
        dimension_content = metadata.get("dimensionContent", [])
        for item in dimension_content:
            item_parent_id = item.get("ParentID")
            if item_parent_id == {}:
                item_parent_id = None

            if item_parent_id == parent_id:
                # Add the current item's name with explicit \n and \t
                result += "\\t" * indent + f"{item['Name']}\\n"
                # Recursively add children
                result += build_hierarchy_string([metadata], parent_id=item["ID"], indent=indent + 1)
    return result

class FixedListReply(BaseModel):
    dimensions: list
    items: List[dict]

def create_fixed_list(state: ListSubchartState):
    """
    Create a fixed list using the LLM (SpecDec) and filtered metadata.
    """
    current_list = state["List"]
    all_metadata = state["ReportMetadata"]
    chosen_dims = state.get("dimensions", [])

    filtered_metadata = []
    dims = []
    for dim in all_metadata:
        if dim["name"] in chosen_dims:
            new_dim = {
                "name": dim["name"],
                "alias": dim["alias"],
                "dimensionContent": dim.get("dimensionContent", [])
            }
            filtered_metadata.append(new_dim)
            dims.append(dim["name"])

    fixedlist_prompt = load_xml_instructions("fixedlist_prompt.xml")
    system_msg = SystemMessage(content=fixedlist_prompt)

    hierarchystring = build_hierarchy_string(filtered_metadata)
    user_input = {
        "listObject": current_list,
        "dimensions": dims,
        "hierarchy": hierarchystring
    }
    user_msg = HumanMessage(content=json.dumps(user_input, indent=2))

    structured_llm = modelSpec.with_structured_output(
        FixedListReply,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg, user_msg]

    # Use safe_invoke_specdec
    output = safe_invoke_specdec(structured_llm, conversation, stream=False)
    parsed_output = output["parsed"]
    final_list = parsed_output.model_dump()

    list_name = current_list.get("list", "Unnamed List")
    named_list = {list_name: final_list}

    return {"JsonLists": [named_list] }

class DynamicListReply(BaseModel):
    dimensions: list
    type: str
    dynamicconfig: dict

def create_dynamic_list(state: ListSubchartState):
    """
    Create a dynamic list using the Versatile model (no token limit logic needed).
    """
    current_list = state["List"]
    all_metadata = state["ReportMetadata"]
    chosen_dims = state.get("dimensions", [])

    filtered_metadata = []
    for dim in all_metadata:
        if dim["name"] in chosen_dims:
            new_dim = {
                "name": dim["name"],
                "alias": dim["alias"],
                "dimensionContent": dim.get("dimensionContent", [])
            }
            filtered_metadata.append(new_dim)

    dynamiclist_prompt = load_xml_instructions("dynamiclist_prompt.xml")
    system_msg = SystemMessage(content=dynamiclist_prompt)

    user_input = {
        "listObject": current_list,
        "filteredMetadata": build_hierarchy_string(filtered_metadata)
    }
    user_msg = HumanMessage(content=json.dumps(user_input, indent=2))

    # This is a Versatile model call, so no special token limit logic
    structured_llm = modelSpec.with_structured_output(
        DynamicListReply,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg, user_msg]
    safe_invoke_specdec(structured_llm, conversation, stream=False)

    parsed_output = output["parsed"]
    final_list = parsed_output.model_dump()

    list_name = current_list.get("list", "Unnamed List")
    named_list = {list_name: final_list}

    return {"JsonLists": [named_list] }

#########################################################
# 4. Build the subgraph: define nodes & edges
#########################################################
subgraph = StateGraph(ListSubchartState)

# Add nodes
subgraph.add_node("check_if_list_exists", check_if_list_exists)
subgraph.add_node("return_existing_list", return_existing_list)
subgraph.add_node("check_dynamic_or_fixed", check_dynamic_or_fixed)
subgraph.add_node("create_fixed_list", create_fixed_list)
subgraph.add_node("create_dynamic_list", create_dynamic_list)

# Edges
subgraph.add_edge(START, "check_if_list_exists")

def list_exists_routing(state: ListSubchartState):
    if state.get("listExists") is True:
        return "return_existing_list"
    else:
        return "check_dynamic_or_fixed"

subgraph.add_conditional_edges("check_if_list_exists", list_exists_routing,
                               ["return_existing_list", "check_dynamic_or_fixed"])
subgraph.add_edge("return_existing_list", END)

def dynamic_or_fixed_routing(state: ListSubchartState):
    if state.get("listType") == "Dynamic":
        return "create_dynamic_list"
    else:
        return "create_fixed_list"

subgraph.add_conditional_edges("check_dynamic_or_fixed", dynamic_or_fixed_routing,
                               ["create_dynamic_list", "create_fixed_list"])
subgraph.add_edge("create_dynamic_list", END)
subgraph.add_edge("create_fixed_list", END)

# Compile the subchart
generate_list_subchart = subgraph.compile()

def remove_none_members(data):
    """
    Recursively removes all None members from a JSON-like dictionary or list structure.
    """
    if isinstance(data, dict):
        return {k: remove_none_members(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [remove_none_members(item) for item in data if item is not None]
    else:
        return data

def consolidate_lists_to_layout(state: OverallState):
    """
    Consolidates all the lists generated by the subgraph into the final layout JSON.
    """
    if "JsonLayout" not in state or state["JsonLayout"] is None:
        state["JsonLayout"] = {}

    if "Lists" not in state["JsonLayout"]:
        state["JsonLayout"]["lists"] = {}

    generated_lists = state.get("JsonLists", {})

    for generated_list in generated_lists:
        if isinstance(generated_list, dict):
            state["JsonLayout"]["lists"].update(generated_list)

    return {"JsonLayout": state["JsonLayout"]}

# Main graph
graph = StateGraph(OverallState)

graph.add_node("generate_layout", generate_layout)
graph.add_node("generate_component", generate_component)
graph.add_node("update_json_layout", update_json_layout)
graph.add_node("gatheruniquelists", gatheruniquelists)
graph.add_node("generate_list_subchart", generate_list_subchart)  # Subchart
graph.add_node("consolidate_lists_to_layout", consolidate_lists_to_layout)

graph.add_edge(START, "generate_layout")
graph.add_conditional_edges("generate_layout", continue_to_components, ["generate_component"])
graph.add_edge("generate_component", "update_json_layout")
graph.add_edge("update_json_layout", "gatheruniquelists")
graph.add_conditional_edges("gatheruniquelists", continue_to_lists, ["generate_list_subchart"])
graph.add_edge("generate_list_subchart", "consolidate_lists_to_layout")
graph.add_edge("consolidate_lists_to_layout", END)

# Compile the main graph
app = graph.compile()

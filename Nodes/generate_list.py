import json
from pydantic import BaseModel, Field
from langchain_core.messages import (
    AIMessage, 
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from typing import Optional, TypedDict, List, Dict, Any, Union
from langgraph.constants import Send

from Classes.state_classes import *
from Classes.llm_classes import *
from Nodes.load_xml_instructions import load_xml_instructions




##########################
# List Subchart
##########################

def check_if_list_exists(state: ListSubchartState):
    # Example logic (replace with real check):
    list_exists_flag = False  # always assume it doesn't exist
    return {"listExists": list_exists_flag}

def return_existing_list(state: ListSubchartState):
    existing_list_data = {"someExistingList": True}  # Placeholder
    return {
        "FinalList": existing_list_data,
        "JsonLists": [existing_list_data]
    }

class DynamicOrFixedReply(BaseModel):
    type: str
    dimensions: list

def check_dynamic_or_fixed(state: ListSubchartState):
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
    conversation = [system_msg, user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")
    parsed_output = output["parsed"].model_dump()

    return {
        "listType": parsed_output["type"],
        "dimensions": parsed_output["dimensions"]
    }

def build_hierarchy_string(filtered_metadata):
    """
    Build a readable tree structure for each dimension.
    Each dimension starts with a header, and then its items are printed as a tree.
    """
    result = ""
    for metadata in filtered_metadata:
        # Print the dimension header once
        header = f"Dimension: {metadata['name']}"
        if metadata.get("alias"):
            header += f" ({metadata['alias']})"
        result += header + "\n"
        # Build the tree for this dimension's items
        result += build_items(metadata.get("dimensionContent", []), parent_id=None, indent=1)
    return result

def build_items(items, parent_id, indent):
    """
    Recursively build the tree structure for items.
    Items whose 'ParentID' matches the provided parent_id are printed with the current indentation.
    Then, their children are printed recursively with an increased indent.
    """
    result = ""
    for item in items:
        # Normalize ParentID (treat {} as None)
        item_parent_id = item.get("ParentID")
        if item_parent_id == {}:
            item_parent_id = None

        if item_parent_id == parent_id:
            # Print the current item with the correct indentation
            result += "\t" * indent + f"{item['Name']}\n"
            # Recursively add its children
            result += build_items(items, parent_id=item["ID"], indent=indent + 1)
    return result


class FixedListReply(BaseModel):
    dimensions: list
    items: List[dict]

def create_fixed_list(state: ListSubchartState):
    current_list = state["List"]
    all_metadata = state["ReportMetadata"]
    chosen_dims = list(set(state.get("dimensions", [])))

    filtered_metadata = []
    dims = []
    for dim in all_metadata:
        if dim["name"] in chosen_dims and dim["name"] not in dims:
            new_dim = {
                "name": dim["name"],
                "alias": dim["alias"],
                "dimensionContent": dim.get("dimensionContent", [])
            }
            filtered_metadata.append(new_dim)
            dims.append(dim["name"])

    fixedlist_prompt = load_xml_instructions("fixedlist_prompt.xml")
    system_msg = SystemMessage(content=fixedlist_prompt)

    print(json.dumps(filtered_metadata, separators=(",", ":")))
    user_input = {
        "listObject": current_list,
        "dimensions": dims,
        "hierarchy": build_hierarchy_string(filtered_metadata)
    }
    user_msg = HumanMessage(content=json.dumps(user_input, indent=2))

    structured_llm = modelVers.with_structured_output(
        FixedListReply,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg, user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")
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

    # Build a simple string from the metadata
    user_input = {
        "listObject": current_list,
        "filteredMetadata": filtered_metadata
    }
    user_msg = HumanMessage(content=json.dumps(user_input, indent=2))

    structured_llm = modelVers.with_structured_output(
        DynamicListReply,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg, user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")
    
    parsed_output = output["parsed"]
    final_list = parsed_output.model_dump()

    list_name = current_list.get("list", "Unnamed List")
    named_list = {list_name: final_list}

    return {"JsonLists": [named_list] }

def list_exists_routing(state: ListSubchartState):
    if state.get("listExists") is True:
        return "return_existing_list"
    else:
        return "check_dynamic_or_fixed"

def dynamic_or_fixed_routing(state: ListSubchartState):
    if state.get("listType") == "Dynamic":
        return "create_dynamic_list"
    else:
        return "create_fixed_list"

# Then route each gathered list to the list subchart
def continue_to_lists(state: OverallState):
    return [Send("generate_list_subchart",{"List": l, "ReportMetadata": state["ReportMetadata"] }) for l in state["Lists"]]

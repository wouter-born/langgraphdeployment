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

class FoundListReply(BaseModel):
    found: bool
    foundListID: str

def check_if_list_exists(state: ListSubchartState):
    # Load the system instructions (with new ID-based logic)
    check_if_list_exists_xml = load_xml_instructions("check_if_list_exists_prompt.xml")
    system_msg = SystemMessage(content=check_if_list_exists_xml)

    # Prepare the user message
    candidate_list_name = state["List"].get("list", "")
    candidate_create_desc = state["List"].get("createDescription", "")

    for idx, lst in enumerate(state["ExistingLists"], start=1):
        if "ListID" not in lst:
            lst["ListID"] = f"list_{idx}"

    # IMPORTANT: include ListID in the "existingLists" data
    existing_lists_stripped = [
        {
            "ListID": lst["ListID"],
            "ListName": lst["ListName"],
            "CreateDescription": lst.get("CreateDescription", "")
        }
        for lst in state["ExistingLists"]
    ]

    user_input = {
        "existingLists": existing_lists_stripped,
        "candidateList": {
            "ListName": candidate_list_name,
            "CreateDescription": candidate_create_desc
        }
    }
    user_msg = HumanMessage(content=json.dumps(user_input, indent=2))

    # Call your structured LLM with the new FoundListReply model
    structured_llm = modelSpec.with_structured_output(
        FoundListReply,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg, user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")
    parsed_output = output["parsed"].model_dump()

    return {
        "listExists": parsed_output["found"],
        "foundListID": parsed_output["foundListID"]
    }

def return_existing_list(state: ListSubchartState):
    if not state["listExists"]:
        return {
            "JsonLists": []
        }

    found_list_id = state.get("foundListID", "")
    existing_lists = state["ExistingLists"]

    # Find the matching list by ID
    matched_list = next(
        (lst for lst in existing_lists if lst["ListID"] == found_list_id),
        None
    )
    if matched_list is None:
        # If no match found, return empty
        return {
            "JsonLists": []
        }

    # Parse the JSON string from "ListContents"
    list_contents_str = matched_list["ListContents"]
    list_contents = json.loads(list_contents_str)

    # "Add the name in front of the list" means use matched_list["ListName"] as the key
    named_list = {matched_list["ListName"]: list_contents}

    return {
        "JsonLists": [named_list]
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
    Build a tree structure for each dimension where each branch prints
    its children first, then the parent (i.e. the parent appears below its children).
    """
    result = ""
    for metadata in filtered_metadata:
        # Print the dimension header only once
        header = f"Dimension: {metadata['name']}"
        if metadata.get("alias"):
            header += f" ({metadata['alias']})"
        result += header + "\n"
        # Process the dimension's items with children-first ordering
        result += build_items_reversed(metadata.get("dimensionContent", []), parent_id=None, indent=1)
    return result

def build_items_reversed(items, parent_id, indent):
    """
    Recursively process items such that for each item whose ParentID matches `parent_id`,
    first process its children and then print the item.
    """
    result = ""
    for item in items:
        # Normalize ParentID (treat empty dict as None)
        item_parent_id = item.get("ParentID") or None
        if item_parent_id == parent_id:
            # First, process children of this item (if any)
            result += build_items_reversed(items, parent_id=item["ID"], indent=indent + 1)
            # Then, print the current item with the current indentation
            result += "\t" * indent + f"{item['Name']}\n"
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
    return [Send("generate_list_subchart",{"List": l, "ReportMetadata": state["ReportMetadata"], "ExistingLists": state["ExistingLists"] }) for l in state["Lists"]]

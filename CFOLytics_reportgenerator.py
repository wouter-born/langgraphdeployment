"""
CFOLytics_reportgenerator.py

LangGraph workflow for generating a financial report layout + component configs + lists,
without using 'add_collector'.

Requires:
- langgraph
- langchain-core
- langchain-community
- langchain-openai
"""

import os
import json
import operator
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# LangGraph / LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

# -----------------------------------------------------------------------------
# LLM Setup
# -----------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# -----------------------------------------------------------------------------
# XML Loader Utility
# -----------------------------------------------------------------------------
def load_xml_instructions(filename: str) -> str:
    """
    Load the content from an XML file that resides in the same folder as this script.
    Adjust if you store your XML differently.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "XML_instructions", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

# -----------------------------------------------------------------------------
# STATE DEFINITION
# -----------------------------------------------------------------------------
class FinalReportState(TypedDict):
    """
    Holds data throughout our report-generation process.
    """
    user_prompt: str                # The user's initial instructions
    instructions_clear: bool        # Whether instructions are considered clear
    layout_json: dict               # Generated layout JSON
    final_json: dict                # The final JSON with component configs + list data

class ReportGraphState(MessagesState, FinalReportState):
    """
    If you need to store messages plus our custom fields.
    """
    pass

# -----------------------------------------------------------------------------
# 1) VERIFY INSTRUCTIONS
# -----------------------------------------------------------------------------
def verify_instructions(state: ReportGraphState):
    """
    Node to check if the user instructions are clear or if we need clarification.
    Uses instructions from 'verify_instructions.xml'.
    """
    user_prompt = state["user_prompt"]

    system_instructions = load_xml_instructions("verify_instructions.xml")
    system_msg = SystemMessage(content=system_instructions)
    user_msg = HumanMessage(content=user_prompt)

    result = llm.invoke([system_msg, user_msg])
    text = result.content.lower()

    # Basic logic to see if LLM says "not clear" or "unclear"
    if "not clear" in text or "unclear" in text:
        return {"instructions_clear": False}
    return {"instructions_clear": True}

# -----------------------------------------------------------------------------
# 2) ASK CLARIFICATION
# -----------------------------------------------------------------------------
def ask_clarification(state: ReportGraphState):
    """
    Node that asks the user for clarifications if instructions are not clear.
    Uses instructions from 'clarification_prompt.xml'.
    """
    clarification_prompt = load_xml_instructions("clarification_prompt.xml")
    # Typically you'd display or return this prompt so the user can respond.
    # For demonstration, we'll just store it in the state.
    return {"clarification_request": clarification_prompt}

# -----------------------------------------------------------------------------
# 3) GENERATE LAYOUT JSON
# -----------------------------------------------------------------------------
def generate_layout_json(state: ReportGraphState):
    """
    Generates the layout JSON using the user's prompt, guided by 'render_layout.xml'.
    """
    user_prompt = state["user_prompt"]
    system_instructions = load_xml_instructions("render_layout.xml")

    system_msg = SystemMessage(content=system_instructions)
    user_msg = HumanMessage(content=user_prompt)

    result = llm.invoke([system_msg, user_msg])
    try:
        layout_dict = json.loads(result.content)
    except json.JSONDecodeError:
        # If LLM output isn't valid JSON, store it in an error structure.
        layout_dict = {
            "error": "Could not parse LLM output as JSON",
            "raw_llm_output": result.content
        }

    return {"layout_json": layout_dict}

# -----------------------------------------------------------------------------
# 4) USER CONFIRM LAYOUT
# -----------------------------------------------------------------------------
def user_confirm_layout(state: ReportGraphState):
    """
    Let the user confirm or reject the layout in a real UI.
    For the example, we'll assume user always approves the layout.
    """
    layout_ok = True
    return {"layout_ok": layout_ok}

# -----------------------------------------------------------------------------
# 5) GENERATE COMPONENT CONTENT (SEQUENTIAL APPROACH)
# -----------------------------------------------------------------------------
def generate_components_config(state: ReportGraphState):
    """
    Iterates over all components in the layout and uses an LLM to generate configs.
    Uses instructions from 'component_content_gen.xml'.
    
    This version does it SEQUENTIALLY (no parallel tasks) to avoid the need for add_collector.
    """
    layout_json = state.get("layout_json", {})
    if "error" in layout_json:
        # If we have an error from earlier, just return
        return {}

    # 1) Find all components
    components = []

    def find_components(obj):
        if isinstance(obj, dict):
            # If 'components' in this dict
            if "components" in obj and isinstance(obj["components"], list):
                for c in obj["components"]:
                    components.append(c)
            # Recurse deeper
            for v in obj.values():
                find_components(v)
        elif isinstance(obj, list):
            for v in obj:
                find_components(v)

    find_components(layout_json)

    # 2) For each component, call LLM
    component_instructions = load_xml_instructions("component_content_gen.xml")

    def update_config(obj, comp_id, new_config):
        if isinstance(obj, dict):
            if obj.get("id") == comp_id:
                obj["config"] = new_config
            for v in obj.values():
                update_config(v, comp_id, new_config)
        elif isinstance(obj, list):
            for v in obj:
                update_config(v, comp_id, new_config)

    for c in components:
        comp_id = c.get("id")
        desc = c.get("AI Generation Description", "")
        system_msg = SystemMessage(content=component_instructions)
        user_msg = HumanMessage(content=desc)
        result = llm.invoke([system_msg, user_msg])

        try:
            config_dict = json.loads(result.content)
        except json.JSONDecodeError:
            config_dict = {"error": "Invalid JSON from LLM", "raw": result.content}

        # 3) Insert config back into layout
        update_config(layout_json, comp_id, config_dict)

    return {"layout_json": layout_json}

# -----------------------------------------------------------------------------
# 6) IDENTIFY AND UNIFY LISTS (OPTIONAL)
# -----------------------------------------------------------------------------
def identify_and_unify_lists(state: ReportGraphState):
    """
    Example placeholder to unify identical lists. 
    Could load instructions from 'list_unification.xml' if you want LLM-based logic.
    For now, we'll do nothing fancy.
    """
    layout_json = state["layout_json"]
    # Add your logic to unify lists with same meaning here if needed
    return {"layout_json": layout_json}

# -----------------------------------------------------------------------------
# 7) CREATE LIST CONTENTS (SEQUENTIAL APPROACH)
# -----------------------------------------------------------------------------
def create_lists_contents(state: ReportGraphState):
    """
    Finds all lists in the updated layout JSON and populates them. 
    Uses instructions from 'list_creation.xml' (optional).
    """
    layout_json = state["layout_json"]

    # Gather references
    lists_found = []

    def walk_for_lists(obj):
        if isinstance(obj, dict):
            # If 'lists' in this dict
            if "lists" in obj and isinstance(obj["lists"], list):
                for l_ in obj["lists"]:
                    lists_found.append(l_)
            for v in obj.values():
                walk_for_lists(v)
        elif isinstance(obj, list):
            for v in obj:
                walk_for_lists(v)

    walk_for_lists(layout_json)

    # Suppose we do a direct approach to fill each list with members
    # If you need the LLM to interpret dimension queries, do so. Here, we'll just fill dummy data.
    for l_ in lists_found:
        ref = l_.get("listReference", "")
        # system_instructions = load_xml_instructions("list_creation.xml")
        # user_msg = HumanMessage(content=f"Populate list {ref} ...")
        # Possibly do an LLM call or direct dimension query
        # For demonstration, fill with placeholders:
        l_["list"] = ["Jan", "Feb", "Mar"]
        l_["AI Generation Description"] = "Populated with 3 months as example"

    return {"layout_json": layout_json}

# -----------------------------------------------------------------------------
# 8) FINALIZE THE REPORT JSON
# -----------------------------------------------------------------------------
def finalize_report_json(state: ReportGraphState):
    """
    Merge or finalize data. The 'layout_json' can be considered the final JSON.
    """
    layout_json = state["layout_json"]
    # Copy it into final_json
    return {"final_json": layout_json}

# -----------------------------------------------------------------------------
# BUILD THE GRAPH
# -----------------------------------------------------------------------------
builder = StateGraph(ReportGraphState)

# 1) Start => verify_instructions
builder.add_node("verify_instructions", verify_instructions)
builder.add_edge(START, "verify_instructions")

# Decide whether to proceed or ask clarifications
def instructions_decider(state: ReportGraphState):
    if state["instructions_clear"]:
        return "generate_layout_json"
    else:
        return "ask_clarification"

builder.add_conditional_edges(
    "verify_instructions",
    instructions_decider,
    ["generate_layout_json", "ask_clarification"]
)

# 2) ask_clarification => verify again (user presumably updates instructions)
builder.add_node("ask_clarification", ask_clarification)
builder.add_edge("ask_clarification", "verify_instructions")

# 3) generate_layout_json => user_confirm_layout
builder.add_node("generate_layout_json", generate_layout_json)
builder.add_edge("generate_layout_json", "user_confirm_layout")

# 4) user_confirm_layout => if OK => generate_components_config, else => generate_layout_json
builder.add_node("user_confirm_layout", user_confirm_layout)

def layout_confirmation_router(state: ReportGraphState):
    if state.get("layout_ok"):
        return "generate_components_config"
    else:
        return "generate_layout_json"

builder.add_conditional_edges(
    "user_confirm_layout",
    layout_confirmation_router,
    ["generate_components_config", "generate_layout_json"]
)

# 5) generate_components_config => unify_lists
builder.add_node("generate_components_config", generate_components_config)
builder.add_edge("generate_components_config", "identify_and_unify_lists")

# 6) unify_lists => create_lists
builder.add_node("identify_and_unify_lists", identify_and_unify_lists)
builder.add_edge("identify_and_unify_lists", "create_lists_contents")

# 7) create_lists => finalize
builder.add_node("create_lists_contents", create_lists_contents)
builder.add_edge("create_lists_contents", "finalize_report_json")

# 8) finalize => END
builder.add_node("finalize_report_json", finalize_report_json)
builder.add_edge("finalize_report_json", END)

# Compile the state graph
graph = builder.compile()

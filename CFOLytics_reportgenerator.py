"""
report_generation_workflow.py

Example LangGraph workflow for building a dynamic report-generation flow.

Requires:
- langgraph
- langchain-core
- langchain-community
- langchain-openai
- typing_extensions
- pydantic
"""

import operator
import os
from typing_extensions import TypedDict
from typing import List, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import ChatOpenAI

from langgraph.constants import Send
from langgraph.graph import START, END, MessagesState, StateGraph

# -----------------------------------------------------------------------------
# LLM Setup
# -----------------------------------------------------------------------------

# Example usage with GPT-4 or GPT-3.5, etc.
# You can adjust the model name as needed (e.g., "gpt-3.5-turbo")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# -----------------------------------------------------------------------------
# XML Loader Utility
# -----------------------------------------------------------------------------
def load_xml_instructions(filename: str) -> str:
    """
    Loads the content from an XML file in the same folder as this script.
    You can store all your LLM instructions in XML format.
    """
    # This method just reads from an XML file on disk.
    # In real usage, you might parse or do more advanced logic.
    # For now, we do a simple read:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "XML_instructions", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

# -----------------------------------------------------------------------------
# STATE DEFINITIONS
# -----------------------------------------------------------------------------

class LayoutComponent(BaseModel):
    """Placeholder Pydantic model for a single component's partial JSON."""
    id: str
    type: str
    AIGenerationDescription: Optional[str] = Field(
        default=None,
        description="Full instructions that the LLM used to generate this component"
    )
    config: dict = Field(default_factory=dict)

class LayoutJSON(BaseModel):
    """
    Example structure that holds the layout portion of the final JSON.
    This is only a partial structure – you can expand it as needed.
    """
    AIGenerationDescription: str = Field(description="User's entire prompt")
    reportTitle: str
    layout: dict

class FinalReportState(TypedDict):
    """
    The final state that we track in the workflow.
    """
    # The user’s prompt or instructions
    user_prompt: str
    # Whether instructions are clear or not
    instructions_clear: bool
    # The layout JSON we generate
    layout_json: dict
    # The final JSON after content generation
    final_json: dict

# -----------------------------------------------------------------------------
# 1) Verify Instructions Node
# -----------------------------------------------------------------------------

def verify_instructions(state: FinalReportState):
    """
    Node to check if the user instructions are clear or if we need clarification.
    Uses an LLM with instructions from 'verify_instructions.xml'.
    """
    user_prompt = state["user_prompt"]

    # Load the system instructions from an external XML file
    system_instructions = load_xml_instructions("verify_instructions.xml")
    
    # We'll pass the user_prompt to the LLM to see if it's ambiguous
    # This example is simplistic: we ask the LLM, "Are the instructions clear or not?"
    # The LLM then can respond with "clear" or "not clear", or we can parse a structured output.
    
    system_msg = SystemMessage(content=system_instructions)
    user_msg = HumanMessage(content=user_prompt)
    result = llm.invoke([system_msg, user_msg])

    # We expect the LLM to output something we can parse, e.g. the word "clear" or "not_clear"
    # Example approach (very simplistic):
    text = result.content.lower()
    if "not clear" in text or "unclear" in text:
        return {"instructions_clear": False}
    # Otherwise assume it's clear
    return {"instructions_clear": True}

# -----------------------------------------------------------------------------
# 2) Ask Clarification Node
# -----------------------------------------------------------------------------

def ask_clarification(state: FinalReportState):
    """
    Node that asks the user for clarifications if instructions are not clear.
    Uses instructions from 'clarification_prompt.xml' to craft a question.
    """
    # The system tries to form a clarifying question to the user
    system_instructions = load_xml_instructions("clarification_prompt.xml")

    # We'll store a textual prompt in the state so the user can respond.
    return {
        # In LangGraph Studio, you can intercept this output or show it to the user:
        "clarification_request": system_instructions
    }

# -----------------------------------------------------------------------------
# 3) Generate Layout Node
# -----------------------------------------------------------------------------

def generate_layout_json(state: FinalReportState):
    """
    Node that generates a partial layout JSON structure using the user's prompt.
    Uses instructions from 'render_layout.xml'.
    """
    user_prompt = state["user_prompt"]
    system_instructions = load_xml_instructions("render_layout.xml")
    
    system_msg = SystemMessage(content=system_instructions)
    human_msg = HumanMessage(content=user_prompt)
    
    # You may want to parse structured output. For simplicity, we’ll assume
    # you want a raw JSON string or a dictionary from the model.
    # Suppose we direct the LLM to return valid JSON in its content:
    
    # We'll do a direct LLM call. We might parse as JSON afterwards.
    result = llm.invoke([system_msg, human_msg])
    
    # Parse LLM output as JSON
    import json
    try:
        layout_dict = json.loads(result.content)
    except json.JSONDecodeError:
        # If the LLM didn't return valid JSON, fallback or raise error.
        layout_dict = {
            "error": "LLM output was not valid JSON",
            "raw_output": result.content
        }
    
    return {"layout_json": layout_dict}

# -----------------------------------------------------------------------------
# 4) User Confirms Layout Node
# -----------------------------------------------------------------------------

def user_confirm_layout(state: FinalReportState):
    """
    In a real system, we'd let the user see the layout JSON and confirm or reject.
    Here, we do a no-op. You can intercept or check if they said 'yes' in LangGraph Studio.
    """
    # For demonstration, we assume the user always says "OK".
    layout_ok = True
    return {"layout_ok": layout_ok}

# -----------------------------------------------------------------------------
# 5) Identify and Generate Component Content Node (Parallel Steps)
# -----------------------------------------------------------------------------

def identify_components(state: FinalReportState):
    """
    Identify all components in the layout that need content generation.
    Return them so we can spawn parallel tasks with Send().
    """
    layout_json = state.get("layout_json", {})
    if not layout_json:
        return {"components_to_generate": []}
    
    # Example: scan through layout JSON for "components" arrays.
    # This naive approach walks the 'layout' dictionary deeply.
    components_found = []
    
    def walk_layout(obj):
        if isinstance(obj, dict):
            # If 'components' in this dict
            if "components" in obj and isinstance(obj["components"], list):
                for c in obj["components"]:
                    comp_id = c.get("id")
                    comp_type = c.get("type")
                    desc = c.get("AI Generation Description")
                    # Save minimal info for generation
                    components_found.append({
                        "id": comp_id,
                        "type": comp_type,
                        "desc": desc
                    })
            # Recurse
            for v in obj.values():
                walk_layout(v)
        elif isinstance(obj, list):
            for v in obj:
                walk_layout(v)

    walk_layout(layout_json)
    return {"components_to_generate": components_found}


def generate_component_content(state: dict):
    """
    A node that runs *per component* to produce the actual config (charts, table, etc.).
    Uses instructions from 'component_content_gen.xml'.
    
    This function is designed for the Send() approach.
    """
    component_id = state["component_id"]
    component_desc = state.get("component_desc", "")
    system_instructions = load_xml_instructions("component_content_gen.xml")

    # We’ll feed the description to the LLM to get a config dictionary
    system_msg = SystemMessage(content=system_instructions)
    user_msg = HumanMessage(content=component_desc)

    result = llm.invoke([system_msg, user_msg])

    import json
    try:
        config_dict = json.loads(result.content)
    except json.JSONDecodeError:
        config_dict = {
            "error": "Failed to parse component config",
            "raw": result.content
        }

    return {
        "component_id": component_id,
        "config": config_dict
    }

# -----------------------------------------------------------------------------
# 6) Identify and Unify Lists Node
# -----------------------------------------------------------------------------

def identify_and_unify_lists(state: FinalReportState):
    """
    Scan the updated layout JSON for all lists, unify them if they are duplicates.
    Uses instructions from 'list_unification.xml' if you want to use LLM logic.
    Otherwise, do an algorithmic unify. This example is a placeholder.
    """
    # Example: we do a naive unify by name alone
    layout_json = state["layout_json"]

    # Insert your logic here to unify lists, e.g. "12periods" vs. "12Months"
    # For demonstration, we do nothing special.
    
    # Optionally call an LLM to interpret whether lists are duplicates
    # system_instructions = load_xml_instructions("list_unification.xml")
    # ...
    return {"layout_json": layout_json}

# -----------------------------------------------------------------------------
# 7) Create List Contents (Parallel Steps)
# -----------------------------------------------------------------------------

def identify_unique_lists(state: FinalReportState):
    """
    After unifying, gather each unique list that needs actual dimension members.
    Then we’ll do parallel tasks to fill them in.
    """
    layout_json = state["layout_json"]
    all_lists = []

    def walk_config(obj):
        if isinstance(obj, dict):
            if "lists" in obj and isinstance(obj["lists"], list):
                for l in obj["lists"]:
                    # We store the listReference and the entire sub-dict
                    all_lists.append(l)
            for v in obj.values():
                walk_config(v)
        elif isinstance(obj, list):
            for v in obj:
                walk_config(v)

    walk_config(layout_json)
    
    # Deduplicate by some logic
    # For simplicity, let's just keep them as is
    unique_lists = all_lists
    
    return {"lists_to_create": unique_lists}

def create_list_contents(state: dict):
    """
    Creates or retrieves dimension members for a given list.
    Uses instructions from 'list_creation.xml' if you need LLM help.
    Otherwise, you might call your backend (OneStream, etc.) directly.
    This is also designed for the Send() approach.
    """
    list_ref = state["listReference"]
    # Example: we do a simple stub
    # In real usage, you'd query dimension hierarchies, etc.
    # Possibly an LLM node if there's ambiguous user instructions.

    # system_instructions = load_xml_instructions("list_creation.xml")
    # ...
    # For demonstration, we’ll just return a static example
    return {
        "listReference": list_ref,
        "members": ["Jan", "Feb", "Mar"]  # Hard-coded example
    }

# -----------------------------------------------------------------------------
# 8) Final Assembly Node
# -----------------------------------------------------------------------------

def finalize_report_json(state: FinalReportState):
    """
    Merges all partial outputs (component configs, list contents) back into the layout_json.
    Returns the final JSON.
    """
    final_layout = state["layout_json"]
    return {"final_json": final_layout}

# -----------------------------------------------------------------------------
# BUILD THE GRAPH
# -----------------------------------------------------------------------------

class ReportGraphState(MessagesState, FinalReportState):
    """
    Combine the typed dict with MessagesState if you plan to hold conversation logs, etc.
    Or keep it simple if you do not need to store message history.
    """
    pass

builder = StateGraph(ReportGraphState)

# 1) Start => verify_instructions
builder.add_node("verify_instructions", verify_instructions)
builder.add_edge(START, "verify_instructions")

# 2) If instructions not clear => ask_clarification => after user clarifies => re-check
def instructions_decider(state: ReportGraphState):
    if state["instructions_clear"]:
        return "generate_layout"
    else:
        return "ask_clarification"

builder.add_conditional_edges("verify_instructions", instructions_decider,
                              ["generate_layout", "ask_clarification"])

builder.add_node("ask_clarification", ask_clarification)

# After clarifying, you might want the user to input a new prompt or update the same. 
# Then jump back to verify_instructions.
builder.add_edge("ask_clarification", "verify_instructions")

# 3) generate_layout
builder.add_node("generate_layout", generate_layout_json)

# 4) user_confirm_layout
builder.add_node("user_confirm_layout", user_confirm_layout)
builder.add_edge("generate_layout", "user_confirm_layout")

# If user says layout_ok => proceed, else => generate_layout again
def layout_confirmation_router(state: ReportGraphState):
    if "layout_ok" in state and state["layout_ok"]:
        return "identify_components"
    else:
        return "generate_layout"

builder.add_conditional_edges("user_confirm_layout", layout_confirmation_router,
                              ["identify_components", "generate_layout"])

# 5) identify_components => parallel "generate_component_content" calls
builder.add_node("identify_components", identify_components)

def parallel_components(state: ReportGraphState):
    comps = state.get("components_to_generate", [])
    if not comps:
        return "identify_and_unify_lists"  # If none, skip
    tasks = []
    for c in comps:
        tasks.append(
            Send(
                "generate_component_content",
                {
                    "component_id": c["id"],
                    "component_desc": c.get("desc", "")
                }
            )
        )
    return tasks

builder.add_node("generate_component_content", generate_component_content)
builder.add_conditional_edges("identify_components", parallel_components, ["identify_and_unify_lists"])

# We'll assume all parallel tasks come back into a single place to update the layout JSON
# You might store each returned "config" in the global layout. This example is simplified.

def collect_component_configs(state: ReportGraphState, results: List[dict]):
    """
    This is invoked automatically once all parallel 'generate_component_content' tasks finish.
    We then inject the returned config into the layout JSON in the correct place.
    """
    layout_json = state["layout_json"]

    # A naive approach: search by ID and fill "config"
    def assign_config(obj, comp_id, config_data):
        if isinstance(obj, dict):
            if "id" in obj and obj["id"] == comp_id:
                obj["config"] = config_data
            else:
                for v in obj.values():
                    assign_config(v, comp_id, config_data)
        elif isinstance(obj, list):
            for v in obj:
                assign_config(v, comp_id, config_data)

    # Iterate over parallel results
    for r in results:
        comp_id = r["component_id"]
        config_data = r["config"]
        assign_config(layout_json, comp_id, config_data)

    return {"layout_json": layout_json}

builder.add_collector("generate_component_content", collect_component_configs,
                      next_step="identify_and_unify_lists")

# 6) unify lists
builder.add_node("identify_and_unify_lists", identify_and_unify_lists)

builder.add_edge("identify_and_unify_lists", "identify_unique_lists")

# 7) create lists in parallel
builder.add_node("identify_unique_lists", identify_unique_lists)

def parallel_list_creation(state: ReportGraphState):
    lists_ = state.get("lists_to_create", [])
    if not lists_:
        return "finalize_report_json"
    tasks = []
    for l_ in lists_:
        tasks.append(
            Send(
                "create_list_contents",
                {
                    "listReference": l_.get("listReference"),
                }
            )
        )
    return tasks

builder.add_node("create_list_contents", create_list_contents)
builder.add_conditional_edges("identify_unique_lists", parallel_list_creation,
                              ["finalize_report_json"])

# Suppose we then unify references in the final JSON again.
def collect_list_contents(state: ReportGraphState, results: List[dict]):
    """
    Once all parallel 'create_list_contents' tasks finish, update the layout JSON
    with the actual dimension members or other data.
    """
    layout_json = state["layout_json"]

    # We have to find each list by "listReference" and store the members
    def walk_and_assign(obj, list_ref, members):
        if isinstance(obj, dict):
            if "lists" in obj and isinstance(obj["lists"], list):
                for l_ in obj["lists"]:
                    if l_["listReference"] == list_ref:
                        l_["list"] = members
            for v in obj.values():
                walk_and_assign(v, list_ref, members)
        elif isinstance(obj, list):
            for v in obj:
                walk_and_assign(v, list_ref, members)

    for r in results:
        ref = r["listReference"]
        members = r["members"]
        walk_and_assign(layout_json, ref, members)

    return {"layout_json": layout_json}

builder.add_collector("create_list_contents", collect_list_contents,
                      next_step="finalize_report_json")

# 8) finalize
builder.add_node("finalize_report_json", finalize_report_json)
builder.add_edge("finalize_report_json", END)

# Finally, compile the graph
graph = builder.compile()


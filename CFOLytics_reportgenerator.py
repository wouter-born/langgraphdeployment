"""
CFOLytics_reportgenerator.py

Revised LangGraph workflow that:
1) Takes an initial user prompt.
2) Checks clarity via LLM.
3) If unclear, LLM asks a clarifying question → user answers → appended to conversation.
4) Re-checks instructions with updated conversation.
5) If clear, proceeds to generate a layout JSON using the entire conversation.
6) Pauses at layout confirmation (interrupt) to let the user confirm or reject.
7) Iterates over each component, passing the entire conversation plus the
   "AI Generation Description" for that component to the LLM to generate config JSON.
8) Unifies lists if needed, populates them, and finalizes the JSON.

Requires:
- langgraph
- langchain-core
- langchain-community
- langchain-openai (or your custom classes)
"""

import os
import json
import re
from typing import List
from typing_extensions import TypedDict

# LangChain / LangGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

# ---------------------------------------------------------------------------
# 1) LLM Setup
# ---------------------------------------------------------------------------
# Using your custom ChatGroq per your snippet
from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-specdec",
    api_key="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"
)

# ---------------------------------------------------------------------------
# 2) XML Loader
# ---------------------------------------------------------------------------
def load_xml_instructions(filename: str) -> str:
    """
    Load system instructions from an XML file in a "XML_instructions" folder.
    Adjust if you store them differently or embed them directly.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "XML_instructions", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------------------------
# 3) State Definition
# ---------------------------------------------------------------------------
class FinalReportState(TypedDict):
    """
    Holds data throughout the report-generation process.
    """
    instructions_clear: bool          # Whether instructions are considered clear
    layout_json: dict                 # Generated layout JSON
    final_json: dict                  # The final JSON after content generation

class ReportGraphState(MessagesState, FinalReportState):
    """
    Extends the base MessagesState (which holds a conversation list)
    with our custom keys for instructions clarity, layout JSON, etc.
    """
    pass

# ---------------------------------------------------------------------------
# 4) verify_instructions node
# ---------------------------------------------------------------------------
def verify_instructions(state: ReportGraphState):
    """
    Checks the conversation so far to see if instructions are clear or need clarification.
    Uses instructions from 'verify_instructions.xml'.
    Appends the LLM's 'check' output as an AIMessage to state["messages"].
    """
    system_instructions = load_xml_instructions("verify_instructions.xml")
    system_msg = SystemMessage(content=system_instructions)

    conversation = [system_msg] + state["messages"]
    result = llm.invoke(conversation)

    # Store the result in the conversation
    state["messages"].append(AIMessage(content=result.content, name="system-check"))

    text_lower = result.content.lower()
    if "not clear" in text_lower or "unclear" in text_lower:
        return {"instructions_clear": False}
    return {"instructions_clear": True}

# ---------------------------------------------------------------------------
# 5) ask_clarification node
# ---------------------------------------------------------------------------
def ask_clarification(state: ReportGraphState):
    """
    If instructions are unclear, ask a clarifying question (via LLM).
    Appends that question as an AIMessage.
    Next step is 'await_clarification_answer' (interrupt) so user can reply.
    """
    system_instructions = load_xml_instructions("clarification_prompt.xml")
    system_msg = SystemMessage(content=system_instructions)

    conversation = [system_msg] + state["messages"]
    result = llm.invoke(conversation)

    clarification_q = AIMessage(content=result.content, name="clarification_question")
    state["messages"].append(clarification_q)

    return {"clarification_question": result.content}

# ---------------------------------------------------------------------------
# 6) await_clarification_answer node
# ---------------------------------------------------------------------------
def await_clarification_answer(state: ReportGraphState):
    """
    Interrupt node. The user must set 'clarification_answer' in the UI before continuing.
    We then append that to the conversation as a HumanMessage.
    """
    # If user hasn't provided an answer, do nothing so the flow stops here
    if "clarification_answer" not in state or not state["clarification_answer"]:
        return {}
    # Otherwise, we have a user response
    user_answer = state["clarification_answer"]
    state["messages"].append(HumanMessage(content=user_answer, name="user-clarification"))
    return {}

# ---------------------------------------------------------------------------
# 7) generate_layout_json node
# ---------------------------------------------------------------------------
def generate_layout_json(state: ReportGraphState):
    """
    Uses the entire conversation to produce a JSON layout.
    Loads 'render_layout.xml' as system instructions.
    Attempts to parse the LLM's response for JSON via regex or direct parse.
    """
    system_instructions = load_xml_instructions("render_layout.xml")
    system_msg = SystemMessage(content=system_instructions)

    conversation = [system_msg] + state["messages"]
    result = llm.invoke(conversation)
    raw_output = result.content.strip()

    # Attempt to parse JSON from the output
    # We look for a { ... } block
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if match:
        try:
            layout_dict = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            layout_dict = {
                "error": "Could not parse LLM output as JSON",
                "raw_llm_output": raw_output,
                "exception": str(e)
            }
    else:
        layout_dict = {
            "error": "No valid JSON object found in LLM output",
            "raw_llm_output": raw_output
        }

    return {"layout_json": layout_dict}

# ---------------------------------------------------------------------------
# 8) user_confirm_layout node
# ---------------------------------------------------------------------------
def user_confirm_layout(state: ReportGraphState):
    """
    Pauses so the user can confirm the generated layout. 
    We'll read 'layout_confirm' from the state:
      - If user sets state["layout_confirm"] = "yes", proceed.
      - If user sets state["layout_confirm"] = "no", we go back to generate_layout_json.
    """
    # If user hasn't set layout_confirm in the UI, we pause.
    if "layout_confirm" not in state or not state["layout_confirm"]:
        return {}
    else:
        # Return "layout_ok" based on user input
        layout_confirm = state["layout_confirm"].lower()
        if layout_confirm == "yes":
            return {"layout_ok": True}
        else:
            return {"layout_ok": False}

# ---------------------------------------------------------------------------
# 9) generate_components_config node
# ---------------------------------------------------------------------------
def generate_components_config(state: ReportGraphState):
    """
    Iterates over all components in layout_json, calls LLM to produce config.
    We pass the entire conversation (for context) + the component's "AI Generation Description"
    if present. That helps avoid the "you haven't provided instructions" fallback.
    """
    layout_json = state.get("layout_json", {})
    if not layout_json or "error" in layout_json:
        return {}

    # Gather all components
    components = []
    def find_components(obj):
        if isinstance(obj, dict):
            if "components" in obj and isinstance(obj["components"], list):
                for c in obj["components"]:
                    components.append(c)
            for v in obj.values():
                find_components(v)
        elif isinstance(obj, list):
            for v in obj:
                find_components(v)
    find_components(layout_json)

    # We'll load instructions once outside the loop
    system_instructions = load_xml_instructions("component_content_gen.xml")
    system_msg = SystemMessage(content=system_instructions)

    def update_config(obj, comp_id, new_config):
        if isinstance(obj, dict):
            if obj.get("id") == comp_id:
                obj["config"] = new_config
            else:
                for v in obj.values():
                    update_config(v, comp_id, new_config)
        elif isinstance(obj, list):
            for v in obj:
                update_config(v, comp_id, new_config)

    # For each component, pass the entire conversation plus the AI Generation Description
    for comp in components:
        desc = comp.get("AI Generation Description", "")
        # If no desc, we can fallback to "No generation instructions"
        if not desc:
            desc = "No specific AI Generation Description was provided."

        # We'll pass the entire conversation + system instructions + user message with the desc
        conversation = [system_msg] + state["messages"] + [HumanMessage(content=desc, name="comp-desc")]

        result = llm.invoke(conversation)
        raw_out = result.content.strip()

        # Attempt to parse JSON
        try:
            config_dict = json.loads(raw_out)
        except json.JSONDecodeError:
            # Maybe there's a code block or partial JSON
            match = re.search(r"\{.*\}", raw_out, re.DOTALL)
            if match:
                try:
                    config_dict = json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    config_dict = {
                        "error": "Invalid JSON from LLM",
                        "raw": raw_out,
                        "exception": str(e)
                    }
            else:
                config_dict = {
                    "error": "Invalid JSON from LLM, no braces found",
                    "raw": raw_out
                }

        # Insert config into layout
        update_config(layout_json, comp.get("id"), config_dict)

    return {"layout_json": layout_json}

# ---------------------------------------------------------------------------
# 10) identify_and_unify_lists node
# ---------------------------------------------------------------------------
def identify_and_unify_lists(state: ReportGraphState):
    """
    Placeholder for optional list unification if multiple components have the same lists
    with different references. You can do LLM-based logic or an algorithmic approach.
    """
    layout_json = state.get("layout_json", {})
    # No real logic here, just pass it through
    return {"layout_json": layout_json}

# ---------------------------------------------------------------------------
# 11) create_lists_contents node
# ---------------------------------------------------------------------------
def create_lists_contents(state: ReportGraphState):
    """
    Looks for any "lists" in the layout JSON and populates them with dimension data.
    We'll just hard-code them for demonstration.
    """
    layout_json = state.get("layout_json", {})
    if not layout_json:
        return {}

    # Gather lists
    lists_found = []
    def walk_for_lists(obj):
        if isinstance(obj, dict):
            if "lists" in obj and isinstance(obj["lists"], list):
                for item in obj["lists"]:
                    lists_found.append(item)
            for v in obj.values():
                walk_for_lists(v)
        elif isinstance(obj, list):
            for v in obj:
                walk_for_lists(v)

    walk_for_lists(layout_json)

    # Hard-code members for demonstration
    for item in lists_found:
        item["list"] = ["Jan", "Feb", "Mar"]
        if "AI Generation Description" not in item:
            item["AI Generation Description"] = "Populated with 3 months as example."

    return {"layout_json": layout_json}

# ---------------------------------------------------------------------------
# 12) finalize_report_json node
# ---------------------------------------------------------------------------
def finalize_report_json(state: ReportGraphState):
    """
    Copy layout_json into final_json, concluding the process.
    """
    layout_json = state.get("layout_json", {})
    return {"final_json": layout_json}

# ---------------------------------------------------------------------------
# 13) Build the Graph
# ---------------------------------------------------------------------------
builder = StateGraph(ReportGraphState)

# START -> verify_instructions
builder.add_node("verify_instructions", verify_instructions)
builder.add_edge(START, "verify_instructions")

# Decide if instructions are clear => layout or not => clarification
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

# ask_clarification -> await_clarification_answer -> back to verify_instructions
builder.add_node("ask_clarification", ask_clarification)
builder.add_node("await_clarification_answer", await_clarification_answer)
builder.add_edge("ask_clarification", "await_clarification_answer")
builder.add_edge("await_clarification_answer", "verify_instructions")

# generate_layout_json -> user_confirm_layout
builder.add_node("generate_layout_json", generate_layout_json)
builder.add_edge("generate_layout_json", "user_confirm_layout")

# user_confirm_layout -> if yes => generate_components_config else => generate_layout_json
builder.add_node("user_confirm_layout", user_confirm_layout)
def layout_confirmation_router(state: ReportGraphState):
    layout_ok = state.get("layout_ok", False)
    if layout_ok:
        return "generate_components_config"
    else:
        return "generate_layout_json"

builder.add_conditional_edges(
    "user_confirm_layout",
    layout_confirmation_router,
    ["generate_components_config", "generate_layout_json"]
)

# generate_components_config -> identify_and_unify_lists
builder.add_node("generate_components_config", generate_components_config)
builder.add_edge("generate_components_config", "identify_and_unify_lists")

# identify_and_unify_lists -> create_lists_contents
builder.add_node("identify_and_unify_lists", identify_and_unify_lists)
builder.add_edge("identify_and_unify_lists", "create_lists_contents")

# create_lists_contents -> finalize_report_json -> END
builder.add_node("create_lists_contents", create_lists_contents)
builder.add_edge("create_lists_contents", "finalize_report_json")

builder.add_node("finalize_report_json", finalize_report_json)
builder.add_edge("finalize_report_json", END)

# We want to interrupt so the user can:
# 1) Provide a clarification answer at 'await_clarification_answer'
# 2) Confirm or deny the layout at 'user_confirm_layout'
graph = builder.compile(
    interrupt_before=["await_clarification_answer", "user_confirm_layout"]
)

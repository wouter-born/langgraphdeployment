"""
CFOLytics_reportgenerator_no_manual_fields.py

LangGraph workflow that:
1) Takes an initial user prompt in the conversation (state["messages"]).
2) Checks clarity using LLM instructions from 'verify_instructions.xml'.
3) If unclear, LLM asks clarifying question -> user responds in the conversation -> we parse user’s new message -> re-check clarity.
4) If clear, LLM generates a layout (render_layout.xml).
5) LLM then asks user “Is this layout okay?” -> user answers in conversation -> we parse yes/no from the conversation -> if no, regenerate, if yes, proceed.
6) Generate components, unify lists, finalize JSON.

No need for manually adding `clarification_answer` or `layout_confirm` to the state. The conversation itself is the source of truth.

Requires:
- langgraph
- langchain-core
- langchain-community
- langchain-openai
- Your custom ChatGroq, or whichever LLM wrapper you use
"""

import os
import json
import re
from typing import List
from typing_extensions import TypedDict

# LangChain / LangGraph
from langchain_core.messages import (
    AIMessage, 
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

# Example: your custom ChatGroq usage
from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-specdec",
    api_key="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"
)

def load_xml_instructions(filename: str) -> str:
    """
    Load system instructions from 'XML_instructions/filename' if you keep them externally.
    Otherwise, just inline your prompts as strings.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "XML_instructions", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

class FinalReportState(TypedDict):
    """
    Our final JSON structures and clarity status.
    """
    instructions_clear: bool
    layout_json: dict
    final_json: dict

class ReportGraphState(MessagesState, FinalReportState):
    """
    Merges the base conversation messages plus our custom fields.
    'messages' is a list of SystemMessage, HumanMessage, or AIMessage.
    """
    pass

# -----------------------------------------------------------------------------
# 1) verify_instructions
# -----------------------------------------------------------------------------

def verify_instructions(state: ReportGraphState):
    """
    Node checks if the conversation so far implies the instructions are clear or not.
    We load instructions from 'verify_instructions.xml'.
    - LLM appends a final line "clear" or "not clear" or "unclear" which we parse.
    - We store the LLM output in the conversation.
    """
    system_instructions = load_xml_instructions("verify_instructions.xml")
    system_msg = SystemMessage(content=system_instructions)

    # We pass the entire conversation plus the system instructions.
    conversation = [system_msg] + state["messages"]
    result = llm.invoke(conversation)

    # Store the LLM's analysis as an AIMessage
    state["messages"].append(AIMessage(content=result.content, name="clarity-check"))
    
    text_lower = result.content.lower()
    if "not clear" in text_lower or "unclear" in text_lower:
        return {"instructions_clear": False}
    return {"instructions_clear": True}


# -----------------------------------------------------------------------------
# 2) ask_clarification
# -----------------------------------------------------------------------------
def ask_clarification(state: ReportGraphState):
    """
    LLM asks the user clarifying questions. We store that question in the conversation.
    """
    system_instructions = load_xml_instructions("clarification_prompt.xml")
    system_msg = SystemMessage(content=system_instructions)

    conversation = [system_msg] + state["messages"]
    result = llm.invoke(conversation)

    # Append the AI’s clarifying question
    question_msg = AIMessage(content=result.content, name="clarification_question")
    state["messages"].append(question_msg)

    return {}  # No direct state changes, just updated conversation


# -----------------------------------------------------------------------------
# 3) get_user_clarification
# -----------------------------------------------------------------------------
def get_user_clarification(state: ReportGraphState):
    idx_question = None
    for i, msg in reversed(list(enumerate(state["messages"]))):
        if isinstance(msg, AIMessage) and msg.name == "clarification_question":
            idx_question = i
            break
    if idx_question is None:
        return {}

    for j in range(idx_question+1, len(state["messages"])):
        msg = state["messages"][j]
        if isinstance(msg, HumanMessage):
            # User responded, proceed to next node
            return {}

    # Remain in this node until the user responds
    return None  # Signal to remain in the current node


# -----------------------------------------------------------------------------
# 4) generate_layout_json
# -----------------------------------------------------------------------------
def generate_layout_json(state: ReportGraphState):
    """
    We assume instructions are now clear. 
    Use 'render_layout.xml' as system instructions, plus the entire conversation.
    Then parse out a JSON layout from the LLM output.
    """
    system_instructions = load_xml_instructions("render_layout.xml")
    system_msg = SystemMessage(content=system_instructions)

    conversation = [system_msg] + state["messages"]
    result = llm.invoke(conversation)
    raw_output = result.content

    # Store the AI’s layout as a message for transparency
    state["messages"].append(AIMessage(content=raw_output, name="layout-draft"))

    # Attempt to parse JSON
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


# -----------------------------------------------------------------------------
# 5) ask_if_layout_ok
# -----------------------------------------------------------------------------
def ask_if_layout_ok(state: ReportGraphState):
    """
    The system/AI asks the user: "Is this layout okay? If not, type 'No' or mention changes."
    We'll store that in the conversation as an AIMessage. Then the next node checks user’s answer.
    """
    # We can simply produce a short system/AI message that says “Here is the layout. Are you OK with this?”
    # Or we can load from an XML if you want more instructions.

    prompt = (
        "Below is the current layout JSON I've generated.\n\n"
        f"{json.dumps(state['layout_json'], indent=2)}\n\n"
        "Are you satisfied with this layout? Please answer 'Yes' if it's good, or 'No' plus any changes you'd like."
    )
    ai_msg = AIMessage(content=prompt, name="layout-confirmation-question")
    state["messages"].append(ai_msg)
    return {}


# -----------------------------------------------------------------------------
# 6) check_layout_confirmation
# -----------------------------------------------------------------------------
def check_layout_confirmation(state: ReportGraphState):
    """
    We look for the user's next message after the "layout-confirmation-question".
    If we see "Yes," we proceed. If "No," we revert to re-generate_layout_json.
    If no user response is found, we remain here waiting.
    """
    # find last AI layout-confirmation-question
    idx_question = None
    for i, msg in reversed(list(enumerate(state["messages"]))):
        if isinstance(msg, AIMessage) and msg.name == "layout-confirmation-question":
            idx_question = i
            break
    if idx_question is None:
        # No question was asked? Just proceed
        return {"layout_ok": True}
    
    # see if there's a user message after idx_question
    user_answer_msg = None
    for j in range(idx_question+1, len(state["messages"])):
        msg = state["messages"][j]
        if isinstance(msg, HumanMessage):
            user_answer_msg = msg
            break

    if not user_answer_msg:
        # No answer yet, remain in this node
        return {}

    # parse user message
    content_lower = user_answer_msg.content.lower().strip()
    if content_lower.startswith("yes"):
        # layout OK
        return {"layout_ok": True}
    else:
        # user said "No" or "No plus changes"
        return {"layout_ok": False}


# -----------------------------------------------------------------------------
# 7) generate_components_config
# -----------------------------------------------------------------------------
def generate_components_config(state: ReportGraphState):
    """
    Iterates over each component in layout_json, uses 'component_content_gen.xml'
    plus the conversation to produce config. 
    Stores the final config in layout_json.
    """
    layout_json = state.get("layout_json", {})
    if not layout_json or "error" in layout_json:
        return {}

    # find components
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
    walk(layout_json)

    comp_instructions = load_xml_instructions("component_content_gen.xml")
    system_msg = SystemMessage(content=comp_instructions)

    def set_config(obj, comp_id, new_config):
        if isinstance(obj, dict):
            if obj.get("id") == comp_id:
                obj["config"] = new_config
            else:
                for v in obj.values():
                    set_config(v, comp_id, new_config)
        elif isinstance(obj, list):
            for v in obj:
                set_config(v, comp_id, new_config)

    # For each component
    for comp in components:
        desc = comp.get("AI Generation Description", "")
        if not desc:
            desc = "No AI Generation Description was provided."
        # entire conversation + system instructions + user message with desc
        conversation = [system_msg] + state["messages"] + [
            HumanMessage(content=desc, name="component-desc")
        ]
        result = llm.invoke(conversation)
        raw_out = result.content.strip()

        # parse JSON
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
                "error": "No valid JSON object found",
                "raw": raw_out
            }
        set_config(layout_json, comp["id"], config_dict)

    return {"layout_json": layout_json}


# -----------------------------------------------------------------------------
# 8) identify_and_unify_lists
# -----------------------------------------------------------------------------
def identify_and_unify_lists(state: ReportGraphState):
    """
    Placeholder for list unification. We'll just pass layout through.
    """
    layout_json = state.get("layout_json", {})
    return {"layout_json": layout_json}


# -----------------------------------------------------------------------------
# 9) create_lists_contents
# -----------------------------------------------------------------------------
def create_lists_contents(state: ReportGraphState):
    """
    Finds 'lists' keys and populates them with dimension members. Hard-coded example.
    """
    layout_json = state.get("layout_json", {})
    if not layout_json:
        return {}

    lists_found = []
    def walk(obj):
        if isinstance(obj, dict):
            if "lists" in obj and isinstance(obj["lists"], list):
                for l_ in obj["lists"]:
                    lists_found.append(l_)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(layout_json)

    for item in lists_found:
        item["list"] = ["Jan", "Feb", "Mar"]
        if "AI Generation Description" not in item:
            item["AI Generation Description"] = "Populated with months for demonstration."

    return {"layout_json": layout_json}


# -----------------------------------------------------------------------------
# 10) finalize_report_json
# -----------------------------------------------------------------------------
def finalize_report_json(state: ReportGraphState):
    """
    Copy layout_json => final_json
    """
    layout_json = state.get("layout_json", {})
    return {"final_json": layout_json}


# -----------------------------------------------------------------------------
# Build the Graph
# -----------------------------------------------------------------------------
builder = StateGraph(ReportGraphState)

# 1. START => verify_instructions
builder.add_node("verify_instructions", verify_instructions)
builder.add_edge(START, "verify_instructions")

# 2. Decide: If instructions_clear => generate_layout_json, else => ask_clarification
def instructions_decider(state: ReportGraphState):
    return "generate_layout_json" if state["instructions_clear"] else "ask_clarification"

builder.add_conditional_edges(
    "verify_instructions",
    instructions_decider,
    ["generate_layout_json","ask_clarification"]
)

# 3. ask_clarification => get_user_clarification => verify_instructions
builder.add_node("ask_clarification", ask_clarification)
builder.add_node("get_user_clarification", get_user_clarification)

builder.add_edge("ask_clarification", "get_user_clarification")
builder.add_edge("get_user_clarification", "verify_instructions")

# 4. generate_layout_json => ask_if_layout_ok => check_layout_confirmation
builder.add_node("generate_layout_json", generate_layout_json)
builder.add_node("ask_if_layout_ok", ask_if_layout_ok)
builder.add_node("check_layout_confirmation", check_layout_confirmation)

builder.add_edge("generate_layout_json", "ask_if_layout_ok")
builder.add_edge("ask_if_layout_ok", "check_layout_confirmation")

def layout_ok_decider(state: ReportGraphState):
    if "layout_ok" not in state:
        return None  # Stay in the current node
    return "generate_components_config" if state["layout_ok"] else "generate_layout_json"


builder.add_conditional_edges(
    "check_layout_confirmation",
    layout_ok_decider,
    ["generate_components_config", "generate_layout_json"]
)

# 5. generate_components_config => identify_and_unify_lists => create_lists_contents => finalize_report_json => END
builder.add_node("generate_components_config", generate_components_config)
builder.add_edge("generate_components_config", "identify_and_unify_lists")

builder.add_node("identify_and_unify_lists", identify_and_unify_lists)
builder.add_edge("identify_and_unify_lists", "create_lists_contents")

builder.add_node("create_lists_contents", create_lists_contents)
builder.add_edge("create_lists_contents", "finalize_report_json")

builder.add_node("finalize_report_json", finalize_report_json)
builder.add_edge("finalize_report_json", END)

# Finally, compile without using interrupt_before, because we rely on conversation-based logic
graph = builder.compile(interrupt_before=['get_user_clarification','check_layout_confirmation'])

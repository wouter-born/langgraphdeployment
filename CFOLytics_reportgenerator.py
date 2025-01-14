"""
CFOLytics_reportgenerator_structured.py

LangGraph workflow that:
1) Takes an initial user prompt + clarifications.
2) Uses a JSON schema with 'with_structured_output' to strictly format
   the layout JSON in the 'generate_layout_json' step.

Requires:
- langgraph
- langchain-core
- langchain-community
- langchain-openai
"""

import os
import json
from typing_extensions import TypedDict
from typing import Optional, List

# LangChain / LangGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

# ----------------------------------------------------------------------------
# 1) Environment Variables (example placeholders)
#    Ensure these are set in your environment or .env file as you see fit:
# ----------------------------------------------------------------------------
# os.environ["OPENAI_API_KEY"] = "sk-proj-..."
# os.environ["TAVILY_API_KEY"] = "tvly-..."
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_..."
# os.environ["LANGCHAIN_PROJECT"] = "TestAgent"

# ----------------------------------------------------------------------------
# 2) LLM Setup
# ----------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ----------------------------------------------------------------------------
# 3) JSON Schema for Structured Output
# ----------------------------------------------------------------------------
# This dictionary mirrors the JSON schema you provided:
REPORT_SCHEMA = {
    "name": "financial_report_config",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "reportTitle": {"type": "string"},
            "numberFormat": {
                "type": "object",
                "properties": {
                    "currency": {"type": "string"},
                    "scale": {"type": "string"},
                    "decimals": {"type": "integer"}
                },
                "required": ["currency", "scale", "decimals"],
                "additionalProperties": False
            },
            "layout": {
                "type": "object",
                "properties": {
                    "gridColumns": {
                        "type": "object",
                        "properties": {
                            "sm": {"type": "string"},
                            "md": {"type": "string"},
                            "lg": {"type": "string"}
                        },
                        "required": ["sm", "md", "lg"],
                        "additionalProperties": False
                    },
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "columns": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "colSpan": {
                                                "type": "object",
                                                "properties": {
                                                    "sm": {"type": "string"},
                                                    "md": {"type": "string"},
                                                    "lg": {"type": "string"}
                                                },
                                                "required": ["sm", "md", "lg"],
                                                "additionalProperties": False
                                            },
                                            "components": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "id": {"type": "string"},
                                                        "type": {"type": "string"},
                                                        "title": {"type": "string"},
                                                        "noborder": {"type": "boolean"},
                                                        "height": {"type": "integer"},
                                                        "numberFormat": {
                                                            "type": "object",
                                                            "properties": {
                                                                "scale": {"type": "string"},
                                                                "decimals": {"type": "integer"}
                                                            },
                                                            "required": ["scale", "decimals"],
                                                            "additionalProperties": False
                                                        }
                                                    },
                                                    "required": ["id", "type"],
                                                    "additionalProperties": False
                                                }
                                            }
                                        },
                                        "required": ["colSpan", "components"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["columns"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["gridColumns", "rows"],
                "additionalProperties": False
            }
        },
        "required": ["reportTitle", "numberFormat", "layout"],
        "additionalProperties": False
    }
}

# ----------------------------------------------------------------------------
# 4) XML Loader (optional)
# ----------------------------------------------------------------------------
def load_xml_instructions(filename: str) -> str:
    """
    Load system instructions from an XML file, if you keep them separate.
    Adjust to your actual file structure.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "XML_instructions", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ----------------------------------------------------------------------------
# 5) State Definition
# ----------------------------------------------------------------------------
class FinalReportState(TypedDict):
    """
    Holds data throughout our report-generation process.
    """
    instructions_clear: bool
    layout_json: dict
    final_json: dict

class ReportGraphState(MessagesState, FinalReportState):
    """
    Merge 'MessagesState' (which holds a conversation) with custom fields.
    'messages' is a list of AIMessage/HumanMessage/SystemMessage.
    """
    pass

# ----------------------------------------------------------------------------
# 6) verify_instructions Node
# ----------------------------------------------------------------------------
def verify_instructions(state: ReportGraphState):
    """
    Checks if the conversation's instructions are clear.
    Uses 'verify_instructions.xml' for the system instructions.
    """
    system_instructions = load_xml_instructions("verify_instructions.xml")
    system_msg = SystemMessage(content=system_instructions)

    # We pass the entire conversation to the LLM
    conversation = [system_msg] + state["messages"]

    result = llm.invoke(conversation)
    text = result.content.lower()

    # Optionally store the response in the conversation for debugging
    state["messages"].append(AIMessage(content=result.content, name="system-check"))

    if "not clear" in text or "unclear" in text:
        return {"instructions_clear": False}
    return {"instructions_clear": True}

# ----------------------------------------------------------------------------
# 7) ask_clarification Node
# ----------------------------------------------------------------------------
def ask_clarification(state: ReportGraphState):
    """
    If instructions are unclear, we generate a clarifying question.
    The question is appended as an AIMessage.
    Next step: 'await_clarification_answer' so user can respond.
    """
    system_instructions = load_xml_instructions("clarification_prompt.xml")
    system_msg = SystemMessage(content=system_instructions)
    conversation = [system_msg] + state["messages"]

    result = llm.invoke(conversation)
    clarification_question = AIMessage(content=result.content, name="clarification_question")
    state["messages"].append(clarification_question)

    return {"clarification_question": result.content}

# ----------------------------------------------------------------------------
# 8) await_clarification_answer Node
# ----------------------------------------------------------------------------
def await_clarification_answer(state: ReportGraphState):
    """
    This node interrupts, waiting for the user to supply state["clarification_answer"].
    That answer is appended as a HumanMessage, then we return to verify_instructions.
    """
    # If the user hasn't provided a clarification answer, we do nothing -> wait.
    if "clarification_answer" not in state or not state["clarification_answer"]:
        return {}
    else:
        # The user has typed their clarification. Append it to messages.
        user_answer = state["clarification_answer"]
        state["messages"].append(HumanMessage(content=user_answer, name="user-clarification"))
        return {}

# ----------------------------------------------------------------------------
# 9) generate_layout_json Node (With Structured Output)
# ----------------------------------------------------------------------------
def generate_layout_json(state: ReportGraphState):
    """
    Generates the layout JSON. We use 'with_structured_output(json_schema=REPORT_SCHEMA)'
    so the LLM must adhere to the specified schema. That ensures we get a valid object.
    """
    system_instructions = load_xml_instructions("render_layout.xml")
    system_msg = SystemMessage(content=system_instructions)
    conversation = [system_msg] + state["messages"]

    # The 'with_structured_output' ensures the LLM's output is validated.
    structured_llm = llm.with_structured_output(json_schema=REPORT_SCHEMA)

    # Invoke
    structured_output = structured_llm.invoke(conversation)
    # 'structured_output' should be a Python dictionary conforming to the schema.
    # Or it could be a typed object in some cases. We'll assume it's a dict here.

    # We store it in layout_json
    return {"layout_json": structured_output}

# ----------------------------------------------------------------------------
# 10) user_confirm_layout Node
# ----------------------------------------------------------------------------
def user_confirm_layout(state: ReportGraphState):
    """
    Let the user confirm or reject the layout. For example, we always proceed here.
    """
    return {"layout_ok": True}

def layout_confirmation_router(state: ReportGraphState):
    return "generate_components_config" if state.get("layout_ok") else "generate_layout_json"

# ----------------------------------------------------------------------------
# 11) generate_components_config Node
# ----------------------------------------------------------------------------
def generate_components_config(state: ReportGraphState):
    """
    Iterates over all 'components' in the layout, calling an LLM prompt (if desired)
    to fill additional config. We'll skip the structured approach here, though you can
    do the same if you have a sub-schema for components.
    """
    layout_json = state.get("layout_json", {})
    if not isinstance(layout_json, dict):
        return {}

    # Find components
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

    # If you want to refine each component:
    component_instructions = load_xml_instructions("component_content_gen.xml")
    for comp in components:
        desc = comp.get("AI Generation Description", "")
        if desc:
            system_msg = SystemMessage(content=component_instructions)
            user_msg = HumanMessage(content=desc)
            result = llm.invoke([system_msg, user_msg])
            try:
                config_dict = json.loads(result.content)
                # Insert config into 'comp' or do what you want here
                comp["autoGeneratedConfig"] = config_dict
            except json.JSONDecodeError:
                comp["autoGeneratedConfig"] = {"error": "Invalid JSON from LLM"}

    return {"layout_json": layout_json}

# ----------------------------------------------------------------------------
# 12) identify_and_unify_lists Node
# ----------------------------------------------------------------------------
def identify_and_unify_lists(state: ReportGraphState):
    """
    Placeholder for unifying repeated lists. You can do an LLM approach or
    an algorithmic approach. We'll do nothing.
    """
    layout_json = state["layout_json"]
    return {"layout_json": layout_json}

# ----------------------------------------------------------------------------
# 13) create_lists_contents Node
# ----------------------------------------------------------------------------
def create_lists_contents(state: ReportGraphState):
    """
    Suppose we fill dimension lists from a backend. We'll just hardcode some data.
    """
    layout_json = state["layout_json"]
    # If needed, traverse and fill "list": [...]
    return {"layout_json": layout_json}

# ----------------------------------------------------------------------------
# 14) finalize_report_json Node
# ----------------------------------------------------------------------------
def finalize_report_json(state: ReportGraphState):
    layout_json = state["layout_json"]
    return {"final_json": layout_json}

# ----------------------------------------------------------------------------
# 15) Build the Graph
# ----------------------------------------------------------------------------
builder = StateGraph(ReportGraphState)

# Step 1: START -> verify_instructions
builder.add_node("verify_instructions", verify_instructions)
builder.add_edge(START, "verify_instructions")

# Step 2: instructions_decider
def instructions_decider(state: ReportGraphState):
    return "generate_layout_json" if state["instructions_clear"] else "ask_clarification"

builder.add_conditional_edges(
    "verify_instructions",
    instructions_decider,
    ["generate_layout_json", "ask_clarification"]
)

# Step 3: ask_clarification -> await_clarification_answer -> verify_instructions
builder.add_node("ask_clarification", ask_clarification)
builder.add_node("await_clarification_answer", await_clarification_answer)

builder.add_edge("ask_clarification", "await_clarification_answer")
builder.add_edge("await_clarification_answer", "verify_instructions")

# Step 4: generate_layout_json -> user_confirm_layout
builder.add_node("generate_layout_json", generate_layout_json)
builder.add_edge("generate_layout_json", "user_confirm_layout")

# Step 5: user_confirm_layout -> generate_components_config or back to generate_layout_json
builder.add_node("user_confirm_layout", user_confirm_layout)

builder.add_conditional_edges(
    "user_confirm_layout",
    layout_confirmation_router,
    ["generate_components_config", "generate_layout_json"]
)

# Step 6: generate_components_config -> identify_and_unify_lists
builder.add_node("generate_components_config", generate_components_config)
builder.add_edge("generate_components_config", "identify_and_unify_lists")

# Step 7: identify_and_unify_lists -> create_lists_contents
builder.add_node("identify_and_unify_lists", identify_and_unify_lists)
builder.add_edge("identify_and_unify_lists", "create_lists_contents")

# Step 8: create_lists_contents -> finalize_report_json -> END
builder.add_node("create_lists_contents", create_lists_contents)
builder.add_edge("create_lists_contents", "finalize_report_json")

builder.add_node("finalize_report_json", finalize_report_json)
builder.add_edge("finalize_report_json", END)

# Compile the graph
graph = builder.compile(
    # Interrupt so the user can type in a "clarification_answer" at 'await_clarification_answer'
    interrupt_before=["await_clarification_answer"]
)

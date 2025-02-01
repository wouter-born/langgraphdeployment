import os
from typing import Literal
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from Classes.state_classes import ModifyReportState
from Nodes.clarify_instructions import *
from Nodes.generate_json_patches import *
from Nodes.generate_layout import *
from Nodes.generate_component import *

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("CUSTOM_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("CUSTOM_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("CUSTOM_PROJECT", "CFOLytics_reporteditor")

def should_continue(state) -> Literal["generate_json_patches", "human_clarify_instructions"]:
    print(state['instruction_correct'])
    if state.get('instruction_correct') is True:
        return "generate_json_patches"
    # Default to human clarification to prevent invalid transitions
    return "human_clarify_instructions"  

##########################
# MAIN GRAPH
##########################
graph = StateGraph(ModifyReportState)


# Nodes
graph.add_node("clarify_instructions", clarify_instructions)
graph.add_node("human_clarify_instructions", human_clarify_instructions)
graph.add_node("generate_json_patches", generate_json_patches)


# Edges
graph.add_edge(START, "clarify_instructions")
graph.add_conditional_edges("clarify_instructions",should_continue)
graph.add_edge("human_clarify_instructions","clarify_instructions")
graph.add_edge("generate_json_patches",END)

checkpointer = MemorySaver()

# Compile the main graph
app = graph.compile(checkpointer=checkpointer)

#from IPython.display import Image, display
#display(Image(app.get_graph().draw_mermaid_png()))
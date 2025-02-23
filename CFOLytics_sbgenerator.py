from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END


from Classes.state_classes import StoryboardState

from Nodes.sbgenerator.interpreter import *
from Nodes.sbgenerator.human_check_prompt import *
from Nodes.sbgenerator.human_clarify_prompt import *
from Nodes.sbgenerator.check_metadata import *
from Nodes.sbgenerator.human_clarify_details import *
from Nodes.sbgenerator.update_prompt import *
from Nodes.sbgenerator.generate_pages import *

# State
graph = StateGraph(StoryboardState)

# Routing functions
def ispromptaccurate(state: StoryboardState) -> Literal["check_metadata","human_clarify_prompt"]:
    pass
    if state['isaccurate']:
        return "check_metadata"
    return "human_clarify_prompt"

def ismetadatacorrect(state: StoryboardState) -> Literal["generate_pages","human_clarify_details"]:
    pass
    # Forcing Generate pages.
    return "generate_pages"
    # if state['isvalid']:
    #     return "generate_pages"
    # return "human_clarify_details"


graph.add_node("interpreter", interpreter)
graph.add_node("human_check_prompt", human_check_prompt)
graph.add_node("human_clarify_prompt", human_clarify_prompt)
graph.add_node("check_metadata", check_metadata)
graph.add_node("human_clarify_details", human_clarify_details)
graph.add_node("update_prompt", update_prompt)
graph.add_node("generate_pages", generate_pages)

graph.add_edge(START, "interpreter")
graph.add_edge("interpreter", "human_check_prompt")

graph.add_conditional_edges("human_check_prompt", ispromptaccurate)
graph.add_edge("human_clarify_prompt", "interpreter")

graph.add_conditional_edges("check_metadata", ismetadatacorrect)
graph.add_edge("human_clarify_details", "update_prompt")
graph.add_edge("update_prompt", "human_check_prompt")

graph.add_edge("generate_pages", END)


checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
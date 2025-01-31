from langgraph.graph import StateGraph, START, END

from Nodes.clarify_instructions import clarify_instructions
from Nodes.generate_layout import *
from Nodes.generate_component import *



def should_continue(state):
    if state['instruction_correct'] == True:
        return END
    else:
        return "clarify_instructions"

##########################
# MAIN GRAPH
##########################
graph = StateGraph(OverallState)

graph.add_node("clarify_instructions", clarify_instructions)
#graph.add_node("generate_json_patches", generate_json_patches)
#graph.add_node("apply_json_patches", generate_json_patches)

graph.add_edge(START, "clarify_instructions")
graph.add_conditional_edges("clarify_instructions",should_continue)


#graph.add_edge("generate_layout","generate_json_patches")
#graph.add_edge("generate_json_patches","apply_json_patches")
#graph.add_edge("apply_json_patches", END)

# Compile the main graph
app = graph.compile()
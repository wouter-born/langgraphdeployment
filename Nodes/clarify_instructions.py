##### CLARIFY_INSTRUCTIONS #######
import json
from typing import Optional, TypedDict
from pydantic import BaseModel, Field
from langgraph.types import interrupt, Command
from langchain_core.messages import (
    AIMessage, 
    HumanMessage,
    SystemMessage,
    BaseMessage
)

# Custom Libraries
from Classes.state_classes import ModifyReportState
from Nodes.load_xml_instructions import load_xml_instructions
from Classes.llm_classes import *
#from Classes.state_classes import ModifyReportState

# Pydantic Class
class Instructions(BaseModel):
    instruction_correct: bool
    clarification_questions: Optional[str]

def clarify_instructions(state: ModifyReportState):
    system_instructions = load_xml_instructions("verify_instructions.xml")
    system_msg = SystemMessage(content=system_instructions)
    
    inputMessage = {
        "instruction":state['instruction'], 
        "input_json":state['input_json']
    }
    
    user_msg = HumanMessage(content=json.dumps(inputMessage))
    
    structured_llm = modelSpec.with_structured_output(
        Instructions,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")

    #print(output)

    if output.get("parsed") is None:
        raise ValueError("Model didn't reply a valid output. Review input and answer format")
    parsed_output = output["parsed"]

    return parsed_output

##### HUMAN_CLARIFY_INSTRUCTIONS #########
def human_clarify_instructions(state: ModifyReportState):
    result = interrupt(
        {
            "task": "Las instrucciones no son correctas. Por favor, proporcione una versión corregida.",
            "current_instruction": state["instruction"],
            "clarification_needed": state["clarification_questions"]
        }
    )

    result["corrected_instruction"] = input(state["clarification_questions"])

    new_instruction = result.get("corrected_instruction")
    if not new_instruction:
        raise ValueError("No new instruction received.")

    state["instruction"] = new_instruction
    state["instruction_correct"] = False

    return state
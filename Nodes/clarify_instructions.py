
from typing import Optional, TypedDict
from pydantic import BaseModel

from langchain_core.messages import (
    AIMessage, 
    HumanMessage,
    SystemMessage,
    BaseMessage
)

# Custom Libraries
from Nodes.load_xml_instructions import load_xml_instructions
from Classes.llm_classes import *
#from Classes.state_classes import ModifyReportState

# Pydantic Class
class Instructions(BaseModel):
    instruction_correct: bool
    clarification_questions: Optional[str]

def clarify_instructions(state: Instructions):
    system_instructions = load_xml_instructions("verify_instructions.xml")
    system_msg = SystemMessage(content=system_instructions)
    user_msg = HumanMessage(content=[ state['instruction'], state['input_json'] ])
    
    structured_llm = modelSpec.with_structured_output(
        Instructions,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")
    parsed_output = output["parsed"].model_dump()


    #ModifyReportState["instruction_correct"] = True
    return {
        "instruction_correct":parsed_output['clarification_questions'], 
        "clarification_questions": parsed_output['clarification_questions']
    }
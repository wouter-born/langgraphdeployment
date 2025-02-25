import json
from typing import List
from pydantic import BaseModel

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from Classes.llm_classes import *
from Nodes.load_xml_instructions import load_xml_instructions
from Classes.state_classes import *

class MetadataCheck(BaseModel):
    isvalid: bool
    metadata_feedback: str

def check_metadata(state: StoryboardState):
    # Load instructions from XML and create the system message.
    instructions = load_xml_instructions("sbgenerator/check_metadata.xml")
    system_msg = SystemMessage(content=instructions)

    # Extract values from state.
    narrative = state['narrative']
    report_metadata = state["ReportMetadata"]
    pov = state['POV']
    # Assuming that lists is stored under the key 'Lists' (instead of duplicating POV).
    ExistingLists = state['ExistingLists']

    # Combine all necessary details into one JSON payload.
    combined_content = {
        "narrative": narrative,
        "ReportMetadata": report_metadata,
        "POV": pov,
        "ExistingLists": ExistingLists
    }
    
    # Create the user message with the combined payload.
    user_msg = HumanMessage(content=json.dumps(combined_content))
    
    # Optional state/context message; here we leave it empty.
    state_msg = AIMessage(content="")

    # Set up the structured LLM to output a JSON response matching the MetadataCheck model.
    structured_llm = modelVers.with_structured_output(
        MetadataCheck,
        method="json_mode",
        include_raw=True
    )
    
    conversation = [system_msg, state_msg, user_msg]
    output = structured_llm.invoke(
        conversation, 
        stream=False, 
        response_format="json", 
        config=thread_config
    )
    
    output = structured_llm.invoke(conversation, stream=False, response_format="json", config=thread_config)
    return { 'isvalid': output['parsed'].isvalid, 'metadata_feedback': output['parsed'].metadata_feedback }
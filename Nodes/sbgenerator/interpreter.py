from typing import List
from langchain_core.messages import ( AIMessage, HumanMessage, SystemMessage, BaseMessage )
from Classes.llm_classes import *
from Nodes.load_xml_instructions import load_xml_instructions
from Classes.state_classes import *


def Narrative(BaseModel):
    narrative: str

def interpreter(state: StoryboardState):

    instructions = load_xml_instructions("sbgenerator/interpreter.xml")
    system_msg = SystemMessage(content=instructions)

    user_msg = HumanMessage(content=state['input_prompt'])

    # report_metadata = state["ReportMetadata"]
    # pov = state['POV']
    # lists = state['POV']

    structured_llm = modelSpec.with_structured_output(
        Narrative,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json", config=thread_config)
    return { "narrative": output['parsed']['narrative'] , "isaccurate": False }
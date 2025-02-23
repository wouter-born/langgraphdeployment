from typing import List
from langchain_core.messages import ( AIMessage, HumanMessage, SystemMessage, BaseMessage )
from Classes.llm_classes import *
from Nodes.load_xml_instructions import load_xml_instructions
from Classes.state_classes import *

def PageClass(BaseMode):
    page: int
    instructions: str

def PagesClass(BaseModel):
    pages: List[dict]
    #pages: List[PageClass]

def generate_pages(state: StoryboardState):

    instructions = load_xml_instructions("sbgenerator/generate_pages.xml")
    system_msg = SystemMessage(content=instructions)
    user_msg = HumanMessage(content=state['narrative'])
    structured_llm = modelSpec.with_structured_output(
        PagesClass,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json", config=thread_config)

    print(output['parsed'])

    return { "pages": output['parsed']['pages'] }
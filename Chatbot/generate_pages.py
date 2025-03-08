import json
import logging
from typing import List
from langchain_core.messages import ( AIMessage, HumanMessage, SystemMessage, BaseMessage )
from langchain_groq import ChatGroq
from Nodes.load_xml_instructions import load_xml_instructions
from langgraph.graph import MessagesState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize the model
model = ChatGroq(
    model="deepseek-r1-distill-llama-70b-specdec",
    temperature=0,
    max_tokens=2048,
    timeout=5,
    stop=None,
    model_kwargs={  # Explicitly pass additional parameters here
        "response_format": {"type": "json_object"},        
    },
    disable_streaming=True
)

def PagesClass(BaseModel):
    pages: List[dict]

def generate_pages(state: MessagesState):

    instructions = load_xml_instructions("chatbot/generate_pages.xml")
    system_msg = SystemMessage(content=instructions)

    msg = state["messages"][-1]

    logger.info("--- Generate Pages --- ")
    logger.info(msg.content)

    user_msg = HumanMessage(content=msg.content)

    
    structured_llm = model.with_structured_output(
        PagesClass,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")

    # print(output['parsed']['pages'])

    return { "pages": output['parsed']['pages'] }
    return None
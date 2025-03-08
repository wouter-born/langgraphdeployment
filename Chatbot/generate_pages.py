from typing import List
from langchain_core.messages import ( AIMessage, HumanMessage, SystemMessage, BaseMessage )
from langchain_groq import ChatGroq
from Nodes.load_xml_instructions import load_xml_instructions
from langgraph.graph import MessagesState


# Initialize the model
model = ChatGroq(
    model="deepseek-r1-distill-llama-70b-specdec",
    temperature=0,
    max_tokens=512,
    timeout=5,
    stop=None,
    model_kwargs={  # Explicitly pass additional parameters here
        "model_provider":"groq",
        "stream": False,
        "response_format": {"type": "json_object"},
        
    }
)

def PagesClass(BaseModel):
    pages: List[dict]

def generate_pages(state: MessagesState):

    instructions = load_xml_instructions("Chatbot/generate_pages.xml")
    system_msg = SystemMessage(content=instructions)

    msg = state["messages"]

    print(msg)

    user_msg = HumanMessage(content=msg.content)
    structured_llm = model.with_structured_output(
        PagesClass,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")

    print(output['parsed']['pages'])

    return { "pages": output['parsed']['pages'] }
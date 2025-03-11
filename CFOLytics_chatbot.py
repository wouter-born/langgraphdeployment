import os
import logging
from typing import Dict, Any, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain.output_parsers import PydanticOutputParser
from langgraph.constants import Send

# NODES
from Nodes.load_xml_instructions import load_xml_instructions
# CHATBOT FUNCTIONS
from Chatbot.chatbot_model_optimizer import model_optimizer
from Chatbot.generate_pages import _generate_pages
from Chatbot.build_storyboard import build_storyboard
# STATES
from Classes.state_classes import ChatbotState
# OTHER GRAPHS
from CFOLytics_reportgenerator import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = ["GROQ_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

# Define Pydantic class for structured output
class ChatResponse(BaseModel):
    answer: str = Field(..., description="The chatbot's response to the user query")
    generate_pages: Optional[bool] = Field(..., description="the chatbot's reponse if pages need to be generated")

parser = PydanticOutputParser(pydantic_object=ChatResponse)

# Initialize the model
model = init_chat_model(
    model="deepseek-r1-distill-llama-70b-specdec",
    model_provider="groq",
    temperature=0,
    max_tokens=512,
    timeout=5,
    stop=None,
    model_kwargs={  # Explicitly pass additional parameters here
        "stream": False,
        "response_format": {"type": "json_object"},
    },
    disable_streaming=True
)
#logger.info("Model initialized successfully")

# Load the prompt XML and create a prompt template with a placeholder for conversation messages.
prompt = load_xml_instructions("chatbot/prompt_template.xml")
format_instructions = parser.get_format_instructions()

# Optimize the model definition
model_definition = model_optimizer("requestwithbsdata.json")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", f"{prompt} \n Use the following data: {model_definition} "),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create the state graph using the ChatbotState schema
workflow = StateGraph(state_schema=ChatbotState)

# Function to call the model and update the message history
def _call_model(state: ChatbotState):
    messages = state["messages"]
    try:
        
        # Build prompt and invoke the model
        prompt_text = prompt_template.invoke(state["messages"])
        response = model.invoke(prompt_text)
        
        response_json = response.content
        parsed_response = parser.parse(response_json)
        
        # Add AI message to the messages
        ai_answer = AIMessage(parsed_response.answer)
        messages = [ai_answer]

        return {"messages": messages, "generate_pages": parsed_response.generate_pages}
    except Exception as e:
        logger.error(f"Error calling model: {str(e)}", exc_info=True)
        #raise RuntimeError(f"Error calling model: {str(e)}")
        
        # Add AI message to the messages
        ai_answer = AIMessage(f"I couldn't reply to your last prompt. Can you try again? \n Error Message: {str(e)}")
        messages = [ai_answer]
        return {"messages": messages, "generate_pages": False}

# Conditional edge to decide whether to generate pages or say goodbye
def _should_continue(state: ChatbotState) -> Literal["genpages", "goodbye"]:
    if "generate_pages" in state and state["generate_pages"]:
        return "genpages"
    else:
        return "goodbye"

def _consolidate_json(state: ChatbotState):
    return {"final_storyboard_json": build_storyboard(state["JsonLayoutList"])}

def _goodbye(state: ChatbotState):
    return {"generate_pages": state["generate_pages"]}

def _continue_to_reports(state: ChatbotState):
    sends = []
    for page in state["pages"]:
        # 'instructions' => 'ReportQuery'
        payload = {
            "ReportQuery": page["instructions"],
            "POV": state["POV"],
            "ReportMetadata": state["ReportMetadata"],
            "ExistingLists": state["ExistingLists"]
        }
        sends.append(Send("generate_report_subgraph", payload))
    return sends

# Build the graph by adding nodes and edges
workflow.add_node("chatbot", _call_model)
workflow.add_node("genpages", _generate_pages)
workflow.add_node("generate_report_subgraph", app)
workflow.add_node("consolidate_json", _consolidate_json)
workflow.add_node("goodbye", _goodbye)

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", _should_continue)
workflow.add_conditional_edges("genpages", _continue_to_reports, ["generate_report_subgraph"])
workflow.add_edge("generate_report_subgraph", "consolidate_json")
workflow.add_edge("consolidate_json", "goodbye")
workflow.add_edge("goodbye", END)

# Create a MemorySaver to persist state history if needed.
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Chatbot loop that handles user messages, updates the custom state, and prints the response.
def chatbot_loop():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Create a new message and pass it as the starting state
        input_messages = [HumanMessage(user_input)]
        output = graph.invoke(
            {"messages": input_messages},
            {"configurable": {"thread_id": "web_interface"}}
        )
        # Get the latest AI message from the history
        raw_response = output["messages"][-1].content
        logger.info(output["messages"])
        print("ChatBot:", raw_response)

# Run the chatbot loop if this file is executed as the main script.
if __name__ == "__main__":
    chatbot_loop()
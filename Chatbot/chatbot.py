import json
import os
import logging
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain.output_parsers import PydanticOutputParser
import uvicorn
import xml.etree.ElementTree as ET
from pathlib import Path

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
        
    }
)
logger.info("Model initialized successfully")

from load_xml_instructions import load_xml_instructions


prompt = load_xml_instructions("prompt_template.xml")

# Generate format instructions for the model
format_instructions = parser.get_format_instructions()

from chatbot_model_optimizer import model_optimizer

model_definition = model_optimizer("/Users/map/Documents/langgraphdeployment/requestwithbsdata.json")

prompt_template = ChatPromptTemplate.from_messages(
                [( "system", f"{prompt} \n Use the following data: {model_definition} " ), MessagesPlaceholder(variable_name="messages"),]
            )
workflow = StateGraph(state_schema=MessagesState)


# def _generate_pages(state: MessagesState):
#     logger.info("Generating Pages..........")
#     return {"messages": state["messages"]}



def _call_model(state: MessagesState):
    try:
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)

        response_json = response.content
        parsed_response = parser.parse(response_json)

        logger.info(parsed_response.generate_pages)

        return {"messages": parsed_response.answer, "generate_pages": parsed_response.generate_pages}
    except Exception as e:
        logger.error(f"Error calling model: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error calling model: {str(e)}")


def _should_continue(state: MessagesState):
    if "generate_pages" in state and state["generate_pages"]:
        return "generate_pages"
    else:
        return END


from generate_pages import generate_pages as _generate_pages

workflow.add_node("model", _call_model)
workflow.add_node("generate_pages", _generate_pages)


workflow.add_edge(START, "model")       

#workflow.add_edge("model",END)

workflow.add_conditional_edges("model",_should_continue)

workflow.add_edge("generate_pages", END)





memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

    
# Chatbot function
def chatbot_loop():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        message = user_input

        input_messages = [HumanMessage(message)]
        output = app.invoke(
                {"messages": input_messages},
                {"configurable": {"thread_id": "web_interface"}}
        )
        raw_response = output["messages"][-1].content
        logger.info(f"Generated response: {raw_response[:1000]}...")

    
# Run chatbot loop
if __name__ == "__main__":
    chatbot_loop()


from langgraph.types import interrupt
from pydantic import BaseModel
from Classes.state_classes import *
from Nodes.load_xml_instructions import load_xml_instructions
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from Classes.llm_classes import modelSpec

class NarrativeAccuracy(BaseModel):
    isaccurate: bool
    narrative_modif: Optional[str]

def human_check_prompt(state: StoryboardState):
    # Load XML instructions
    instructions = load_xml_instructions("sbgenerator/human_check_prompt.xml")
    
    # Get narrative from state
    narrative = state["narrative"]
    
    # Enable the following snipet to have user interruption
    # With interrupt
    feedback_response = interrupt({
        "comment": "Please provide your detailed feedback on the narrative:",
        "narrative": narrative
    })
    user_feedback = feedback_response.get("feedback")
    

    # Enable the following snipet to pass thru a positive feedback.
    # Without interrupt
    # user_feedback = "Yes! I like it."
    
    if not user_feedback:
        raise ValueError("No feedback provided.")
    
    # Build the conversation with the instructions, state narrative, and user feedback
    system_msg = SystemMessage(content=instructions)
    state_msg = AIMessage(content=f"Narrative: {narrative}")
    feedback_msg = HumanMessage(content=f"Feedback: {user_feedback}")
    
    # Build the structured LLM that expects an output conforming to NarrativeAccuracy
    structured_llm = modelSpec.with_structured_output(
        NarrativeAccuracy,
        method="json_mode",
        include_raw=True
    )
    
    conversation = [system_msg, state_msg, feedback_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json", config=thread_config)

    parsed_output = output.get('parsed')
    if parsed_output is None:
       raise ValueError("LLM output could not be parsed. Check the response format and prompt instructions. This is node human_check_prompt. ")

    return { 'isaccurate': output['parsed'].isaccurate, 'narrative_modif': output['parsed'].narrative_modif }

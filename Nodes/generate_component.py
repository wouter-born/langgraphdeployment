from pydantic import BaseModel, Field
from langchain_core.messages import (
    AIMessage, 
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from Nodes.load_xml_instructions import load_xml_instructions
from Classes.llm_classes import *
from Classes.state_classes import SpecializedComponentState

class ComponentConfig(BaseModel):
    config : dict

def component_selector(state: SpecializedComponentState):
    ctype = state["component"].get("type", "").lower()
    selected_node = "generate_generic_component"

    if ctype == "chart":
        selected_node = "generate_chart_component"
    elif ctype == "reporttable":
        selected_node = "generate_table_component"
    elif ctype == "waterfall":
        selected_node = "generate_waterfall_component"
    elif ctype == "tile":
        selected_node = "generate_tile_component"
    
    # Update the state with the selected node
    state["selected_node"] = selected_node
    return state  # Return the full updated state
    
def selector_routing(state: SpecializedComponentState):
    """
    Return the node name chosen by 'component_selector'.
    The possible values are:
    - 'generate_chart_component'
    - 'generate_table_component'
    - 'generate_waterfall_component'
    - 'generate_tile_component'
    - 'generate_generic_component'
    """
    return state["selected_node"]



def _base_component_generation(component: dict, system_instructions_file: str) -> dict:
    """
    Common logic that calls an LLM to produce a configuration for *any* type of component.
    We just vary the system instruction file to tailor the generation.
    """
    ai_description = component.get("AI_Generation_Description", None)
    ai_description = ai_description.strip() if isinstance(ai_description, str) else ai_description
    if not ai_description:
        # If there's no AI Generation Description, just return an empty config
        return {}

    system_instructions = load_xml_instructions(system_instructions_file)
    system_msg = SystemMessage(content=system_instructions)
    user_msg = HumanMessage(content=ai_description)

    structured_llm = modelVers.with_structured_output(
        ComponentConfig,
        method="json_mode",
        include_raw=True
    )

    conversation = [system_msg, user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")

    parsed_output = output["parsed"]
    if parsed_output:
        return parsed_output.model_dump()  # returns a dict
    else:
        return {}


# CHART
def generate_chart_component(state: SpecializedComponentState):
    """
    Specialized node for chart components
    """
    component = state["component"]
    component_id = component.get("id", "NoId")

    # Use a chart-specific system instructions file
    specialized_config = _base_component_generation(
        component=component,
        system_instructions_file="component_chart_gen.xml"
    )

    generated_config = {
        "id": component_id,
        "generatedConfig": specialized_config
    }
    return {"JsonLayoutWithComponentConfig": [generated_config]}


# TABLE
def generate_table_component(state: SpecializedComponentState):
    component = state["component"]
    component_id = component.get("id", "NoId")

    specialized_config = _base_component_generation(
        component=component,
        system_instructions_file="component_table_gen.xml"
    )

    generated_config = {
        "id": component_id,
        "generatedConfig": specialized_config
    }
    return {"JsonLayoutWithComponentConfig": [generated_config]}


def generate_waterfall_component(state: SpecializedComponentState):
    component = state["component"]
    component_id = component.get("id", "NoId")

    specialized_config = _base_component_generation(
        component=component,
        system_instructions_file="component_waterfall_gen.xml"
    )

    generated_config = {
        "id": component_id,
        "generatedConfig": specialized_config
    }
    return {"JsonLayoutWithComponentConfig": [generated_config]}


def generate_tile_component(state: SpecializedComponentState):
    component = state["component"]
    component_id = component.get("id", "NoId")

    specialized_config = _base_component_generation(
        component=component,
        system_instructions_file="component_tile_gen.xml"
    )

    generated_config = {
        "id": component_id,
        "generatedConfig": specialized_config
    }
    return {"JsonLayoutWithComponentConfig": [generated_config]}


def generate_generic_component(state: SpecializedComponentState):
    """
    Fallback if component type is unrecognized. 
    Reuses the original 'component_content_gen.xml' instructions.
    """
    component = state["component"]
    component_id = component.get("id", "NoId")

    specialized_config = _base_component_generation(
        component=component,
        system_instructions_file="component_content_gen.xml"
    )

    generated_config = {
        "id": component_id,
        "generatedConfig": specialized_config
    }
    return {"JsonLayoutWithComponentConfig": [generated_config]}


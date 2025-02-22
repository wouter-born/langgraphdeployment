from pydantic import BaseModel, Field
from typing import Optional, TypedDict, List, Dict, Any, Union
from typing import Annotated
import operator
import json
from langchain_core.messages import (
    AIMessage, 
    HumanMessage,
    SystemMessage,
    BaseMessage
)

from Nodes.load_xml_instructions import load_xml_instructions

from Classes.llm_classes import *
from Classes.state_classes import OverallState

#############################################################
# SCHEMA DEFINITIONS
#############################################################
class Components(BaseModel):
    Components: list[str]

class NestedNumberFormat(BaseModel):
    scale: str
    decimals: int

# Define the lowest-level models first
class GridColumns(BaseModel):
    sm: Union[int, str]
    md: Union[int, str]
    lg: Union[int, str]

    class Config:
        extra = "forbid"

class ColSpan(BaseModel):
    sm: Union[int, str]
    md: Union[int, str]
    lg: Union[int, str]

    class Config:
        extra = "forbid"

class NestedNumberFormat(BaseModel):
    scale: str
    decimals: int

    class Config:
        extra = "forbid"

# Define rows and columns next
class Component(BaseModel):
    id: str
    type: str
    title: Optional[str] = None
    AI_Generation_Description: Optional[str] = Field(
        None,
        alias="AI Generation Description"
    )
    noborder: Optional[bool] = None
    height: Optional[int] = None
    maxheight: Optional[int] = None
    minheight: Optional[int] = None
    numberFormat: Optional[NestedNumberFormat] = None
    config: Optional["Layout"] = None  # Use forward reference for recursive Layout

    class Config:
        populate_by_name = True
        extra = "forbid"

class Column(BaseModel):
    colSpan: ColSpan
    components: List[Component]

    class Config:
        extra = "forbid"

class Row(BaseModel):
    columns: List[Column]

    class Config:
        extra = "forbid"

# Define layout after rows and columns
class Layout(BaseModel):
    gridColumns: Optional[GridColumns] = None
    rows: List[Row]

    class Config:
        extra = "forbid"

# Define the top-level models
class NumberFormat(BaseModel):
    currency: str
    scale: str
    decimals: int

    class Config:
        extra = "forbid"

class ReportConfig(BaseModel):
    reportTitle: str
    numberFormat: NumberFormat
    layout: Layout

    class Config:
        extra = "forbid"
class conceptualdesign_reply(BaseModel):
    conceptualDesign: str

def generate_dimensions_string(state: OverallState) -> str:
    # Extract and clean POV dimensions
    pov_dimensions = [
        {"Name": dim["Name"], "Alias": dim["Alias"]}
        for dim in state.get("POV", [])
        if dim.get("InUse", "True") == "True"
    ]
    # Format the dimensions into a string for the prompt
    pov_string = "\n".join([f"- {dim['Name']} (Alias: {dim['Alias']})" for dim in pov_dimensions])
    return pov_string

def generate_conceptualdesign(state: OverallState):
    system_instructions = load_xml_instructions("conceptualdesign_prompt.xml")
    system_msg = SystemMessage(content=system_instructions)
    
    pov_string = generate_dimensions_string(state)

    prompt = {"userInstructions": state["ReportQuery"]}
    user_prompt = (
        f"{prompt}\n\n"
        "The following dimensions are available in the model:\n"
        f"{pov_string}"
    )
    user_msg = HumanMessage(content=json.dumps(user_prompt)) 

    structured_llm = modelSpec.with_structured_output(
        conceptualdesign_reply,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")
    parsed_output = output["parsed"].model_dump()
    return {"ConceptualDesign": parsed_output["conceptualDesign"]}



def generate_layout(state: OverallState):
    system_instructions = load_xml_instructions("render_layout.xml")
    system_msg = SystemMessage(content=system_instructions)

    # Format the dimensions into a string for the prompt
    pov_string = generate_dimensions_string(state)

    # Update the user message with dimensions
    user_prompt = (
        f"{state['ConceptualDesign']}\n\n"
        "The following dimensions are available in the model:\n"
        f"{pov_string}"
    )

    user_msg = HumanMessage(content=user_prompt)
    report_metadata = state["ReportMetadata"]

    structured_llm = modelSpec.with_structured_output(
        ReportConfig,
        method="json_mode",
        include_raw=True
    )
    conversation = [system_msg] + [user_msg]
    output = structured_llm.invoke(conversation, stream=False, response_format="json")

    if output["parsed"] is None:
        # Construct a meaningful error message
        error_message = (
            "Parsing failed: The 'parsed' field in the output is None.\n"
            "Raw LLM Output:\n"
            f"{output.get('raw', 'No raw output available')}\n\n"
            "Additional Context:\n"
            f"Model Name: {output.get('response_metadata', {}).get('model_name', 'Unknown')}\n"
            f"Token Usage: {output.get('response_metadata', {}).get('token_usage', 'Unknown')}\n"
            f"Finish Reason: {output.get('response_metadata', {}).get('finish_reason', 'Unknown')}\n"
            "Please check the raw output for errors or unexpected formatting."
        )
        raise Exception(error_message)
    parsed_output = output["parsed"].model_dump(exclude_none=True)
    


    # Find components
    components = []
    def walk(obj):
        if isinstance(obj, dict):
            if "components" in obj and isinstance(obj["components"], list):
                for c in obj["components"]:
                    components.append(c)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(parsed_output)
    
    parsed_output["POV"] = state["POV"]

    # Output both JsonLayout and Components
    return {"JsonLayout": parsed_output, "Components": components, "ReportMetadata": report_metadata}

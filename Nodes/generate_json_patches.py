###### APPLY JSON PATCHES FUNCTION ########
import json
from typing import Dict, List
import jsonpatch

from langchain_core.messages import (
    AIMessage, 
    HumanMessage,
    SystemMessage,
    BaseMessage
)

from Classes.state_classes import ModifyReportState
from Nodes.load_xml_instructions import load_xml_instructions
from Classes.llm_classes import modelVers

def modify_json(json_str: Dict, operation_str: Dict):
    modified_json = json_str

    for item in operation_str["items"]:
        if "items" in operation_str and isinstance(operation_str["items"], list):
            operation_object = item #operation_str["items"][0]  # Extract the first operation from the list
        else:
            raise ValueError("Invalid operation data: Expected a list of operations under 'items'.")

        operation_path = operation_object.get("path")
        operation_type = operation_object.get("op")
        replace_value = operation_object.get("value")
        from_path = operation_object.get("from")

        if not operation_path or not operation_type:
            raise ValueError("Invalid operation data: Missing path or operation type.")

        patch_operation = None
        if operation_type == "remove":
            patch_operation = [{"op": "remove", "path": operation_path}]
        elif operation_type == "replace":
            patch_operation = [{"op": "replace", "path": operation_path, "value": replace_value}]
        elif operation_type == "add":
            patch_operation = [{"op": "add", "path": operation_path, "value": replace_value}]
        elif operation_type == "move":
            if not from_path:
                raise ValueError("Invalid move operation: Missing 'from' path.")
            patch_operation = [{"op": "move", "from": from_path, "path": operation_path}]
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")

        patch = jsonpatch.JsonPatch(patch_operation)
        modified_json = patch.apply(modified_json)

    return json.dumps(modified_json, indent=4)

###### GENERATE JSON PATCHES #########

def json_patches(BaseModel):
    items: List[dict]

def generate_json_patches(state: ModifyReportState):
    system_instructions = load_xml_instructions("modifyreport_prompt.xml")
    system_msg = SystemMessage(content=system_instructions)
    
    inputMessage = {
        "instruction":state['instruction'], 
        "input_json":state['input_json']
    }
    user_msg = HumanMessage(content=json.dumps(inputMessage))

    structured_llm = modelVers.with_structured_output(
        json_patches,
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
    
    operation_output = output["parsed"]

    state["output_json"] = modify_json(state["input_json"],operation_output)

    #return state
    return { "output_json": state["output_json"] }


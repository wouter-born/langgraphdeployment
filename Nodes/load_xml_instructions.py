import os
import requests

def load_xml_instructions(filename: str, base_url: str = "https://thepetshop.backend.cfolytics.com/api/Prompt/get-prompts") -> str:
    """
    Load system instructions from a remote HTTP location.

    Parameters:
    - filename: The name of the XML file to retrieve.
    - base_url: The base URL where the XML files are stored.

    Returns:
    - The content of the XML file as a string.

    Raises:
    - HTTPError: If the HTTP request returned an unsuccessful status code.
    """

    # Remove the .xml extension if present
    filename = os.path.splitext(filename)[0]

    url = f"{base_url}"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    response_json = response.json()
    for p in response_json:
        if p["PromptID"] == filename:
            return p["PromptText"]
    return "Prompt Not Found!"


# Manually set the __file__ variable to the notebook's directory
__file__ = os.path.abspath("notebook_name.ipynb")

def load_xml_instructions_local(filename: str) -> str:
    """
    Load system instructions from 'XML_instructions/filename' if you keep them externally.
    Otherwise, just inline your prompts as strings.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "XML_instructions", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    

# Example usage:
if __name__ == "__main__":
    xml_content = load_xml_instructions("render_layout.xml")
    print(xml_content)

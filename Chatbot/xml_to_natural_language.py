import xml.etree.ElementTree as ET
from collections import OrderedDict

def clean_text(text):
    """
    Removes extra spaces from the provided text.
    """
    return " ".join(text.split())

def xml_to_natural_language_with_tags(xml_file_path):
    """
    Reads an XML file and extracts all text content,
    grouping text by unique tag names so that each tag appears only once.
    Each tag's aggregated text is printed on a new line.
    Extra spaces in the text are removed.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    tag_texts = OrderedDict()

    def traverse(element):
        # Process the element's text if available.
        if element.text:
            text = clean_text(element.text)
            if text:
                if element.tag not in tag_texts:
                    tag_texts[element.tag] = []
                tag_texts[element.tag].append(text)
        # Recursively process child elements.
        for child in element:
            traverse(child)
            # Process the child's tail text if available.
            if child.tail:
                tail_text = clean_text(child.tail)
                if tail_text:
                    if child.tag not in tag_texts:
                        tag_texts[child.tag] = []
                    tag_texts[child.tag].append(tail_text)

    traverse(root)
    
    # Build the final output with each unique tag on a new line.
    lines = []
    for tag, texts in tag_texts.items():
        combined_text = " ".join(texts)
        lines.append(f"{tag}: {combined_text}")
    
    return "\n".join(lines)

# Example usage:
if __name__ == "__main__":
    xml_file_path = "prompt_template.xml"  # Use the file prompt_template.xml
    natural_language_text = xml_to_natural_language_with_tags(xml_file_path)
    print(natural_language_text)
from typing import List
import json
import os

# Manually set the __file__ variable to the notebook's directory
__file__ = os.path.abspath("notebook_name.ipynb")

# --- Optimization helper functions ---
def optimize_report_metadata(report_metadata: List[dict]) -> List[dict]:
    optimized = []
    for item in report_metadata:
        optimized.append({
            "name": item.get("name"),
            "alias": item.get("alias"),
            "default": item.get("default"),
            # Process dimensionContent: first remove bottom level members
            "dimensionContent": [x['Name'] for x in remove_bottom_level_members_from_dimension(item.get("dimensionContent", []))]
        })
    return optimized

def optimize_pov(pov: List[dict]) -> List[dict]:
    optimized = []
    for item in pov:
        optimized.append({
            "Name": item.get("Name"),
            "Default": item.get("Default")
        })
    return optimized

def optimize_existing_lists(existing_lists: List[dict]) -> List[dict]:
    optimized = []
    for item in existing_lists:
        list_contents = item.get("ListContents", {})
        optimized.append({
            "ListName": item.get("ListName"),
            "item_count": len(list_contents.get("items", [])),
            "CreateDescription": item.get("CreateDescription")
        })
    return optimized

# --- New Function 1: Remove bottom level members ---
def remove_bottom_level_members_from_dimension(dimension: List[dict]) -> List[dict]:
    """
    Returns a filtered dimension list keeping only items whose ID is used as a ParentID by at least one item.
    """
    # Collect all ParentIDs (skip None values)
    parent_ids = {member.get("ParentID") for member in dimension if member.get("ParentID") is not None}
    # Filter out members whose own ID is not in the set of ParentIDs.
    filtered = [member for member in dimension if member.get("ID") in parent_ids]
    return filtered

# --- New Function 2: Convert ai_data into TSV format with Titles ---
def ai_data_to_tsv(ai_data: dict) -> str:
    """
    Converts ai_data into a TSV string with titles.
    Produces two sections: one for ReportMetadata and one for POV.
    """
    lines = []
    # Process ReportMetadata
    lines.append("=== ReportMetadata ===")
    # Header for ReportMetadata (flattening dimensionContent to a comma separated string)
    rm_header = ["name", "alias", "default", "dimensionContent"]
    lines.append("\t".join(rm_header))
    for item in ai_data.get("ReportMetadata", []):
        # Join the dimensionContent list into a single comma-separated string
        dim_str = ", ".join(item.get("dimensionContent", []))
        row = [str(item.get("name", "")),
               str(item.get("alias", "")),
               str(item.get("default", "")),
               dim_str]
        lines.append("\t".join(row))
    
    lines.append("")  # Blank line between sections

    # Process POV
    lines.append("=== POV ===")
    pov_header = ["Name", "Default"]
    lines.append("\t".join(pov_header))
    for item in ai_data.get("POV", []):
        row = [str(item.get("Name", "")),
               str(item.get("Default", ""))]
        lines.append("\t".join(row))
    
    return "\n".join(lines)


def model_optimizer(filename:str) -> str:
    with open(filename, 'r') as file:
        input_data = json.load(file)
    
    ai_data = {
        "ReportMetadata": optimize_report_metadata(input_data["ReportMetadata"]),
        "POV": optimize_pov(input_data["POV"])
    }
    # Convert to TSV and print
    tsv_output = ai_data_to_tsv(ai_data)
    return tsv_output


# --- Main execution ---
if __name__ == "__main__":
    # Load your input_data from JSON file
    with open('samplesb.json', 'r') as file:
        input_data = json.load(file)
    
    ai_data = {
        "ReportMetadata": optimize_report_metadata(input_data["ReportMetadata"]),
        "POV": optimize_pov(input_data["POV"])
    }
    
    # Print the JSON output (optional)
    print("Optimized ai_data (JSON):")
    print(json.dumps(ai_data, indent=2))
    
    # Convert to TSV and print
    tsv_output = ai_data_to_tsv(ai_data)
    print("\nConverted ai_data (TSV):")
    print(tsv_output)
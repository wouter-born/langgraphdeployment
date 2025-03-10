

from typing import Any, Dict, List


def build_storyboard(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Converts multiple 'report' JSON objects into a single storyboard JSON,
    with these modifications:
      1) Removes "ScaleSelectors" components
      2) Renames component IDs by prefixing the page number (C1 -> P1C1).
    """

    if not reports:
        # Edge case: if empty, return a minimal structure
        return {"POV": [], "numberFormat": {}, "sections": [], "lists": {}}

    # 1) POV and numberFormat from the first report
    storyboard = {
        "POV": reports[0].get("POV", []),
        "numberFormat": reports[0].get("numberFormat", {}),
        "sections": [],
        "lists": {}
    }

    # Helper function: recursively rename IDs & remove ScaleSelectors
    def rename_and_filter_rows(rows: List[Dict[str, Any]], page_number: int):
        """
        In-place:
          - Skips (removes) any component whose type == "ScaleSelectors"
          - Renames 'id' -> f"P{page_number}{old_id}"
          - Recursively processes NestedRow -> config.rows
        """
        for row in rows:
            columns = row.get("columns", [])
            for col in columns:
                new_components = []
                for comp in col.get("components", []):
                    # 1) Skip ScaleSelectors
                    if comp.get("type") == "ScaleSelectors":
                        continue
                    # 2) Rename the 'id' if present
                    if "id" in comp:
                        old_id = comp["id"]
                        comp["id"] = f"P{page_number}{old_id}"
                    # 3) If NestedRow, recurse into its config.rows
                    if comp.get("type") == "NestedRow":
                        subrows = comp.get("config", {}).get("rows", [])
                        rename_and_filter_rows(subrows, page_number)
                    new_components.append(comp)
                col["components"] = new_components

    # 2) For each report in the list
    for i, r in enumerate(reports):
        page_number = i + 1  # 1-based page numbering

        # (a) Make a Header row for the reportTitle
        header_section = {
            "type": "Row",
            "gridColumns": {
                "sm": 8,
                "md": 8,
                "lg": 8
            },
            "columns": [
                {
                    "colSpan": {
                        "sm": 8,
                        "md": 8,
                        "lg": 8
                    },
                    "components": [
                        {
                            "type": "Header",
                            "content": r.get("reportTitle", "Untitled Report"),
                            "align": "left"
                        }
                    ]
                }
            ]
        }
        storyboard["sections"].append(header_section)

        # (b) Copy the rows from the report into the storyboard
        layout = r.get("layout", {})
        grid_cols = layout.get("gridColumns", {"sm": 8, "md": 8, "lg": 8})
        rows = layout.get("rows", [])

        # Rename IDs and remove ScaleSelectors before adding them
        rename_and_filter_rows(rows, page_number)

        for row in rows:
            new_section = {
                "type": "Row",
                "gridColumns": grid_cols,
                "columns": row.get("columns", [])
            }
            storyboard["sections"].append(new_section)

        # (c) Add a pagebreak after each report
        storyboard["sections"].append({"type": "PageBreak"})

        # (d) Gather the lists from this report
        report_lists = r.get("lists", {})
        for list_key, list_val in report_lists.items():
            # If there's a collision in list keys, you can decide whether to overwrite or skip
            if list_key not in storyboard["lists"]:
                storyboard["lists"][list_key] = list_val
            else:
                # Overwrite or skip? Here we skip if it already exists
                pass

    return storyboard


def remove_null_entries(data):
    """
    Recursively removes keys with null (None) values from a dictionary or list.
    """
    if isinstance(data, dict):
        return {k: remove_null_entries(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [remove_null_entries(v) for v in data]
    else:
        return data

# Example usage:
# cleaned_data = remove_null_entries(parsed_output)

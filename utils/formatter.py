import re


def format_disease_info(text: str):
    """
    Convert ChromaDB text into structured sections.
    """

    fields = {
        "Disease": "",
        "Plant": "",
        "Symptoms": "",
        "Cause": "",
        "Treatment": "",
        "Prevention": ""
    }

    for key in fields.keys():
        pattern = rf"{key}:\s*(.*?)(?=(Disease|Plant|Symptoms|Cause|Treatment|Prevention):|$)"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            fields[key] = match.group(1).strip()

    return fields
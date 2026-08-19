import re


def parse_gemini_response(response):

    disease = "Unknown"
    plant = "Tomato"
    confidence = 0.0

    disease_match = re.search(
        r"Disease:\s*(.*)",
        response,
        re.IGNORECASE
    )

    plant_match = re.search(
        r"Plant:\s*(.*)",
        response,
        re.IGNORECASE
    )

    confidence_match = re.search(
        r"Confidence:\s*([0-9]+(?:\.[0-9]+)?)",
        response,
        re.IGNORECASE
    )

    if disease_match:
        disease = disease_match.group(1).strip()

    if plant_match:
        plant = plant_match.group(1).strip()

    if confidence_match:
        confidence = float(confidence_match.group(1))

    return disease, plant, confidence
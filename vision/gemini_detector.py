from google import genai
from PIL import Image

from config import GEMINI_API_KEY, VISION_MODEL

# Create Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def detect_disease(image_path: str):

    image = Image.open(image_path)

    prompt = """
You are a tomato plant disease classification assistant.

The image is a tomato leaf.

You MUST choose exactly ONE disease from this list:

1. Tomato___Bacterial_spot
2. Tomato___Early_blight
3. Tomato___healthy
4. Tomato___Late_blight
5. Tomato___Leaf_Mold
6. Tomato___Septoria_leaf_spot
7. Tomato___Spider_mites Two-spotted_spider_mite
8. Tomato___Target_Spot
9. Tomato___Tomato_mosaic_virus
10. Tomato___Tomato_Yellow_Leaf_Curl_Virus

Do NOT return any disease outside this list.

Return ONLY:

Disease: <one disease from the list>
Plant: Tomato
Confidence: <number from 0 to 100>

If you cannot confidently identify one of these diseases, return:

Disease: Unknown
Plant: Tomato
Confidence: 0
"""

    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[
            prompt,
            image
        ]
    )

    return response.text.strip()
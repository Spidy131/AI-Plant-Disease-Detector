from google import genai
from PIL import Image

from config import GEMINI_API_KEY, VISION_MODEL


# Create Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


def detect_disease(image_path: str):

    image = Image.open(image_path)

    prompt = """
You are a tomato plant disease classification assistant.

Analyze the uploaded tomato plant/leaf image and identify the most likely disease.

IMPORTANT:

- Identify the actual disease visible in the image.
- The disease does NOT have to belong to the 10 diseases listed below.
- You may return a disease outside this list.
- Do NOT force the prediction into the 10-class list.
- Do NOT return "Unknown" unless the image genuinely cannot be interpreted.

Known diseases in the current knowledge base include:

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

The above list is ONLY a reference for diseases currently available
in the knowledge base. It is NOT a restriction on your prediction.

Return ONLY:

Disease: <actual detected disease name>
Plant: Tomato
Confidence: <number from 0 to 100>
"""

    try:

        response = client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                prompt,
                image
            ]
        )

        return response.text.strip()

    except Exception as e:

        print("========== GEMINI ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("==================================")

        raise

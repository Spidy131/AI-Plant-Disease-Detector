from embeddings.query_db import search_disease
from vision.gemini_detector import detect_disease


def analyze_leaf(image_path):
    """
    Complete RAG Pipeline
    """

    # Step 1: Detect disease from image
    detection = detect_disease(image_path)

    print("\nGemini Detection")
    print(detection)

    # Step 2: Search ChromaDB
    disease_info = search_disease(detection)

    return detection, disease_info
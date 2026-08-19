import json
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, EMBEDDING_MODEL

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Create Chroma client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Create collection
collection = client.get_or_create_collection("plant_diseases")

# Load disease data
with open("data/diseases.json", "r", encoding="utf-8") as file:
    diseases = json.load(file)

for index, disease in enumerate(diseases):

    document = f"""
    Disease: {disease['disease']}
    Plant: {disease['plant']}
    Symptoms: {disease['symptoms']}
    Cause: {disease['cause']}
    Treatment: {disease['treatment']}
    Prevention: {disease['prevention']}
    """

    embedding = model.encode(document).tolist()

    collection.add(
        ids=[str(index)],
        documents=[document],
        embeddings=[embedding],
        metadatas=[{"disease": disease["disease"]}]
    )

print("✅ ChromaDB created successfully!")
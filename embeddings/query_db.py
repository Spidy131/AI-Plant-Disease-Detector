import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_collection("plant_diseases")


def search_disease(query, threshold=0.8):

    results = collection.query(
        query_texts=[query],
        n_results=1,
        include=["documents", "distances"]
    )

    if not results["documents"]:
        return None

    if not results["documents"][0]:
        return None

    distance = results["distances"][0][0]

    print("Query:", query)
    print("Chroma distance:", distance)

    if distance > threshold:
        return None

    return results["documents"][0][0]
import os
from pathlib import Path
from dotenv import load_dotenv

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# Gemini Configuration
# ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VISION_MODEL = "gemini-2.5-flash"

# ==============================
# ChromaDB Configuration
# ==============================
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ==============================
# Dataset Configuration
# ==============================
DATASET_PATH = Path(
    r"C:\Users\user\python_codes\AI-Plant-Disease-Detector\datasets\tomato_dataset"
)

# ==============================
# Model Configuration
# ==============================
MODEL_PATH = "models/tomato_model.keras"

# ==============================
# Training Configuration
# ==============================
IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.2

SEED = 42

EPOCHS = 10

LEARNING_RATE = 0.0001
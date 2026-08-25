🌿 AI Plant Disease Detector

An AI-powered tomato plant disease detection application that combines Gemini Vision, ChromaDB-based RAG, and Streamlit to identify tomato leaf diseases and provide relevant disease information.

🚀 Live Demo

Try the application here:https://ai-plant-disease-detector-bwsupagp3udwewxmpwdria.streamlit.app/?utm_source=chatgpt.com

🌿 AI Plant Disease Detector — Live Demo

📌 Project Overview

The AI Plant Disease Detector analyzes an uploaded tomato leaf image and predicts the possible disease using Google Gemini 2.5 Flash.

After identifying the disease, the application uses a ChromaDB vector database to retrieve relevant information from the plant-disease knowledge base.

The application provides:

🍅 Disease prediction
📊 Prediction confidence
⚡ Prediction time
🦠 Disease symptoms
⚠️ Disease cause
💊 Treatment recommendations
🛡️ Prevention methods
🧠 Architecture
                    ┌──────────────────────┐
                    │    Tomato Leaf       │
                    │       Image          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Gemini 2.5 Flash     │
                    │ Vision Classification│
                    └──────────┬───────────┘
                               │
                       Disease Prediction
                               │
                               ▼
                    ┌──────────────────────┐
                    │    ChromaDB RAG      │
                    │ Knowledge Retrieval  │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │ Disease Information             │
              │                                │
              │ • Symptoms                     │
              │ • Cause                        │
              │ • Treatment                    │
              │ • Prevention                   │
              └────────────────────────────────┘
✨ Features
1. 🍅 AI Disease Detection

The application accepts:

.jpg
.jpeg
.png

tomato leaf images.

Gemini Vision analyzes the image and selects one of the supported tomato disease classes.

2. 🧠 Gemini Vision

The project uses:

Gemini 2.5 Flash

for image-based disease classification.

The model is instructed to classify the tomato leaf into the predefined disease categories.

3. 🔎 ChromaDB RAG

After disease detection, the predicted disease is used as a query against the ChromaDB vector database.

Example:

Gemini Prediction
       ↓
Early Blight
       ↓
"Tomato Early Blight"
       ↓
ChromaDB
       ↓
Relevant disease information
4. 📚 Disease Knowledge

The application retrieves:

Information	Description
🦠 Symptoms	Visible symptoms associated with the disease
⚠️ Cause	Pathogen/cause of the disease
💊 Treatment	Recommended treatment information
🛡️ Prevention	Methods to reduce disease occurrence
5. 📊 Confidence Score

The application displays the model's confidence:

Confidence: 90.00%

and provides a visual confidence indicator.

6. ⚡ Prediction Time

The application also measures the time required to process the uploaded image.

Example:

Prediction Time: 6.70s
🦠 Supported Tomato Classes

The application is configured for the following 10 classes:

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

If Gemini cannot confidently identify one of the supported classes, the application can return:

Disease: Unknown
Plant: Tomato
Confidence: 0
🛠️ Technology Stack
Frontend
Streamlit
AI / Computer Vision
Google Gemini
Gemini 2.5 Flash
Pillow
RAG / Knowledge Retrieval
ChromaDB
Sentence Transformers
FAISS/vector-search concepts
Machine Learning
TensorFlow
Keras
EfficientNetB0
Programming Language
Python
Deployment
Streamlit Community Cloud
GitHub
📁 Project Structure
AI-Plant-Disease-Detector/
│
├── app.py
├── config.py
├── requirements.txt
├── .gitignore
│
├── app/
│   └── main.py
│
├── data/
│   └── diseases.json
│
├── chroma_db/
│   ├── chroma.sqlite3
│   └── ...
│
├── embeddings/
│   ├── create_db.py
│   └── query_db.py
│
├── models/
│   └── tomato_model.keras
│
├── services/
│   └── rag_service.py
│
├── training/
│   ├── callbacks.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model_builder.py
│   ├── predict.py
│   └── train_model.py
│
├── ui/
│   └── streamlit_ui.py
│
├── utils/
│   ├── formatter.py
│   ├── gemini_parser.py
│   └── helpers.py
│
└── vision/
    └── gemini_detector.py
🔄 Application Workflow
Step 1 — Upload Image

The user uploads a tomato leaf image through Streamlit.

Upload Tomato Leaf
        ↓
Image preprocessing
Step 2 — Gemini Analysis

The image is sent to Gemini Vision.

Image
  ↓
Gemini 2.5 Flash
  ↓
Disease
  ↓
Confidence
Step 3 — Parse Prediction

The Gemini response is parsed to extract:

Disease
Plant
Confidence
Step 4 — ChromaDB Retrieval

The detected disease is used to query the ChromaDB collection:

Tomato Early Blight
        ↓
ChromaDB
        ↓
Nearest knowledge document
Step 5 — Display Results

The application displays:

🍅 Prediction
Early Blight

📊 Confidence
90%

📚 Disease Information

🦠 Symptoms
⚠️ Cause
💊 Treatment
🛡️ Prevention

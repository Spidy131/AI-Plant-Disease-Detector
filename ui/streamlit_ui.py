import streamlit as st
import os
from training.predict import predict_image
st.set_page_config(
    page_title="AI Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 AI Plant Disease Detector")

st.markdown(
    """
Detect tomato leaf diseases using:

- 🤖 Local AI Model (TensorFlow)
- ✨ Gemini 2.5 Flash
- 📚 ChromaDB Knowledge Base
"""
)

mode = st.radio(
    "Choose Prediction Mode",
    [
        "Local AI",
        "Gemini AI"
    ]
)

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    st.image(uploaded_file, width=350)

    # Save uploaded image temporarily
    temp_path = os.path.join("temp", uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Detecting disease..."):

        disease, confidence = predict_image(temp_path)

    st.success("Prediction Completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Disease", disease.replace("Tomato___", ""))

    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")
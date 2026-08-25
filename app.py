import os
import time
import tempfile

import streamlit as st

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from vision.gemini_detector import detect_disease
from utils.formatter import format_disease_info
from utils.gemini_parser import parse_gemini_response
from embeddings.query_db import search_disease
from utils.formatter import format_disease_info
#from training.predict import predict_image
#$from embeddings.query_db import search_disease


st.set_page_config(
    page_title="AI Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
.main-title{
    font-size:40px;
    font-weight:700;
    color:#2E8B57;
}
.card{
    background:#f5f7fa;
    padding:18px;
    border-radius:12px;
    border:1px solid #ddd;
    margin-bottom:12px;
}
.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🌿 AI Plant Disease Detector")
    mode = st.radio(
        "Prediction Engine",
        ["🧠 Local AI", "✨ Gemini AI"]
    )
    st.divider()
    st.markdown("### Model")
    st.info("EfficientNetB0\n\nPlantVillage\n\n10 Tomato Classes")
    st.divider()
    st.caption("Built by Sachin Malode")

st.markdown('<div class="main-title">🌿 AI Plant Disease Detector</div>', unsafe_allow_html=True)
st.caption("TensorFlow • Gemini • ChromaDB")

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    left, right = st.columns([1,2])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getbuffer())
        image_path = tmp.name

    with left:
        st.subheader("📷 Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    with right:

        start = time.time()

        with st.spinner("🔍 Analyzing leaf..."):
            if mode.startswith("🧠"):
                from training.predict import predict_image
                disease, confidence = predict_image(image_path)
                disease_name = disease.replace("Tomato___","").replace("_"," ")
            else:
                response = detect_disease(image_path)

                disease_name, plant_name, confidence = parse_gemini_response(response)

                # Clean Gemini disease name
                disease_name = (
                    disease_name
                    .replace("Tomato___", "")
                    .replace("_", " ")
                    .strip()
                )

        elapsed = time.time()-start

        st.subheader("🍅 Prediction")

        c1,c2,c3 = st.columns(3)

        with c1:
            #st.metric("Disease", disease_name.title())
            st.write(f"### {disease_name.title()}")

        with c2:
            if confidence is not None:
                confidence=float(confidence)
                st.metric("Confidence",f"{confidence:.2f}%")
                st.progress(confidence/100)
            else:
                st.metric("Confidence","N/A")
                if confidence is None:

                    st.metric(
                        "Confidence",
                        "Gemini AI"
                    )

                   

        with c3:
            st.metric("Prediction Time",f"{elapsed:.2f}s")

        if confidence is not None:
            if confidence>=90:
                st.success("🟢 Excellent Confidence")
            elif confidence>=75:
                st.warning("🟡 Good Confidence")
            else:
                st.error("🔴 Low Confidence")

        query = disease_name

        try:
         knowledge = search_disease(f"Tomato {disease_name}")

if knowledge is None:

    st.warning(
        "⚠️ Disease information is not available "
        "in the current knowledge base."
    )

else:

    info = format_disease_info(knowledge)

    st.divider()
    st.subheader("📚 Disease Information")

    e1, e2 = st.columns(2)

    with e1:
        with st.expander("🦠 Symptoms", expanded=True):
            st.write(info["Symptoms"])

        with st.expander("⚠️ Cause", expanded=True):
            st.write(info["Cause"])

    with e2:
        with st.expander("💊 Treatment", expanded=True):
            st.write(info["Treatment"])

        with st.expander("🛡 Prevention", expanded=True):
            st.write(info["Prevention"])
        except Exception as e:
            st.warning(f"Knowledge retrieval failed: {e}")

    os.remove(image_path)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div class="footer">🌿 Built with TensorFlow • Streamlit • ChromaDB • Gemini 2.5 Flash</div>',
    unsafe_allow_html=True
)

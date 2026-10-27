"""PathoCell AI - Streamlit demo.  Run: streamlit run app/app.py"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from PIL import Image
from inference import predict

st.set_page_config(page_title="PathoCell AI", page_icon="🔬", layout="wide")

with st.sidebar:
    st.title("🔬 PathoCell AI")
    st.caption("Explainable Histopathology Cancer Detection Engine")
    st.markdown("---")
    st.markdown(
        "**About this tool**\n\nAcademic mini-project prototype. Classifies a histopathology "
        "image as *benign* or *malignant* and shows a Grad-CAM heatmap of the regions that "
        "most influenced the prediction.\n\n⚠️ **Not a certified diagnostic tool.**"
    )
    st.markdown("---")
    st.markdown("**Team:** Devansh · Navneet Soni · Himanshu Chaudhary")
    st.markdown("**GLA University**, B.Tech CSE (AI & ML)")

st.title("PathoCell AI")
st.subheader("Upload a histopathology image to get a prediction and explanation")

uploaded = st.file_uploader("Upload tissue image (PNG/JPG)", type=["png", "jpg", "jpeg"])
col1, col2 = st.columns(2)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    with col1:
        st.markdown("### Uploaded Image")
        st.image(image, use_container_width=True)
    with st.spinner("Running model + generating explanation..."):
        result = predict(image)
    with col2:
        st.markdown("### Explainability Heatmap (Grad-CAM)")
        st.image(result["heatmap"], use_container_width=True)
    st.markdown("---")
    msg = f"### Prediction: **{result['label'].upper()}**  ·  Confidence: {result['confidence']}"
    (st.error if result["label"] == "malignant" else st.success)(msg)
    st.caption("Warmer colors (red/yellow) indicate regions with higher influence on the prediction.")
else:
    st.info("👆 Upload an image above to see a prediction and explanation.")

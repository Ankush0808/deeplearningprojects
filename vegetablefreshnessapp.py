import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from pathlib import Path

# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="Vegetable Freshness Detector",
    page_icon="🥦",
    layout="centered"
)


# -------------------------------------------------------------
# Custom Styling (Green Theme)
# -------------------------------------------------------------
st.markdown("""
    <style>
    /* App background and text */
    .stApp {
        background-color:#abd860; /* Rich green background */
        color: #181718; /* White text for contrast */
    }

    /* Button styling */
    div.stButton > button:first-child {
        background-color: white; /* White button */
        color: #2e7d32 !important; /* Green text */
        border-radius: 10px;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: 0.3s;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
    }

    /* Hover effect */
    div.stButton > button:hover {
        background-color:#20d860; /* Turn green on hover */
        color: black !important; /* White text on hover */
        border: 2px solid white;
    }

    /* Button when clicked or focused */
    div.stButton > button:focus, div.stButton > button:active {
        background-color: #1b5e20; /* Darker green on click */
        color: black !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: black; /* Keep headings white for visibility */
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# UI Header
# -------------------------------------------------------------
st.title("🥦 Vegetable Freshness Detection")
img_path = Path("C://Users//Lenovo//Downloads//veggies.png")
st.image(img_path)

# -------------------------------------------------------------
# File Upload
# -------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a vegetable image",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "jfif", "pjpeg", "pjp"]
)

# -------------------------------------------------------------
# Load Model (cached for performance)
# -------------------------------------------------------------
@st.cache_resource
def load_freshness_model():
    # Make sure this file is in the same directory or adjust the path
    model = load_model("C://Users//Lenovo//Downloads//freshness_model_pretrained.keras")
    return model

model = load_freshness_model()

# -------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------
def predict_freshness(img):
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize

    preds = model.predict(x)
    score = float(preds[0][0])  # Sigmoid output between 0–1
    return score

# -------------------------------------------------------------
# Prediction and Display Logic
# -------------------------------------------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    if st.button("Check Freshness"):
        with st.spinner("🔍 Analyzing freshness..."):
            score = predict_freshness(img)

            # If your model outputs higher values for spoiled → invert
            freshness_percent = (1 - score) * 100

            st.markdown("---")
            st.subheader(f"🌿 Freshness: **{freshness_percent:.1f}%**")
            st.progress(freshness_percent / 100)

            if freshness_percent >= 50:
                st.write("✅ The vegetable looks **Fresh!** 🥬")
            else:
                st.write("❌ The vegetable seems **Spoiled.** 🥀")

st.markdown("---")
st.markdown("Built with ❤️ by **Ankush Dilip Tonde** 🌱")

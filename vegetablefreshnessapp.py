import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
from pathlib import Path
import boto3
import tempfile
import os
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
    .stApp { background-color:#abd860; color: #181718; }
    div.stButton > button:first-child {
        background-color: white; color: #2e7d32 !important; border-radius: 10px;
        border: none; padding: 0.6em 1.2em; font-weight: 600; transition: 0.3s;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover { background-color:#20d860; color: black !important; border: 2px solid white; }
    div.stButton > button:focus, div.stButton > button:active { background-color: #1b5e20; color: black !important; }
    h1, h2, h3 { color: black; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# UI Header
# -------------------------------------------------------------
st.title("🥦 Vegetable Freshness Detection")
img_path = Path("veggies.png")
st.image(img_path)

# -------------------------------------------------------------
# Camera Input and File Upload
# -------------------------------------------------------------
st.write("📸 Take a picture using your camera or upload an image.")

camera_image = st.camera_input("Take a picture")
uploaded_file = st.file_uploader(
    "Or upload a vegetable image",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"]
)

# -------------------------------------------------------------
# Load Model (cached)
# -------------------------------------------------------------
@st.cache_resource
def load_freshness_model():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
    )

    bucket_name = "veggies-freshness"
    object_key = "freshness_model.tflite"

    temp_file_path = os.path.join(tempfile.gettempdir(), "temppy.tflite")

    # Download your model from S3
    s3.download_file(bucket_name, object_key, temp_file_path)

    # Load the model
    model = load_model(temp_file_path)
    return model

st.write("Loading model...")
model = load_freshness_model()
st.success("Model loaded successfully!")
# -------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------
def predict_freshness(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    score = float(preds[0][0])
    return score

# -------------------------------------------------------------
# Determine which image to use
# -------------------------------------------------------------
img_to_predict = None

if camera_image is not None:
    img_to_predict = Image.open(camera_image)
    st.image(img_to_predict, caption="📸 Captured Image", use_container_width=True)
elif uploaded_file is not None:
    img_to_predict = Image.open(uploaded_file)
    st.image(img_to_predict, caption="📸 Uploaded Image", use_container_width=True)

# -------------------------------------------------------------
# Prediction Button
# -------------------------------------------------------------
if img_to_predict is not None:
    if st.button("Check Freshness"):
        with st.spinner("🔍 Analyzing freshness..."):
            score = predict_freshness(img_to_predict)
            freshness_percent = (1 - score) * 100  # Adjust if your model outputs higher for spoiled

            st.markdown("---")
            st.subheader(f"🌿 Freshness: **{freshness_percent:.1f}%**")
            st.progress(freshness_percent / 100)

            if freshness_percent >= 50:
                st.write("✅ The vegetable looks **Fresh!** 🥬")
            else:
                st.write("❌ The vegetable seems **Spoiled.** 🥀")

st.markdown("---")
st.markdown("Built with ❤️ by **Ankush Dilip Tonde** 🌱")

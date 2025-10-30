import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Vegetable Freshness Detector", page_icon="🥦", layout="centered")

st.title("🥕 Vegetable Freshness Prediction App")
st.write("Upload an image of a vegetable, and the model will predict its **freshness percentage**.")

# Accept all common image formats
uploaded_file = st.file_uploader(
    "Upload a vegetable image",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "jfif", "pjpeg", "pjp"]
)

# ---------------------------------------------------------------
@st.cache_resource
def load_freshness_model():
    model = load_model("C://Users//Lenovo//Downloads//freshness_model_pretrained.keras")  # path to your saved model
    return model

model = load_freshness_model()

# -----------------------------------------------------------
# Prediction Function
# -----------------------------------------------------------
def predict_freshness(img):
    # Convert any image type to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # normalize

    # Predict probability (0 to 1)
    preds = model.predict(x)
    score = float(preds[0][0])  # sigmoid output
    return score

# -----------------------------------------------------------
# Display Prediction
# -----------------------------------------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing freshness..."):
        score = predict_freshness(img)

        # If model outputs higher value for spoiled → invert the score
        freshness_percent = (1 - score) * 100

        st.subheader(f"🌿 Freshness: **{freshness_percent:.1f}%**")

        # Add progress bar visualization
        st.progress(min(int(freshness_percent), 100))

        if freshness_percent >= 50:
            st.success("✅ The vegetable looks **Fresh!**")
        else:
            st.error("❌ The vegetable seems **Spoiled.**")

st.markdown("---")
st.markdown("Built by **Ankush Dilip Tonde** 🌱")

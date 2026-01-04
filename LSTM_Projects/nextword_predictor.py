# next_word_predictor_app.py

import streamlit as st
from pathlib import Path
import pickle
import base64
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔮",
    layout="centered"
)

# -----------------------------
# Background image (local)
# -----------------------------
image_path = Path("bgimage_nextword.png")  # your local image
with open(image_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

# -----------------------------
# Custom CSS
# -----------------------------
# -----------------------------
# Custom CSS
# -----------------------------
# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    f"""
    <style>
    /* App background */
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        color: white;  /* default text color */
    }}

    /* Main title */
    h1 {{
        color: #FFFFFF;       /* bright green */
        font-size: 1000px;      /* bigger font */
        font-weight: 1000;
    }}

    /* Description paragraph */
    p {{
        color: #FFFFFF;       /* bright green */
        font-size: 100px;      /* bigger than h2/h3 */
        font-weight: 1000;
    }}

    /* Other headings */
    h2, h3 {{
        color: white;
        font-size: 50px;
    }}

    /* Labels (for input box, etc.) */
    label {{
        color: white;
        font-size: 30px;
    }}

    /* Text input box */
    input[type="text"] {{
        font-size: 28px;         
        padding: 15px 20px;      
        border-radius: 20px;     
        background-color: rgba(255, 255, 255, 0.2);  
        color: white;            
    }}

    /* Placeholder inside text input */
    input[type="text"]::placeholder {{
        color: white;
        font-weight: 500;
        font-size: 28px;
    }}

    /* Buttons */
    div.stButton > button:first-child {{
        background-color: white; 
        color: #2e7d32 !important; 
        border-radius: 10px;
        border: none; 
        padding: 0.8em 1.5em; 
        font-weight: 600; 
        font-size: 22px;
        transition: 0.3s;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
    }}
    div.stButton > button:hover {{
        background-color:#20d860; 
        color: white !important; 
        border: 2px solid white;
    }}
    div.stButton > button:focus, div.stButton > button:active {{
        background-color: #1b5e20; 
        color: white !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Load model and tokenizer
# -----------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("next_word_model.h5")
max_len = 20  # replace with your training max_len

# -----------------------------
# Prediction function
# -----------------------------
def predict_top_k_words(model, tokenizer, text, k=3):
    """
    Predict the top k next words for a given input text.
    
    Args:
        model      : Trained language model
        tokenizer  : Tokenizer fitted on the training corpus
        text       : Input text string
        max_len    : Maximum sequence length used during training
        k          : Number of top predictions to return (default 3)
    
    Returns:
        List of top k predicted words
    """
    # Convert text to sequence of integers
    seq = tokenizer.texts_to_sequences([text])[0]
    
    # Pad sequence to match model input
    max_len = 20
    padded = pad_sequences([seq], maxlen=max_len-1, padding='pre')
    
    # Predict probabilities for all words in the vocabulary
    predicted_probs = model.predict(padded, verbose=0)[0]
    
    # Get indices of top k predictions
    top_indices = predicted_probs.argsort()[-k:][::-1]  # descending order
    
    # Map indices back to words
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    top_words = [index_word.get(i, "") for i in top_indices]
    
    return top_words

# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown('<h1>🔮 Next Word Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p>Type a sentence and press Enter to see the top 3 next-word suggestions.</p>', unsafe_allow_html=True)

# Use text_input so Enter submits
user_input = st.text_input(label="", placeholder="Enter text here:")

# Show predictions after Enter
if user_input.strip():
    suggestions = predict_top_k_words(model, tokenizer, user_input,k=3)
    st.success(f"Top predictions: **{', '.join(suggestions)}**")
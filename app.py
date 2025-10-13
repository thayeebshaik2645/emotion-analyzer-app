import streamlit as st
import pandas as pd
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="wide",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* --- Overall layout --- */
    .main {
        background: #f8fafc;
        padding: 2rem 3rem;
        font-family: 'Inter', sans-serif;
    }

    /* --- Title --- */
    h1 {
        color: #1e293b;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* --- Subheaders --- */
    h2, h3 {
        color: #334155;
        font-weight: 600;
    }

    /* --- Text area --- */
    textarea {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        padding: 10px !important;
        background-color: #ffffff !important;
    }

    /* --- Buttons --- */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        transition: all 0.2s ease-in-out;
        border: none;
    }

    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        transform: scale(1.02);
    }

    /* --- DataFrame styling --- */
    .stDataFrame {
        border-radius: 10px !important;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    }

    /* --- Footer --- */
    footer, .reportview-container .main footer {
        visibility: hidden;
    }

    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def initialize_classifier():
    """Initialize emotion classifier and cache."""
    try:
        with st.spinner(f"Loading model `{MODEL_NAME}`..."):
            classifier = pipeline(
                "text-classification",
                model=MODEL_NAME,
                return_all_scores=True
            )
        st.success("✅ Model loaded successfully!")
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def detect_emotions(classifier, texts):
    if not texts:
        return []

    predictions = classifier(texts)
    results = []
    for text, prediction_list in zip(texts, predictions):
        best = max(prediction_list, key=lambda x: x['score'])
        results.append({
            "Input Text": text,
            "Dominant Emotion": best['label'].upper(),
            "Confidence": f"{best['score']:.4f}"
        })
    return results

# --- HEADER ---
st.title("🧠 Emotion Detector Dashboard")
st.markdown("""
Detect emotions in text using a fine-tuned Transformer model.
Simply paste your sentences below and click **Analyze**!
""")

st.divider()

# --- INPUT ---
st.subheader("📝 Step 1: Enter Text(s) to Analyze")

default_text = """I am so incredibly happy and proud of what we achieved today!
This is confusing; I need someone to clarify the instructions for step three.
My heart is racing, I'm genuinely terrified of what might happen next."""

input_text = st.text_area(
    "Enter one sentence per line:",
    value=default_text,
    height=180
)
texts = [t.strip() for t in input_text.split("\n") if t.strip()]

# --- ANALYZE BUTTON ---
analyze = st.button("🔍 Analyze Emotions")

classifier = initialize_classifier()

# --- RESULTS ---
if analyze:
    if texts:
        st.subheader("📊 Step 2: Results")
        with st.spinner("Analyzing emotions..."):
            results = detect_emotions(classifier, texts)
            df = pd.DataFrame(results)
            st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.warning("Please enter some text before clicking *Analyze*.")

st.divider()
st.caption("Built with ❤️ using Streamlit and Hugging Face Transformers.")


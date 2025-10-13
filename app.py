import streamlit as st
import pandas as pd
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="🧠 Emotion Detector",
    page_icon="🧠",
    layout="wide",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* --- Overall layout --- */
    .main {
        background: #f0f4f8;
        padding: 2rem 3rem;
        font-family: 'Inter', sans-serif;
    }

    /* --- Titles --- */
    h1 {
        color: #1e293b;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    h2, h3 {
        color: #334155;
        font-weight: 700;
    }

    /* --- Text area --- */
    textarea {
        border-radius: 15px !important;
        border: 1px solid #cbd5e1 !important;
        padding: 12px !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-size: 16px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.2s ease-in-out;
    }

    textarea:focus {
        border: 2px solid #2563eb !important;
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
    }

    /* --- Buttons --- */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.6em 1.5em;
        transition: all 0.3s ease-in-out;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* --- DataFrame --- */
    .stDataFrame {
        border-radius: 12px !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }

    /* --- Footer --- */
    footer, .reportview-container .main footer {
        visibility: hidden;
    }

    /* --- Divider --- */
    hr {
        border: 0;
        height: 1px;
        background: #cbd5e1;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def initialize_classifier():
    """Load and cache the transformer model."""
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
Paste your sentences below and click **Analyze** to see the results!
""")

st.markdown("---")

# --- INPUT ---
st.subheader("📝 Step 1: Enter Text(s) to Analyze")

default_text = """I am so incredibly happy and proud of what we achieved today!
This is confusing; I need someone to clarify the instructions for step three.
My heart is racing, I'm genuinely terrified of what might happen next."""

input_text = st.text_area(
    "Enter one sentence per line:",
    value=default_text,
    height=200
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
            
            # Show table
            st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.warning("Please enter some text before clicking *Analyze*.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Hugging Face Transformers.")


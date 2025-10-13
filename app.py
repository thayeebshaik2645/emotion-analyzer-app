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

# --- CUSTOM CSS (DARK/NEON THEME - Font Overhaul Included) ---
st.markdown("""
    <style>
    /* ---------------------------------------------------- */
    /* 1. COLOR PALETTE DEFINITION (DARK/NEON) */
    :root {
        --primary-color: #00ffc8; /* Neon Cyan/Green Accent */
        --primary-dark: #00b38c;  /* Darker Neon */
        --background-dark: #121212; /* Very Dark Background */
        --surface-color: #1e1e1e; /* Card/Container Background */
        --text-color-light: #f0f0f0; /* Light Text */
        --text-color-secondary: #aaaaaa; /* Gray Text */
        --mono-font: 'Consolas', 'Courier New', monospace; /* Futuristic Monospace Font */
    }
    
    /* ---------------------------------------------------- */
    /* 2. OVERALL LAYOUT & BACKGROUND */
    .main {
        background: var(--background-dark);
        padding: 2.5rem 4rem; 
        font-family: 'Inter', sans-serif;
        color: var(--text-color-light); 
    }
    .stText, .stMarkdown {
        color: var(--text-color-light) !important;
    }

    /* 3. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--primary-color);
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        /* Neon Glow Effect */
        text-shadow: 0 0 5px var(--primary-color), 0 0 10px var(--primary-dark); 
        letter-spacing: 2px;
    }

    h2, h3 {
        color: var(--text-color-light);
        font-weight: 700;
        padding-left: 5px; 
        border-left: 5px solid var(--primary-color); 
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* 4. TEXT AREA EFFECTS (THE KEY CHANGE) */
    .stTextArea label {
        font-weight: 600;
        color: var(--primary-color);
        font-family: 'Inter', sans-serif; /* Keep label readable */
    }

    /* Target the text input element itself */
    textarea {
        border-radius: 10px !important;
        border: 2px solid var(--primary-dark) !important;
        padding: 15px !important;
        background-color: var(--surface-color) !important;
        color: var(--primary-color) !important; /* Neon text color inside box */
        font-size: 17px !important; /* Slightly larger for emphasis */
        font-family: var(--mono-font) !important; /* Monospace/Code-like font */
        line-height: 1.6; /* Increased line height for better reading */
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5); 
        transition: all 0.3s ease;
    }

    textarea:focus {
        border: 2px solid var(--primary-color) !important;
        box-shadow: 0 0 15px var(--primary-color) !important; 
    }

    /* 5. BUTTON EFFECTS (NEON GLOW) */
    div.stButton > button:first-child {
        background: var(--primary-dark);
        color: var(--background-dark);
        font-weight: 800;
        border-radius: 10px;
        padding: 0.8em 2em;
        transition: all 0.4s ease;
        border: none;
        box-shadow: 0 0 15px var(--primary-color); 
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px); 
        background: var(--primary-color); 
        box-shadow: 0 0 25px var(--primary-color), 0 0 5px var(--primary-color); 
    }
    
    div.stButton > button:first-child:active {
        transform: translateY(0px);
        box-shadow: 0 0 5px var(--primary-color);
    }

    /* 6. DATAFRAME EFFECTS & EMOTION COLORS */
    .stDataFrame {
        border-radius: 10px !important; 
        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.7); 
        border: 1px solid var(--primary-dark); 
        overflow: hidden;
        background-color: var(--surface-color); 
    }
    
    /* Dataframe content text font */
    .stDataFrame .data-row, .stDataFrame th, .stDataFrame td {
        color: var(--text-color-light) !important;
        font-family: var(--mono-font) !important; /* Use monospace font for results */
    }

    /* --- EMOTION SPECIFIC COLORING (Cyberpunk/Neon Tones) --- */
    [data-testid="stDataframe"] div:nth-child(3) > div:not(:first-child) { 
        font-weight: 800 !important;
        padding: 6px 10px !important;
        border-radius: 6px;
        text-align: center;
        margin: 4px 0;
        display: inline-block;
        min-width: 110px;
        transition: all 0.3s ease; 
        text-shadow: 0 0 2px black;
        font-family: 'Inter', sans-serif !important; /* Emotion label stands out */
    }

    /* 🔴 ANGER / DISGUST (Fiery Red/Pink) */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("ANGER"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("DISGUST") {
        background-color: #ff3366; 
        color: var(--background-dark); 
        box-shadow: 0 0 8px #ff3366;
    }
    
    /* 🟡 JOY / HAPPINESS / EXCITEMENT (Electric Yellow) */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("JOY"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("HAPPINESS"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("EXCITEMENT") {
        background-color: #fffb00; 
        color: var(--background-dark); 
        box-shadow: 0 0 8px #fffb00;
    }
    
    /* 🔵 SADNESS / LONELINESS (Deep Cyber Blue) */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("SADNESS"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("LONELINESS") {
        background-color: #00aaff; 
        color: var(--background-dark); 
        box-shadow: 0 0 8px #00aaff;
    }
    
    /* 🟣 FEAR / SURPRISE (Vibrant Purple) */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("FEAR"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("SURPRISE") {
        background-color: #ff00ff; 
        color: var(--background-dark); 
        box-shadow: 0 0 8px #ff00ff;
    }

    /* ⚪ NEUTRAL / OTHER (Primary Neon Color) */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("NEUTRAL") {
        background-color: var(--primary-color); 
        color: var(--background-dark); 
        box-shadow: 0 0 8px var(--primary-color);
    }

    /* 7. DIVIDER & FOOTER */
    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(to right, rgba(0,0,0,0), var(--primary-dark), rgba(0,0,0,0)); 
        margin: 3rem 0;
    }
    
    .st-emotion-detector-caption { 
        text-align: center;
        color: var(--text-color-secondary);
        font-style: italic;
        margin-top: 1rem;
        animation: fadeIn 3s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    footer, header {
        visibility: hidden !important;
    }

    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def initialize_classifier():
    """Load and cache the transformer model."""
    try:
        # Custom message style for loading
        st.markdown(f'<div style="color: {st.get_style().get("colors", {}).get("primary", "#00ffc8")}; font-family: var(--mono-font);">Initializing core systems... Please wait.</div>', unsafe_allow_html=True)
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
st.title("🧠 EMOTION DETECTOR FROM TEXT")
st.markdown(f'<p style="color: {st.get_style().get("colors", {}).get("secondary", "#aaaaaa")}; text-align: center; font-family: var(--mono-font);">Analyze sentiment in text with a futuristic, neon glow interface.</p>', unsafe_allow_html=True)

st.markdown("---")

# --- INPUT ---
st.subheader("📝 ENTER TEXT TO ANALYZE")

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
analyze = st.button("🔍 INITIATE ANALYSIS")

# Initialize classifier
classifier = initialize_classifier()

# --- RESULTS ---
if analyze:
    if texts:
        st.subheader("📊 ANALYSIS RESULTS")
        with st.spinner("Processing data... Stand by."):
            results = detect_emotions(classifier, texts)
            df = pd.DataFrame(results)
            
            # Show table
            st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.warning("Input required. Please provide text before initiating analysis.")

st.markdown("---")
# Custom footer for the fade-in effect
st.markdown('<p class="st-emotion-detector-caption">SYSTEM ONLINE | BUILD BY CSE-A</p>', unsafe_allow_html=True)

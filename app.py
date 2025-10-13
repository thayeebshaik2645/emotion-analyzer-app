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

# --- CUSTOM CSS (Enhanced with Colors, Effects, and Animations) ---
st.markdown("""
    <style>
    /* ---------------------------------------------------- */
    /* 1. COLOR PALETTE DEFINITION */
    /* Primary: A deep, professional blue */
    /* Secondary: A warm, modern gray */
    :root {
        --primary-color: #007bff; /* Bright Blue */
        --primary-dark: #0056b3;  /* Dark Blue */
        --secondary-color: #495057; /* Dark Gray */
        --background-light: #f4f6f9; /* Off-White Background */
        --border-color: #ced4da;
        --shadow-color: rgba(0, 0, 0, 0.08);
    }
    
    /* ---------------------------------------------------- */
    /* 2. OVERALL LAYOUT & BACKGROUND */
    .main {
        background: var(--background-light);
        padding: 2.5rem 4rem; /* Slightly more spacious padding */
        font-family: 'Inter', sans-serif;
        transition: background-color 0.5s ease; /* Subtle background transition */
    }

    /* 3. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--primary-dark);
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.05); /* Subtle text shadow */
    }

    h2, h3 {
        color: var(--secondary-color);
        font-weight: 700;
        padding-left: 5px; /* Indent for visual flow */
        border-left: 5px solid var(--primary-color); /* Primary color accent line */
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* 4. TEXT AREA EFFECTS */
    /* Target the text input element */
    .stTextArea label {
        font-weight: 600;
        color: var(--secondary-color);
    }

    textarea {
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        padding: 15px !important;
        background-color: #ffffff !important;
        color: #212529 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 10px var(--shadow-color);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); /* Smoother transition curve */
    }

    textarea:focus {
        border: 2px solid var(--primary-color) !important;
        box-shadow: 0 0 0 4px rgba(0, 123, 255, 0.25) !important; /* Prominent focus ring */
    }

    /* 5. BUTTON EFFECTS (THE BIG ONE) */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, var(--primary-color) 0%, #00aaff 100%);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.8em 2em; /* Slightly larger padding */
        transition: all 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55); /* Bouncy transition */
        border: none;
        box-shadow: 0 6px 15px rgba(0, 123, 255, 0.3); /* Blue shadow for the button */
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    div.stButton > button:first-child:hover {
        /* Scale and move slightly up on hover (bouncy effect) */
        transform: translateY(-4px) scale(1.02); 
        box-shadow: 0 10px 20px rgba(0, 123, 255, 0.4); 
        background: linear-gradient(135deg, var(--primary-dark) 0%, #0077cc 100%); /* Darker gradient on hover */
    }
    
    div.stButton > button:first-child:active {
        /* Push down slightly on click */
        transform: translateY(0px) scale(1.0);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    /* 6. DATAFRAME EFFECTS & EMOTION COLORS */
    .stDataFrame {
        border-radius: 16px !important; /* Slightly more rounded */
        box-shadow: 0px 10px 25px rgba(0,0,0,0.15); /* Deeper shadow for a floating effect */
        border: none; /* Remove default border, rely on shadow */
        overflow: hidden;
    }
    
    /* Apply a slight hover effect to the entire dataframe container */
    .stDataFrame:hover {
        box-shadow: 0px 12px 30px rgba(0,0,0,0.2);
        transform: translateY(-2px);
        transition: all 0.3s ease-out;
    }

    /* --- EMOTION SPECIFIC COLORING (CRITICAL FOR UI ENHANCEMENT) --- */
    /* Target the 'Dominant Emotion' column cells (Assuming it is the 3rd column) */
    /* NOTE: .css-1r6cnx6 is a common Streamlit class for dataframe cells */
    [data-testid="stDataframe"] div:nth-child(3) > div:not(:first-child) { 
        font-weight: 700 !important;
        padding: 6px 10px !important;
        border-radius: 8px;
        text-align: center;
        margin: 4px 0;
        display: inline-block;
        min-width: 110px;
        transition: background-color 0.4s ease; 
    }

    /* 🔴 ANGER / DISGUST */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("ANGER"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("DISGUST") {
        background-color: #ffe0e6; 
        color: #880011; 
        border: 1px solid #ff99aa;
    }
    
    /* 🟡 JOY / HAPPINESS / EXCITEMENT */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("JOY"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("HAPPINESS"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("EXCITEMENT") {
        background-color: #fff9e6; 
        color: #996600; 
        border: 1px solid #ffcc66;
    }
    
    /* 🔵 SADNESS / LONELINESS */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("SADNESS"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("LONELINESS") {
        background-color: #e6f7ff; 
        color: #005c8c; 
        border: 1px solid #99d6ff;
    }
    
    /* 🟣 FEAR / SURPRISE */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("FEAR"),
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("SURPRISE") {
        background-color: #f2e6ff; 
        color: #590099; 
        border: 1px solid #cc99ff;
    }

    /* ⚪ NEUTRAL / OTHER */
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("NEUTRAL") {
        background-color: #e9ecef; 
        color: #495057; 
        border: 1px solid #c9c9c9;
    }

    /* 7. DIVIDER & FOOTER */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(0,0,0,0), var(--border-color), rgba(0,0,0,0)); /* Gradient divider */
        margin: 3rem 0;
    }
    
    .st-emotion-detector-caption { 
        text-align: center;
        color: #6c757d;
        font-style: italic;
        margin-top: 1rem;
        animation: fadeIn 2s;
    }
    
    /* Keyframe for a soft fade-in effect */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    /* Hide Streamlit default footer/header for a cleaner look */
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
st.title("🧠 Emotion Detector From Text")
st.markdown("""
Detect emotions in text using a fine-tuned Transformer model.
Enter your sentences below and click **Analyze** to see the results!
""")

st.markdown("---")

# --- INPUT ---
st.subheader("📝Enter Text to Analyze")

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

# Initialize classifier outside the conditional block to ensure it's ready
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
# Use custom markdown for the footer to apply the CSS class
st.markdown('<p class="st-emotion-detector-caption">BUILD BY CSE-A</p>', unsafe_allow_html=True)

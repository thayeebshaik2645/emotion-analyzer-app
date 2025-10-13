import streamlit as st
import pandas as pd
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- EMOTION GIF MAPPING ---
# NOTE: Using placeholder URLs. Replace these with your actual GIF links!
EMOTION_GIFS = {
    "ANGER": "https://i.giphy.com/media/l4pTsh45DG7rivt2e/giphy.gif",          # Red/Pink GIF
    "DISGUST": "https://i.giphy.com/media/26hirEBLl0l0Qk2s8/giphy.gif",       # Red/Pink GIF
    "JOY": "https://i.giphy.com/media/l4FGp6wB6vR22yO64/giphy.gif",           # Neon Yellow GIF
    "HAPPINESS": "https://i.giphy.com/media/l4FGp6wB6vR22yO64/giphy.gif",     # Neon Yellow GIF
    "EXCITEMENT": "https://i.giphy.com/media/3o7TKr6wQjG123uRkk/giphy.gif",    # Neon Yellow GIF
    "SADNESS": "https://i.giphy.com/media/9Y50g67K1LwY/giphy.gif",            # Blue GIF
    "LONELINESS": "https://i.giphy.com/media/3o6UBd3M910D6F1Jle/giphy.gif",    # Blue GIF
    "FEAR": "https://i.giphy.com/media/26tk05W9Y83y2h4wE/giphy.gif",          # Magenta/Purple GIF
    "SURPRISE": "https://i.giphy.com/media/l4FGp2K4t9mX9sJiw/giphy.gif",      # Magenta/Purple GIF
    "NEUTRAL": "https://i.giphy.com/media/3ohzdM722421LqV1e8/giphy.gif",       # Primary Cyan GIF
}


# --- PAGE SETUP ---
st.set_page_config(
    page_title="🧠 Emotion Detector From Text",
    page_icon="https://p7.hiclipart.com/preview/573/335/801/stock-photography-robot-royalty-free-robots.jpg",
    layout="wide",
)

# --- CUSTOM CSS (DARK/NEON THEME & Font Overhaul) ---
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
    
    /* --- EMOTION SPECIFIC COLOR MAP --- */
    .emotion-anger, .emotion-disgust { --emotion-color: #ff3366; } 
    .emotion-joy, .emotion-happiness, .emotion-excitement { --emotion-color: #fffb00; } 
    .emotion-sadness, .emotion-loneliness { --emotion-color: #00aaff; } 
    .emotion-fear, .emotion-surprise { --emotion-color: #ff00ff; } 
    .emotion-neutral { --emotion-color: var(--primary-color); } 
    
    /* 2. OVERALL LAYOUT & BACKGROUND */
    .main {
        background: var(--background-dark);
        padding: 2.5rem 4rem; 
        font-family: 'Inter', sans-serif;
        color: var(--text-color-light); 
    }
    .stApp .st-emotion-cache-1pxn4ip, .stApp .st-emotion-cache-1v0pmnt {
        background-color: var(--surface-color);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.4);
    }

    /* 3. TITLES & TEXT EFFECTS (Retained) */
    h1 {
        color: var(--primary-color);
        font-weight: 900;
        text-align: center;
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

    /* 4. TEXT AREA EFFECTS (Retained) */
    textarea {
        border-radius: 10px !important;
        border: 2px solid var(--primary-dark) !important;
        padding: 15px !important;
        background-color: var(--surface-color) !important;
        color: var(--primary-color) !important;
        font-size: 17px !important; 
        font-family: var(--mono-font) !important; 
        line-height: 1.6; 
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5); 
    }

    /* 5. BUTTON EFFECTS (Retained) */
    div.stButton > button:first-child {
        background: var(--primary-dark);
        color: var(--background-dark);
        font-weight: 800;
        border-radius: 10px;
        padding: 0.8em 2em;
        box-shadow: 0 0 15px var(--primary-color); 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px); 
        background: var(--primary-color); 
        box-shadow: 0 0 25px var(--primary-color), 0 0 5px var(--primary-color); 
    }
    
    /* 6. CUSTOM RESULT CARDS (Updated for GIF) */
    .result-card {
        background-color: var(--surface-color);
        border: 2px solid var(--emotion-color, var(--primary-dark)); 
        border-left: 10px solid var(--emotion-color, var(--primary-color)); 
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5), 0 0 5px var(--emotion-color, rgba(0, 0, 0, 0));
        transition: all 0.3s ease;
    }
    
    .card-header-line { /* New class to align emotion tag and GIF */
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .emotion-gif { /* Styling for the GIF */
        width: 50px; 
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid var(--emotion-color);
        box-shadow: 0 0 10px var(--emotion-color);
    }

    .result-text {
        color: var(--text-color-light);
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        margin-bottom: 10px;
        font-style: italic;
    }

    .result-emotion {
        display: inline-block;
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--background-dark);
        background-color: var(--emotion-color);
        padding: 5px 12px;
        border-radius: 6px;
        text-transform: uppercase;
        box-shadow: 0 0 5px var(--emotion-color);
    }
    
    .result-confidence {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-color-secondary);
        font-family: var(--mono-font);
    }

    /* 7. DIVIDER & FOOTER (Retained) */
    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(to right, rgba(0,0,0,0), var(--primary-dark), rgba(0,0,0,0)); 
        margin: 3rem 0;
    }
    footer, header { visibility: hidden !important; }

    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def initialize_classifier():
    """Load and cache the transformer model."""
    try:
        # Custom message style for loading
        st.markdown(f'<div style="color: var(--primary-color); font-family: var(--mono-font);">SYSTEM STATUS: Initializing core systems... Please wait.</div>', unsafe_allow_html=True)
        classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            return_all_scores=True
        )
        st.success("✅ SYSTEM STATUS: Model loaded successfully!")
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

# =================================================================
# --- FUTURISTIC UI LAYOUT ---
# =================================================================

st.title("🧠 EMOTION DETECTOR FROM TEXT")
st.markdown(f'<p style="color: var(--text-color-secondary); text-align: center; font-family: var(--mono-font);">DETEC YOUR EMOTION FROM TEXT</p>', unsafe_allow_html=True)

st.markdown("---")

# 1. INPUT BLOCK (Terminal style)
input_container = st.container()
with input_container:
    st.subheader("📝 ENTER THE INPUT")
    
    col1, col_input, col2 = st.columns([1, 4, 1])

    default_text = """I am so incredibly happy and proud of what we achieved today!
This is confusing; I need someone to clarify the instructions for step three.
My heart is racing, I'm genuinely terrified of what might happen next."""
    
    with col_input:
        input_text = st.text_area(
            "Input Log - Enter one sentence per line:",
            value=default_text,
            height=200,
            key="input_text_area"
        )
        texts = [t.strip() for t in input_text.split("\n") if t.strip()]
        
        col_btn_l, col_btn, col_btn_r = st.columns([1.5, 2, 1.5])
        with col_btn:
             analyze = st.button("🔍 INITIATE ANALYSIS", use_container_width=True)

# Initialize classifier
classifier = initialize_classifier()

st.markdown("---")

# 2. RESULTS BLOCK (Custom Data Blocks/Cards with GIFs)
if analyze:
    if texts:
        results_container = st.container()
        with results_container:
            st.subheader("📊 DETAILED ANALYSIS LOG")
            
            # Added two columns for better layout density
            cols = st.columns(2)
            
            with st.spinner("Processing data... Stand by."):
                results = detect_emotions(classifier, texts)
                
                # --- NEW OUTPUT PATTERN: CUSTOM CARDS WITH GIFS ---
                for i, result in enumerate(results):
                    emotion = result['Dominant Emotion']
                    confidence = result['Confidence']
                    input_text = result['Input Text']
                    
                    # 1. Get the GIF URL
                    gif_url = EMOTION_GIFS.get(emotion, "https://i.giphy.com/media/l0HlC9N40qS03S48w/giphy.gif") # Fallback GIF
                    
                    # 2. Determine CSS class for coloring (lowercase emotion)
                    css_class = ""
                    if emotion in ["ANGER", "DISGUST"]: css_class = "emotion-anger"
                    elif emotion in ["JOY", "HAPPINESS", "EXCITEMENT"]: css_class = "emotion-joy"
                    elif emotion in ["SADNESS", "LONELINESS"]: css_class = "emotion-sadness"
                    elif emotion in ["FEAR", "SURPRISE"]: css_class = "emotion-fear"
                    elif emotion == "NEUTRAL": css_class = "emotion-neutral"
                    
                    # Display card in the column
                    with cols[i % 2]: # Alternates between column 0 and 1
                        st.markdown(f"""
                            <div class="result-card {css_class}">
                                <div class="card-header-line">
                                    <span class="result-emotion">{emotion}</span>
                                    <img src="{gif_url}" class="emotion-gif" alt="{emotion} GIF">
                                </div>
                                <div class="result-text">"{input_text}"</div>
                                <div style="display: flex; align-items: center; justify-content: flex-end;">
                                    <div class="result-confidence">
                                        CONFIDENCE: {confidence}
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                # --- END NEW OUTPUT PATTERN ---
                
    else:
        st.warning("Input required. Please provide text before initiating analysis.")

# 3. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption"> BUILD BY CSE-A</p>', unsafe_allow_html=True)

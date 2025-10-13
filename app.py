import streamlit as st
import pandas as pd
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
# Removed emojis from page_icon and all markdown text

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Emotion Detector From Text",
    # Using a simple icon or removing it for cleaner look
    page_icon="🤖", 
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
    /* Define colors for the result blocks */
    .emotion-anger, .emotion-disgust { --emotion-color: #ff3366; } /* Red/Pink */
    .emotion-joy, .emotion-happiness, .emotion-excitement { --emotion-color: #fffb00; } /* Neon Yellow */
    .emotion-sadness, .emotion-loneliness { --emotion-color: #00aaff; } /* Blue */
    .emotion-fear, .emotion-surprise { --emotion-color: #ff00ff; } /* Magenta/Purple */
    .emotion-neutral { --emotion-color: var(--primary-color); } /* Primary Cyan */
    
    /* 2. OVERALL LAYOUT & BACKGROUND */
    .main {
        background: var(--background-dark);
        padding: 2.5rem 4rem; 
        font-family: 'Inter', sans-serif;
        color: var(--text-color-light); 
    }
    .stText, .stMarkdown, .stAlert, .stApp {
        color: var(--text-color-light) !important;
    }
    
    /* Apply surface color to primary Streamlit containers for "block" look */
    .stApp .st-emotion-cache-1pxn4ip, .stApp .st-emotion-cache-1v0pmnt {
        background-color: var(--surface-color);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.4);
    }

    /* 3. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--primary-color);
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
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

    /* System Status Messages */
    .stSuccess {
        background-color: transparent !important;
        border: 1px solid var(--primary-dark) !important;
        color: var(--primary-color) !important;
    }
    .stError {
        background-color: transparent !important;
        border: 1px solid #ff3366 !important;
        color: #ff3366 !important;
    }
    .stWarning {
        background-color: transparent !important;
        border: 1px solid #fffb00 !important;
        color: #fffb00 !important;
    }

    /* 4. TEXT AREA EFFECTS */
    .stTextArea label {
        font-weight: 600;
        color: var(--primary-color);
        font-family: 'Inter', sans-serif;
    }

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
    
    /* 6. CUSTOM RESULT CARDS (DATA BLOCKS) */
    .result-card {
        background-color: var(--surface-color);
        border: 2px solid var(--emotion-color, var(--primary-dark)); /* Use emotion color for border */
        border-left: 10px solid var(--emotion-color, var(--primary-color)); /* Strong left border */
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5), 0 0 5px var(--emotion-color, rgba(0, 0, 0, 0)); /* subtle glow */
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.8), 0 0 8px var(--emotion-color); /* Stronger hover glow */
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
        margin-right: 15px;
        text-transform: uppercase;
        box-shadow: 0 0 5px var(--emotion-color);
    }
    
    .result-confidence {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-color-secondary);
        font-family: var(--mono-font);
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
        # Removed emoji, enhanced status message style
        st.markdown(f'<div style="color: var(--primary-color); font-family: var(--mono-font);">SYSTEM STATUS: Initializing core systems... Please wait.</div>', unsafe_allow_html=True)
        classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            return_all_scores=True
        )
        # Removed emoji from success message
        st.success("SYSTEM STATUS: Model loaded successfully!")
        return classifier
    except Exception as e:
        # Removed emoji from error message
        st.error(f"SYSTEM FAILURE: Error loading model: {e}")
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

st.title("EMOTION DETECTOR FROM TEXT")
st.markdown(f'<p style="color: var(--text-color-secondary); text-align: center; font-family: var(--mono-font);">DEEP LEARNING TEXT ANALYSIS INTERFACE</p>', unsafe_allow_html=True)

st.markdown("---")

# 1. INPUT BLOCK (Terminal style)
with st.container():
    st.subheader("SYSTEM INPUT INTERFACE")
    
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
             # Removed emoji from button text
             analyze = st.button("INITIATE ANALYSIS", use_container_width=True)

# Initialize classifier
classifier = initialize_classifier()

st.markdown("---")

# 2. RESULTS BLOCK (Custom Data Blocks/Cards)
if analyze:
    if texts:
        with st.container():
            st.subheader("ANALYSIS OUTPUT: EMOTION LOG")
            
            with st.spinner("Processing data... Stand by. Executing high-speed classification."):
                results = detect_emotions(classifier, texts)
                
                # Use st.columns to create a two-column grid for the cards for better density
                cols = st.columns(2)
                
                for i, result in enumerate(results):
                    emotion = result['Dominant Emotion'].lower()
                    confidence = result['Confidence']
                    input_text = result['Input Text']
                    
                    # Determine CSS class for coloring
                    css_class = ""
                    if emotion in ["anger", "disgust"]: css_class = "emotion-anger"
                    elif emotion in ["joy", "happiness", "excitement"]: css_class = "emotion-joy"
                    elif emotion in ["sadness", "loneliness"]: css_class = "emotion-sadness"
                    elif emotion in ["fear", "surprise"]: css_class = "emotion-fear"
                    elif emotion == "neutral": css_class = "emotion-neutral"
                    
                    # Display card in the column
                    with cols[i % 2]: # Alternates between column 0 and 1
                        st.markdown(f"""
                            <div class="result-card {css_class}">
                                <div class="result-text">"{input_text}"</div>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <div>
                                        <span class="result-emotion">{result['Dominant Emotion']}</span>
                                    </div>
                                    <div class="result-confidence">
                                        SCORE: {confidence}
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
    else:
        # Removed emoji from warning message
        st.warning("SYSTEM ALERT: Input required. Please provide text before initiating analysis.")

# 3. FOOTER
st.markdown("---")
# Cleaned up footer text
st.markdown('<p class="st-emotion-detector-caption">ENGINEERING BY CSE-A | DEPLOYED SYSTEM v1.2</p>', unsafe_allow_html=True)

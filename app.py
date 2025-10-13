import streamlit as st
import pandas as pd
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="🧠 Emotion Detector From Text",
    page_icon="https://p7.hiclipart.com/preview/573/335/801/stock-photography-robot-royalty-free-robots.jpg",  # Changed icon for futuristic feel
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
    
    /* Info Box styling (for the initial info message) */
    .stAlert {
        background-color: var(--surface-color) !important;
        border: 1px solid var(--primary-dark) !important;
        color: var(--text-color-light) !important;
        border-radius: 8px;
        box-shadow: 0 0 5px var(--primary-dark);
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
    
    /* 6. DATAFRAME STYLING (Kept for the st.info box's internal styling) */
    .stDataFrame {
        border-radius: 10px !important; 
        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.7); 
        border: 1px solid var(--primary-dark); 
        overflow: hidden;
        background-color: var(--surface-color); 
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
    
    # Use columns to center the text area and button
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
        
        # Center the button using columns
        col_btn_l, col_btn, col_btn_r = st.columns([1.5, 2, 1.5])
        with col_btn:
             analyze = st.button("🔍 INITIATE ANALYSIS", use_container_width=True)

# Initialize classifier
classifier = initialize_classifier()

st.markdown("---")

# 2. RESULTS BLOCK (Conditional display - SIMPLIFIED)
if analyze:
    if texts:
        results_container = st.container()
        with results_container:
            st.subheader("📊 ANALYSIS SUMMARY")
            
            # Use columns to keep the summary centered/structured
            col_left, col_summary, col_right = st.columns([1, 2, 1])
            
            with st.spinner("Processing data... Stand by."):
                results = detect_emotions(classifier, texts)
                df = pd.DataFrame(results)
                
                # --- Dominant Emotion Summary ---
                dominant_emotion = df['Dominant Emotion'].mode()[0]
                total_sentences = len(df)
                
                with col_summary:
                    st.info(f"""
                        **SYSTEM SUMMARY**
                        - **Total Entries:** **{total_sentences}**
                        - **Most Frequent Emotion:** **{dominant_emotion}**
                        - **Raw Output Data Log Removed per protocol request.**
                        """)
                    
    else:
        st.warning("Input required. Please provide text before initiating analysis.")

# 3. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption"> BUILD BY CSE-A</p>', unsafe_allow_html=True)

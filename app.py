import streamlit as st
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- EMOTION GIF MAPPING ---
# REPLACE THESE PLACEHOLDER URLS WITH YOUR ACTUAL GIF LINKS
EMOTION_GIFS = {
    "ANGER": "https://i.giphy.com/media/l4pTsh45DG7rivt2e/giphy.gif",          # Angry/Red GIF
    "DISGUST": "https://i.giphy.com/media/26hirEBLl0l0Qk2s8/giphy.gif",       # Disgust/Green GIF
    "JOY": "https://i.giphy.com/media/l4FGp6wB6vR22yO64/giphy.gif",           # Joy/Happy/Yellow GIF
    "HAPPINESS": "https://i.giphy.com/media/l4FGp6wB6vR22yO64/giphy.gif",     # Same as JOY
    "EXCITEMENT": "https://i.giphy.com/media/3o7TKr6wQjG123uRkk/giphy.gif",    # Excitement/Animated Yellow
    "SADNESS": "https://i.giphy.com/media/9Y50g67K1LwY/giphy.gif",            # Sad/Blue GIF
    "LONELINESS": "https://i.giphy.com/media/3o6UBd3M910D6F1Jle/giphy.gif",    # Loneliness/Blue GIF
    "FEAR": "https://i.giphy.com/media/26tk05W9Y83y2h4wE/giphy.gif",          # Fear/Magenta GIF
    "SURPRISE": "https://i.giphy.com/media/l4FGp2K4t9mX9sJiw/giphy.gif",      # Surprise/Magenta GIF
    "NEUTRAL": "https://i.giphy.com/media/3ohzdM722421LqV1e8/giphy.gif",       # Neutral/Cyan GIF
}


# --- PAGE SETUP ---
st.set_page_config(
    page_title="Emotion Detector From Text",
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS (DARK/NEON THEME & Font Overhaul) ---
st.markdown("""
    <style>
    /* ---------------------------------------------------- */
    /* 1. COLOR PALETTE DEFINITION (DARK/NEON) */
    :root {
        --primary-color: #00ffc8; 
        --primary-dark: #00b38c;  
        --background-dark: #121212; 
        --surface-color: #1e1e1e; 
        --text-color-light: #f0f0f0; 
        --text-color-secondary: #aaaaaa; 
        --mono-font: 'Consolas', 'Courier New', monospace; 
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
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: var(--surface-color) !important;
        border-right: 2px solid var(--primary-dark);
    }

    /* 3. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--primary-color);
        font-weight: 900;
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
    
    /* 4. TEXT AREA & INPUTS */
    textarea {
        border-radius: 10px !important;
        border: 2px solid var(--primary-dark) !important;
        padding: 15px !important;
        background-color: var(--surface-color) !important;
        color: var(--primary-color) !important;
        font-size: 17px !important; 
        font-family: var(--mono-font) !important; 
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
    
    /* 6. CUSTOM RESULT CARDS WITH GIFS */
    .result-card {
        background-color: var(--surface-color);
        border: 2px solid var(--emotion-color, var(--primary-dark)); 
        border-left: 10px solid var(--emotion-color, var(--primary-color)); 
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5), 0 0 5px var(--emotion-color, rgba(0, 0, 0, 0));
    }
    
    .card-header-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .emotion-gif {
        width: 50px; /* Standard size for the GIF */
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid var(--emotion-color);
        box-shadow: 0 0 10px var(--emotion-color);
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
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING & DATA PROCESSING (Retained) ---
status_placeholder = st.empty()

@st.cache_resource
def initialize_classifier():
    """Load and cache the transformer model."""
    try:
        status_placeholder.markdown(f'<div style="color: var(--primary-color); font-family: var(--mono-font); margin-bottom: 15px; border-left: 3px solid var(--primary-color); padding-left: 10px;">SYSTEM STATUS: Initializing core systems... Please wait.</div>', unsafe_allow_html=True)
        
        classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            return_all_scores=True
        )
        status_placeholder.success("SYSTEM STATUS: Model loaded successfully!")
        return classifier
    except Exception as e:
        status_placeholder.error(f"SYSTEM FAILURE: Error loading model. Check console for details. {e}")
        st.stop()

def detect_emotions(classifier, texts):
    """Processes texts and returns analysis results."""
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
# --- UI STRUCTURE ---
# =================================================================

st.title("EMOTION DETECTOR FROM TEXT")
st.markdown(f'<p style="color: var(--text-color-secondary); text-align: center; font-family: var(--mono-font);">DEEP LEARNING TEXT ANALYSIS INTERFACE</p>', unsafe_allow_html=True)

st.markdown("---")

# 2. SIDEBAR (Retained)
with st.sidebar:
    st.image("https://p7.hiclipart.com/preview/573/335/801/stock-photography-robot-royalty-free-robots.jpg", use_column_width=True)
    st.subheader("SYSTEM INFO")
    st.markdown(f"""
    <div style="font-family: var(--mono-font); color: var(--text-color-secondary);">
        MODEL: **{MODEL_NAME}**<br>
        TASK: **Text Classification**<br>
        <hr style='margin: 10px 0;'>
    </div>
    """, unsafe_allow_html=True)

# 3. CORE LOGIC: INPUT, PROCESSING, OUTPUT
input_tab, results_tab = st.tabs(["SYSTEM INPUT", "ANALYSIS LOG"])

# --- INPUT TAB ---
with input_tab:
    st.subheader("ENTER TEXT FOR ANALYSIS")
    
    default_text = """I am so incredibly happy and proud of what we achieved today!
This is confusing; I need someone to clarify the instructions for step three.
My heart is racing, I'm genuinely terrified of what might happen next.
Everything seems normal, just another day at the office."""
    
    input_text = st.text_area(
        "Input Console - Enter one sentence per line:",
        value=default_text,
        height=250,
        key="input_text_area"
    )
    texts = [t.strip() for t in input_text.split("\n") if t.strip()]
    
    st.markdown("---")
    
    col_btn_l, col_btn, col_btn_r = st.columns([1.5, 2, 1.5])
    with col_btn:
         analyze = st.button("INITIATE ANALYSIS", use_container_width=True)

classifier = initialize_classifier()

# --- RESULTS TAB ---
if analyze:
    # Programmatically switch to the results tab upon button click
    st.session_state["active_tab"] = "ANALYSIS LOG" 
    
    with results_tab:
        if texts:
            st.subheader("ANALYSIS OUTPUT: EMOTION LOG")
            
            with st.spinner("Processing data... Executing high-speed classification."):
                results = detect_emotions(classifier, texts)
                
                cols = st.columns(2)
                
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
                    with cols[i % 2]:
                        st.markdown(f"""
                            <div class="result-card {css_class}">
                                <div class="card-header-line">
                                    <span class="result-emotion">{emotion}</span>
                                    <img src="{gif_url}" class="emotion-gif" alt="{emotion} GIF">
                                </div>
                                <div style="color: var(--text-color-light); font-style: italic; margin-bottom: 10px;">"{input_text}"</div>
                                <div style="display: flex; align-items: center; justify-content: flex-end;">
                                    <div class="result-confidence">
                                        SCORE: {confidence}
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("SYSTEM ALERT: Input required. Please enter text in the 'SYSTEM INPUT' tab before initiating analysis.")

# 4. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption">ENGINEERING BY CSE-A | DEPLOYED SYSTEM v1.4</p>', unsafe_allow_html=True)

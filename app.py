import streamlit as st
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
PAGE_ICON_URL = "https://cdn-icons-png.flaticon.com/128/10479/10479785.png"

# --- EMOTION GIF MAPPING (UNMODIFIED) ---
EMOTION_GIFS = {
    # User-provided URLs
    "ANGER": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExd29sb29oMWNzYjk4ZnRlMTlkenBmNTd1dmZjemVjaGI0Z21oYXRvZSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/jsNiI5nMGQurggwpkN/giphy.webp",
    "HAPPINESS": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDV5djV3enNnd3kzcXAxdzFydjYxOWN4aWRwOTMzbW50aHN1azQzcyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/USR9bpLz899PYVHk7C/giphy.webp",
    "SADNESS": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXlmdzVmNXdhemlwY2F4N2ZoZnJzaDM3bnBsOTk3ejkwZmtzZ2JtMCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/StAnQV9TUCuys/giphy.webp",
    "JOY": "https://media0.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3MmZnc3Q0c2FoMXR6aDZtc2xxZG45dHhtbHY1Mms3bTFxbnk2eWJoeSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/LN5bH1r7UEpSRbcN7M/giphy.webp",
    "FEAR": "https://media0.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3eTZ6YmNrZHV3OHd1OHhmcmlqOXVzMHJua2JjNXpwYWkxeTJ3bWp6bCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Gl7mfimOjkkGl5mMDS/giphy.webp",
    "NEUTRAL": "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMzRuY2I5dGM0ZnE4aG04NzZiMnY5aW4wdHZ1MXdpN3djeG1hd2t0byZlcD12MV9naWZzX3NlYXJjaCZjdD1n/7CXIO53h5YciXOp505/giphy.webp",
    "DISGUST": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExd29sb29oMWNzYjk4ZnRlMTlkenBmNTd1dmZjemVjaGI0Z21oYXRvZSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/jsNiI5nMGQurggwpkN/giphy.webp",
    "EXCITEMENT": "https://media0.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3MmZnc3Q0c2FoMXR6aDZtc2xxZG45dHhtbHY1Mms3bTFxbnk2eWJoeSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/LN5bH1r7UEpSRbcN7M/giphy.webp",
    "LONELINESS": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXlmdzVmNXdhemlwY2F4N2ZoZnJzaDM3bnBsOTk3ejkwZmtzZ2JtMCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/StAnQV9TUCuys/giphy.webp",
    "SURPRISE": "https://media0.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3eTZ6YmNrZHV3OHd1OHhmcmlqOXVzMHJua2JjNXpwYWkxeTJ3bWp6bCZlcD12MV9naWZzX3NlYXJjaCZjdT1n/Gl7mfimOjkkGl5mMDS/giphy.webp",
}

# --- PAGE SETUP ---
st.set_page_config(
    page_title="🧠 Emotion Detector From Text",
    page_icon=PAGE_ICON_URL,
    layout="wide",
)

# --- CUSTOM CSS (Professional Dark Mode) ---
st.markdown("""
    <style>
    /* ---------------------------------------------------- */
    /* 1. FONT IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700;900&family=Roboto+Mono:wght@400;600&display=swap');
    
    /* 2. COLOR PALETTE DEFINITION (PROFESSIONAL DARK MODE) */
    :root {
        --primary-color: #007bff; /* Vibrant Blue Accent */
        --primary-light: #52a8ff; 
        --background-dark: #1e1e2f; /* Deep Navy Background */
        --surface-color: #2b2b40; /* Slightly lighter Card Surface */
        --text-color-light: #e0e0e0; 
        --text-color-secondary: #aaaaaa; 
        
        --main-font: 'Source Sans 3', sans-serif; 
        --mono-font: 'Roboto Mono', monospace; 
    }
    
    /* --- EMOTION SPECIFIC COLOR MAP (Color-Coded, but less aggressive) --- */
    .emotion-anger, .emotion-disgust { --emotion-color: #dc3545; } /* Red */
    .emotion-joy, .emotion-happiness, .emotion-excitement { --emotion-color: #28a745; } /* Green */
    .emotion-sadness, .emotion-loneliness { --emotion-color: #007bff; } /* Blue (Matches primary) */
    .emotion-fear, .emotion-surprise { --emotion-color: #ffc107; } /* Yellow/Orange */
    .emotion-neutral { --emotion-color: #6c757d; } /* Gray */

    
    /* 3. OVERALL LAYOUT & BACKGROUND */
    .main {
        background: var(--background-dark);
        padding: 3rem 4rem; 
        font-family: var(--main-font); 
        color: var(--text-color-light); 
    }
    .stApp .st-emotion-cache-1pxn4ip, .stApp .st-emotion-cache-1v0pmnt {
        background-color: var(--surface-color);
        border-radius: 8px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4); 
    }

    /* 4. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--primary-color);
        font-weight: 900;
        font-family: var(--main-font);
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    /* --- SECTION SUBHEADER (h3) --- */
    h3 {
        color: var(--text-color-light);
        font-family: var(--main-font); 
        font-weight: 700; 
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 5px;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
    }
    /* Subtitle/model name */
    p[data-testid="stMarkdownContainer"] {
        color: var(--text-color-secondary) !important;
        text-align: center;
        font-family: var(--mono-font);
    }

    /* 5. TEXT AREA EFFECTS */
    textarea {
        border-radius: 8px !important;
        border: 1px solid var(--primary-color) !important;
        background-color: #121220 !important; /* Very dark input field */
        color: var(--text-color-light) !important;
        font-size: 16px !important; 
        font-family: var(--mono-font) !important; 
        box-shadow: 0 0 5px rgba(0, 123, 255, 0.2); 
    }

    /* 6. BUTTON EFFECTS */
    div.stButton > button:first-child {
        background: var(--primary-color);
        color: white;
        font-weight: 700;
        font-family: var(--main-font);
        border-radius: 8px;
        padding: 0.6em 1.5em;
        box-shadow: 0 4px 10px rgba(0, 123, 255, 0.3); 
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        border: none;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px); 
        background: var(--primary-light); 
        box-shadow: 0 6px 15px rgba(0, 123, 255, 0.5); 
    }
    
    /* 7. CUSTOM RESULT CARDS (Original Structure, New Professional Styles) */
    .result-card {
        background-color: #1e1e2f; /* Matches background */
        border: 1px solid #3c3c54;
        border-left: 5px solid var(--emotion-color, var(--primary-color)); 
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .card-header-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .emotion-gif {
        width: 40px; 
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid var(--emotion-color);
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
    }

    .result-emotion {
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--text-color-light); 
        background-color: var(--emotion-color);
        padding: 4px 10px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    
    .result-text {
        color: var(--text-color-light);
        font-family: var(--main-font); 
        font-size: 1rem;
        margin-bottom: 10px;
        font-style: italic;
    }

    .result-confidence {
        font-size: 0.8rem;
        font-weight: 400;
        color: var(--text-color-secondary);
        font-family: var(--mono-font); 
    }

    /* 8. DIVIDER & FOOTER */
    hr {
        border: 0;
        height: 1px;
        background: #444466; 
        margin: 2rem 0;
    }
    .st-emotion-detector-caption {
        text-align: center;
        color: var(--text-color-secondary);
        font-family: var(--mono-font);
        font-size: 0.8rem;
        padding-top: 10px;
    }
    footer, header { visibility: hidden !important; }

    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (Functionality unchanged) ---
@st.cache_resource
def initialize_classifier():
    """Load and cache the transformer model silently."""
    try:
        classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            return_all_scores=True
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def detect_emotions(classifier, texts):
    if not texts:
        return []

    valid_texts = [t for t in texts if t]
    if not valid_texts:
         return []
         
    predictions = classifier(valid_texts)
    
    results = []
    for text, prediction_list in zip(valid_texts, predictions):
        best = max(prediction_list, key=lambda x: x['score'])
        results.append({
            "Input Text": text,
            "Dominant Emotion": best['label'].upper(),
            "Confidence": f"{best['score']:.4f}"
        })
    return results

# =================================================================
# --- APP LAYOUT (PROFESSIONAL DARK MODE) ---
# =================================================================

# MAIN TITLE
st.title("Sentiment & Emotion Analysis Tool 🤖")
st.markdown(f'<p>Powered by Hugging Face: **{MODEL_NAME.split("/")[-1]}**</p>', unsafe_allow_html=True)

st.markdown("---")

# 1. INPUT BLOCK 
with st.container():
    
    st.subheader("1. Enter Text Samples")
    
    # --- TEXT AREA ---
    default_text = """I am so incredibly happy and proud of what we achieved today!
This is confusing; I need someone to clarify the instructions for step three.
My heart is racing, I'm genuinely terrified of what might happen next."""
    
    input_text = st.text_area(
        "Enter sentences separated by new lines:", 
        value=default_text,
        height=200,
        key="input_text_area",
        label_visibility="collapsed"
    )
    
    texts = [t.strip() for t in input_text.split("\n") if t.strip()]
    
    # Center the button
    col_btn_l, col_btn, col_btn_r = st.columns([1.5, 2, 1.5])
    with col_btn:
        analyze = st.button("Run Sentiment Analysis", use_container_width=True)

# Load the model silently
classifier = initialize_classifier()

st.markdown("---")

# 2. RESULTS BLOCK (Original two-column card structure maintained)
if analyze:
    if texts:
        results_container = st.container()
        with results_container:
            st.subheader("2. Analysis Results")
            
            # Use two columns to display results cards (Original UI structure)
            cols = st.columns(2)
            
            with st.spinner("Processing text..."):
                results = detect_emotions(classifier, texts)
                
                if not results:
                    st.warning("No valid text lines found for analysis.")
                
                # Display results in alternating columns
                for i, result in enumerate(results):
                    emotion = result['Dominant Emotion']
                    confidence = result['Confidence']
                    input_text = result['Input Text']
                    
                    # 1. Get the GIF URL
                    gif_url = EMOTION_GIFS.get(emotion, EMOTION_GIFS["NEUTRAL"]) 
                    
                    # 2. Determine CSS class for coloring
                    css_class = ""
                    if emotion in ["ANGER", "DISGUST"]: css_class = "emotion-anger"
                    elif emotion in ["JOY", "HAPPINESS", "EXCITEMENT"]: css_class = "emotion-joy"
                    elif emotion in ["SADNESS", "LONELINESS"]: css_class = "emotion-sadness"
                    elif emotion in ["FEAR", "SURPRISE"]: css_class = "emotion-fear"
                    elif emotion == "NEUTRAL": css_class = "emotion-neutral"
                    
                    # Display card in the current column (i % 2 gives 0 or 1)
                    with cols[i % 2]: 
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
                        
    else:
        st.warning("Please enter text into the box and click 'Run Sentiment Analysis'.")

# 3. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption">Developed by THAYEEB | Data Analysis Tool</p>', unsafe_allow_html=True)

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
    "SURPRISE": "https://media0.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3eTZ6YmNrZHV3OHd1OHhmcmlqOXVzMHJua2JjNXpwYWkxeTJ3bWp6bCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Gl7mfimOjkkGl5mMDS/giphy.webp",
}

# --- PAGE SETUP ---
st.set_page_config(
    page_title="🧠 Emotion Detector From Text",
    page_icon=PAGE_ICON_URL,
    layout="wide",
)

# --- CUSTOM CSS (Redesigned for a cleaner, modern look) ---
st.markdown("""
    <style>
    /* ---------------------------------------------------- */
    /* 1. FONT IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&family=Orbitron:wght@600;800&family=Fira+Code:wght@400&display=swap');
    
    /* 2. COLOR PALETTE DEFINITION (DARK/CYBERPUNK NEON) */
    :root {
        --primary-color: #00eaff; /* Cyan Neon */
        --primary-dark: #0099cc;  /* Darker Cyan */
        --background-dark: #0a0a1a; /* Very dark blue/black */
        --surface-color: #1a1a2e; /* Deep purple/blue */
        --text-color-light: #e0e0f0; 
        --text-color-secondary: #9090aa; 
        
        --main-font: 'Roboto', sans-serif; 
        --mono-font: 'Fira Code', monospace; 
        --header-font: 'Orbitron', sans-serif; 
        
        --glow-strength: 0 0 10px var(--primary-color), 0 0 20px var(--primary-dark);
    }
    
    /* --- EMOTION SPECIFIC COLOR MAP --- */
    .emotion-anger { --emotion-color: #ff3366; } /* Red Neon */
    .emotion-joy { --emotion-color: #ccff33; } /* Lime Green Neon */
    .emotion-sadness { --emotion-color: #33ccff; } /* Blue Neon */
    .emotion-fear { --emotion-color: #ff66ff; } /* Magenta Neon */
    .emotion-neutral { --emotion-color: var(--primary-color); }
    .emotion-disgust { --emotion-color: #66ff33; } /* Dark Green Neon */
    .emotion-surprise { --emotion-color: #ffaa00; } /* Orange Neon */

    
    /* 3. OVERALL LAYOUT & BACKGROUND */
    .main {
        background: var(--background-dark);
        padding: 2.5rem 3rem; 
        font-family: var(--main-font); 
        color: var(--text-color-light); 
    }
    .stApp .st-emotion-cache-1pxn4ip, .stApp .st-emotion-cache-1v0pmnt {
        background-color: var(--surface-color);
        border-radius: 12px;
        padding: 30px; /* Increased padding */
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6); /* Deeper shadow */
    }

    /* 4. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--primary-color);
        font-weight: 800;
        font-family: var(--header-font);
        text-align: center;
        text-shadow: 0 0 8px var(--primary-color), 0 0 15px var(--primary-dark); 
        letter-spacing: 4px;
        margin-bottom: 0.5rem;
    }
    
    /* --- SECTION SUBHEADER (h3) --- */
    h3 {
        color: var(--primary-color);
        font-family: var(--header-font); 
        font-weight: 600; 
        border-bottom: 2px solid var(--primary-dark);
        padding-bottom: 5px;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 5px var(--primary-dark); 
    }

    /* 5. TEXT AREA EFFECTS */
    textarea {
        border-radius: 8px !important;
        border: 1px solid var(--primary-dark) !important;
        background-color: #0d0d21 !important; /* Slightly darker input field */
        color: var(--primary-color) !important;
        font-size: 16px !important; 
        font-family: var(--mono-font) !important; 
        box-shadow: 0 0 8px rgba(0, 234, 255, 0.2); 
    }

    /* 6. BUTTON EFFECTS */
    div.stButton > button:first-child {
        background: var(--primary-dark);
        color: var(--background-dark);
        font-weight: 700;
        font-family: var(--header-font);
        border-radius: 8px;
        padding: 0.8em 2em;
        box-shadow: 0 0 15px var(--primary-color); 
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02); 
        background: var(--primary-color); 
        box-shadow: 0 0 25px var(--primary-color); 
    }
    
    /* 7. CUSTOM RESULT CARDS (Simplified & modern) */
    .result-card {
        background-color: var(--background-dark);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid var(--emotion-color, var(--primary-dark)); 
        box-shadow: 0 0 10px var(--emotion-color, rgba(0, 0, 0, 0.1));
        transition: all 0.3s ease;
    }
    
    .card-row-1 {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .emotion-label {
        display: inline-block;
        font-size: 1.1rem;
        font-weight: 800;
        font-family: var(--header-font);
        color: var(--text-color-light);
        background-color: var(--surface-color);
        padding: 4px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        border: 2px solid var(--emotion-color);
        text-shadow: 0 0 5px var(--emotion-color);
    }
    
    .result-text {
        color: var(--text-color-light);
        font-family: var(--main-font); 
        font-size: 1rem;
        padding: 8px 0;
        border-left: 3px solid var(--emotion-color);
        padding-left: 10px;
        margin-bottom: 10px;
        font-style: italic;
    }

    .result-confidence {
        font-size: 0.9rem;
        font-weight: 400;
        color: var(--text-color-secondary);
        font-family: var(--mono-font); 
        text-align: right;
    }
    
    .emotion-gif-col {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 15px;
        background-color: var(--surface-color);
        border-radius: 10px;
        border: 1px solid var(--primary-dark);
    }
    
    .emotion-gif {
        width: 100px; 
        height: 100px;
        border-radius: 5px;
        object-fit: cover;
        box-shadow: 0 0 15px var(--emotion-color, var(--primary-color));
        border: 2px solid var(--emotion-color, var(--primary-color));
    }

    /* 8. DIVIDER & FOOTER */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(0,0,0,0), var(--primary-color), rgba(0,0,0,0)); 
        margin: 3rem 0;
        box-shadow: 0 0 5px var(--primary-color);
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

# --- MODEL LOADING ---
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

    # Filter out empty/whitespace-only strings before sending to model
    valid_texts = [t for t in texts if t]
    if not valid_texts:
         return []
         
    predictions = classifier(valid_texts)
    
    results = []
    for text, prediction_list in zip(valid_texts, predictions):
        # The model returns a list of dictionaries, one for each label
        best = max(prediction_list, key=lambda x: x['score'])
        results.append({
            "Input Text": text,
            "Dominant Emotion": best['label'].upper(),
            "Confidence": f"{best['score']:.4f}"
        })
    return results

# =================================================================
# --- APP LAYOUT (REVISED) ---
# =================================================================

# MAIN TITLE
st.title("CYBER-SENTIMENT ANALYZER")
st.markdown(f'<p style="color: var(--text-color-secondary); text-align: center; font-family: var(--header-font); letter-spacing: 1px;">TEXT EMOTION DETECTOR // MODEL: {MODEL_NAME.split("/")[-1]}</p>', unsafe_allow_html=True)

st.markdown("---")

# 1. INPUT BLOCK (Centralized and Cleaned)
with st.container():
    
    st.subheader("INPUT DATA STREAM")
    
    # --- TEXT AREA ---
    default_text = """I am so incredibly happy and proud of what we achieved today!
This is confusing; I need someone to clarify the instructions for step three.
My heart is racing, I'm genuinely terrified of what might happen next."""
    
    # Use a single, wide column for input
    input_text = st.text_area(
        "Enter sentences separated by new lines:", 
        value=default_text,
        height=200,
        key="input_text_area",
        label_visibility="collapsed"
    )
    
    # Pre-process texts (split by newline, strip whitespace, filter empty lines)
    texts = [t.strip() for t in input_text.split("\n") if t.strip()]
    
    # Center the button
    col_btn_l, col_btn, col_btn_r = st.columns([1.5, 2, 1.5])
    with col_btn:
        analyze = st.button("INITIATE SCAN // ANALYZE 🚀", use_container_width=True)

# Load the model silently
classifier = initialize_classifier()

st.markdown("---")

# 2. RESULTS BLOCK (Revised Layout: GIF on the side)
if analyze:
    if texts:
        results_container = st.container()
        with results_container:
            st.subheader("ANALYSIS REPORT // EMOTIONAL HITS")
            
            with st.spinner("Processing data... Standby."):
                results = detect_emotions(classifier, texts)
                
                if not results:
                    st.warning("No valid text lines found for analysis.")
                
                # Display results with a fixed layout: Text card on left, GIF on right
                for result in results:
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
                    
                    
                    # Use two columns for the card and the GIF
                    col_card, col_gif = st.columns([3, 1])
                    
                    # --- Result Card (Left Column) ---
                    with col_card:
                        st.markdown(f"""
                            <div class="result-card {css_class}">
                                <div class="card-row-1">
                                    <span class="emotion-label">{emotion}</span>
                                    <div class="result-confidence">TRUST LEVEL: {confidence}</div>
                                </div>
                                <div class="result-text">"{input_text}"</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # --- GIF Display (Right Column) ---
                    with col_gif: 
                        # Use markdown for the GIF display to apply custom styling
                        st.markdown(f"""
                            <div class="emotion-gif-col {css_class}">
                                <img src="{gif_url}" class="emotion-gif" alt="{emotion} GIF">
                            </div>
                        """, unsafe_allow_html=True)
                        
    else:
        st.warning("ALERT! Input text field is empty. Please enter text to begin analysis.")

# 3. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption"> [PROJECT: THAYEEB] // V.1.0 // DATA SOURCE: HUGGING FACE TRANSFORMERS</p>', unsafe_allow_html=True)

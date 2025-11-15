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
    page_title="👾 8-BIT EMOTION DETECTOR 👾",
    page_icon=PAGE_ICON_URL,
    layout="wide",
)

# --- CUSTOM CSS (Retro/8-bit Theme) ---
st.markdown("""
    <style>
    /* ---------------------------------------------------- */
    /* 1. FONT IMPORTS (Retro Fonts) */
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    
    /* 2. COLOR PALETTE DEFINITION (Retro Neon/Vaporwave) */
    :root {
        --primary-color: #00ffc8;      /* Neon Green/Teal */
        --primary-dark: #008c79;       /* Darker Teal */
        --background-dark: #222244;    /* Dark Indigo/Purple for background depth */
        --surface-color: #333366;      /* Slightly lighter indigo for cards/boxes */
        --text-color-light: #ffffff;   /* White text */
        --text-color-secondary: #ff00ff; /* Neon Pink/Magenta for highlights */
        
        --main-font: 'VT323', monospace;      /* Main text - Monospaced */
        --header-font: 'Press Start 2P', cursive; /* Header text - Pixelated */
    }
    
    /* --- EMOTION SPECIFIC COLOR MAP (High Contrast Retro) --- */
    .emotion-anger, .emotion-disgust { --emotion-color: #ff3300; } /* Neon Red */
    .emotion-joy, .emotion-happiness, .emotion-excitement { --emotion-color: #ffff00; } /* Neon Yellow */
    .emotion-sadness, .emotion-loneliness { --emotion-color: #00ffff; } /* Neon Cyan */
    .emotion-fear, .emotion-surprise { --emotion-color: #ff00ff; } /* Neon Magenta */
    .emotion-neutral { --emotion-color: var(--primary-color); }
    
    /* 3. OVERALL LAYOUT & BACKGROUND */
    .main {
        background: var(--background-dark);
        padding: 2.5rem 4rem; 
        font-family: var(--main-font); 
        color: var(--text-color-light); 
        /* Optional: Subtle repeating pattern for retro feel */
        /* background-image: radial-gradient(var(--surface-color) 1px, transparent 1px);
        background-size: 40px 40px; */
    }
    .stApp .st-emotion-cache-1pxn4ip, .stApp .st-emotion-cache-1v0pmnt {
        background-color: var(--surface-color);
        border-radius: 0; /* Boxy aesthetic */
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 4px 4px 0 var(--primary-dark); /* 8-bit shadow effect */
        border: 2px solid var(--text-color-light);
    }
    
    /* 4. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--text-color-light);
        font-family: var(--header-font);
        font-weight: 400; /* Press Start 2P is monoweight */
        text-align: center;
        text-shadow: 
            2px 2px 0 var(--text-color-secondary), /* Pixel shift shadow */
            -2px -2px 0 var(--primary-color); /* Double shadow effect */
        letter-spacing: 5px;
        font-size: 2.5em; /* Increase size for pixel font legibility */
        line-height: 1.2;
    }
    
    /* --- SUBHEADER (h3) - Retro Console Prompt Style --- */
    h3 {
        color: var(--text-color-light);
        font-family: var(--main-font); 
        font-weight: 400; 
        padding-left: 10px; 
        border: none;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.6em; /* VT323 size adjustment */
        
        /* Simulating blinking cursor/console prompt */
        display: inline-block;
        animation: blink-caret .75s step-end infinite;
        text-shadow: 0 0 5px var(--primary-color);
    }
    @keyframes blink-caret {
        from, to { border-right: .15em solid var(--primary-color); }
        50% { border-right: .15em solid transparent; }
    }
    
    /* 5. TEXT AREA EFFECTS (Console Input) */
    textarea {
        border-radius: 0 !important; /* Boxy */
        border: 4px solid var(--text-color-secondary) !important; /* Thick border */
        padding: 15px !important;
        background-color: var(--background-dark) !important;
        color: var(--primary-color) !important; /* Green text on dark background */
        font-size: 20px !important; 
        font-family: var(--main-font) !important; 
        line-height: 1.4; 
        box-shadow: 4px 4px 0 var(--text-color-secondary); /* 8-bit shadow */
    }
    
    /* 6. BUTTON EFFECTS (Arcade Button) */
    div.stButton > button:first-child {
        background: var(--text-color-secondary); /* Neon Pink */
        color: var(--background-dark);
        font-family: var(--header-font);
        font-weight: 400;
        border-radius: 5px; /* Slightly rounded box */
        padding: 0.8em 2em;
        box-shadow: 0 5px 0 var(--primary-dark), 0 0 10px var(--text-color-secondary); 
        text-transform: uppercase;
        letter-spacing: 2px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(2px); /* Simulate button press */
        background: var(--primary-color); /* Color change on hover */
        box-shadow: 0 3px 0 var(--primary-dark), 0 0 15px var(--primary-color); 
    }
    
    /* 7. CUSTOM RESULT CARDS and other elements */
    .result-text {
        color: var(--text-color-light);
        font-family: var(--main-font); 
        font-size: 1.4rem; /* Larger for VT323 */
        margin-bottom: 10px;
        font-style: italic;
    }
    
    .result-confidence {
        font-size: 1.2rem;
        font-weight: 400;
        color: var(--primary-color);
        font-family: var(--main-font); 
    }
    
    .result-card {
        background-color: var(--background-dark); /* Darker background for card content */
        border: 4px solid var(--emotion-color, var(--primary-dark)); 
        border-left: 15px solid var(--emotion-color, var(--primary-color)); 
        border-radius: 0; /* Boxy */
        padding: 15px 20px;
        margin-bottom: 20px;
        box-shadow: 4px 4px 0 var(--emotion-color, var(--primary-dark)), 0 0 15px rgba(0,0,0,0.5);
        transition: all 0.1s ease;
    }
    
    .card-header-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .emotion-gif {
        width: 64px; /* Slightly larger image */
        height: 64px;
        border-radius: 0; /* Boxy */
        object-fit: cover;
        border: 3px dashed var(--emotion-color); /* Dashed border for retro */
        box-shadow: 0 0 15px var(--emotion-color);
    }
    
    .result-emotion {
        display: inline-block;
        font-size: 1.6rem;
        font-weight: 400;
        color: var(--background-dark);
        background-color: var(--emotion-color);
        padding: 5px 12px;
        border-radius: 0;
        text-transform: uppercase;
        box-shadow: 2px 2px 0 var(--primary-dark);
        font-family: var(--header-font);
    }
    
    /* 8. DIVIDER & FOOTER */
    hr {
        border: 0;
        height: 4px;
        background: linear-gradient(to right, rgba(0,0,0,0), var(--text-color-secondary), var(--text-color-secondary), rgba(0,0,0,0)); 
        margin: 3rem 0;
    }
    footer, header { visibility: hidden !important; }
    
    /* Caption for the footer */
    .st-emotion-detector-caption {
        text-align: center; 
        font-family: var(--main-font); 
        color: var(--text-color-secondary); 
        font-size: 1.2rem;
        text-shadow: 0 0 5px var(--text-color-secondary);
    }

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
# --- APP LAYOUT ---
# =================================================================

# MAIN TITLE
st.title("👾 EMOTION DETECTOR V1.98 💾")
st.markdown(f'<p style="color: var(--text-color-secondary); text-align: center; font-family: var(--main-font); font-size: 1.4rem;">SCANNING WORDS FOR FEELING STATES...</p>', unsafe_allow_html=True)

st.markdown("---")

# 1. INPUT BLOCK
input_container = st.container()
with input_container:
    
    st.subheader("ENTER TEXT: >")
    
    # --- TEXT AREA ---
    col1, col_input, col2 = st.columns([1, 4, 1])

    default_text = """I am so incredibly happy and proud of what we achieved today!
This is confusing; I need someone to clarify the instructions for step three.
My heart is racing, I'm genuinely terrified of what might happen next."""
    
    with col_input:
        # Text area without a label
        input_text = st.text_area(
            "", 
            value=default_text,
            height=200,
            key="input_text_area"
        )
        texts = [t.strip() for t in input_text.split("\n") if t.strip()]
        
        col_btn_l, col_btn, col_btn_r = st.columns([1.5, 2, 1.5])
        with col_btn:
              analyze = st.button("▶️ EXECUTE EMOTION CHECK ◀️", use_container_width=True)

# Load the model silently
classifier = initialize_classifier()

st.markdown("---")

# 2. RESULTS BLOCK
if analyze:
    if texts:
        results_container = st.container()
        with results_container:
            st.subheader("SCAN REPORT: EMOTIONS DETECTED >")
            
            # Use two columns to display results cards
            cols = st.columns(2)
            
            with st.spinner("Processing... "):
                results = detect_emotions(classifier, texts)
                
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
        st.warning("ERROR: INPUT BUFFER EMPTY. PLEASE ENTER TEXT.")

# 3. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption">TERMINAL ACCESS BY CSE-A 1985</p>', unsafe_allow_html=True)

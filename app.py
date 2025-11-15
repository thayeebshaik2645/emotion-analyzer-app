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
    page_title="🕯️ THE UPSIDE DOWN EMOTION DETECTOR",
    page_icon=PAGE_ICON_URL,
    layout="wide",
)

# --- CUSTOM CSS (Stranger Things Theme) ---
st.markdown("""
    <style>
    /* ---------------------------------------------------- */
    /* 1. FONT IMPORTS (Stranger Things 'ITC Benguiat' Lookalikes) */
    /* Cinzel for Serif Title look and VCR OSD Mono for 8-bit/Screen text */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=VCR+OSD+Mono&display=swap');
    
    /* 2. COLOR PALETTE DEFINITION (Deep Red, Dark Blue, Black) */
    :root {
        --primary-color: #ff0000;      /* Deep Red Neon Glow */
        --primary-dark: #cc0000;       /* Darker Red */
        --background-dark: #0a0a0a;    /* Near Black */
        --surface-color: #1a1a1a;      /* Dark Gray/Black for components */
        --text-color-light: #f0f0f0;   /* White/Light Gray */
        --text-color-secondary: #0000cc; /* Deep Blue Accent (The Upside Down) */
        
        --main-font: 'VCR OSD Mono', monospace; /* Console/Screen text */
        --header-font: 'Cinzel', serif;        /* Title Font */
    }
    
    /* --- EMOTION SPECIFIC COLOR MAP (Subtle Red/Blue Shift) --- */
    .emotion-anger, .emotion-disgust { --emotion-color: #ff4444; } /* Lighter Red */
    .emotion-joy, .emotion-happiness, .emotion-excitement { --emotion-color: #ffdd00; } /* Yellow/Gold for warmth */
    .emotion-sadness, .emotion-loneliness { --emotion-color: #008cff; } /* Light Blue */
    .emotion-fear, .emotion-surprise { --emotion-color: #ff00ff; } /* Magenta/Purple */
    .emotion-neutral { --emotion-color: var(--primary-color); }
    
    /* 3. OVERALL LAYOUT & BACKGROUND */
    .main {
        background: var(--background-dark);
        padding: 2.5rem 4rem; 
        font-family: var(--main-font); 
        color: var(--text-color-light); 
        /* CRT/Scanline Effect overlay */
        position: relative;
    }
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: repeating-linear-gradient(
            to bottom,
            rgba(0, 0, 0, 0.1),
            rgba(0, 0, 0, 0.5) 1px,
            transparent 1px,
            transparent 3px
        );
        pointer-events: none;
        z-index: 1000;
    }
    .stApp .st-emotion-cache-1pxn4ip, .stApp .st-emotion-cache-1v0pmnt {
        background-color: var(--surface-color);
        border-radius: 2px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(255, 0, 0, 0.2); /* Subtle red glow around elements */
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* 4. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--primary-color);
        font-family: var(--header-font);
        font-weight: 900;
        text-align: center;
        letter-spacing: 0.15em; /* Wide letter spacing */
        font-size: 4em;
        line-height: 1.2;
        
        /* Stranger Things Glow Effect (Red on Black) */
        text-shadow: 
            0 0 10px var(--primary-color),
            0 0 20px var(--primary-dark),
            0 0 30px var(--primary-dark);
    }
    
    /* --- SUBHEADER (h3) - Screen Display Text --- */
    h3 {
        color: var(--text-color-light);
        font-family: var(--main-font); 
        font-weight: 400; 
        padding-left: 10px; 
        border: none;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.4em;
        text-shadow: 0 0 5px var(--primary-color); /* Subtle screen glow */
    }
    
    /* 5. TEXT AREA EFFECTS (Data Stream) */
    textarea {
        border-radius: 0 !important;
        border: 2px solid var(--text-color-secondary) !important; /* Deep Blue border */
        padding: 15px !important;
        background-color: var(--background-dark) !important;
        color: var(--primary-color) !important; /* Red text on black */
        font-size: 1.2rem !important; 
        font-family: var(--main-font) !important; 
        line-height: 1.6; 
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.8), 0 0 5px var(--primary-color); /* Inner shadow for depth */
    }
    
    /* 6. BUTTON EFFECTS (Alert/Urgent) */
    div.stButton > button:first-child {
        background: var(--primary-color); /* Red */
        color: var(--background-dark);
        font-family: var(--header-font);
        font-weight: 600;
        border-radius: 2px;
        padding: 0.8em 2em;
        box-shadow: 0 0 20px var(--primary-color); /* Intense red glow */
        text-transform: uppercase;
        letter-spacing: 3px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02); 
        background: var(--primary-dark); 
        box-shadow: 0 0 30px var(--primary-color), 0 0 5px var(--primary-color); 
    }
    
    /* 7. CUSTOM RESULT CARDS */
    .result-text {
        color: var(--text-color-light);
        font-family: var(--main-font); 
        font-size: 1.1rem;
        margin-bottom: 10px;
        font-style: normal;
    }
    
    .result-confidence {
        font-size: 1rem;
        font-weight: 400;
        color: var(--text-color-light);
        font-family: var(--main-font); 
        text-shadow: 0 0 3px var(--emotion-color, var(--primary-color));
    }
    
    .result-card {
        background-color: var(--surface-color);
        border: 2px solid var(--emotion-color, var(--primary-dark)); 
        border-left: 8px solid var(--emotion-color, var(--primary-color)); 
        border-radius: 4px;
        padding: 15px 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 8px var(--emotion-color, rgba(255, 0, 0, 0.4));
        transition: all 0.3s ease;
    }
    
    .card-header-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .emotion-gif {
        width: 50px; 
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid var(--emotion-color);
        box-shadow: 0 0 10px var(--emotion-color);
    }
    
    .result-emotion {
        display: inline-block;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--surface-color);
        background-color: var(--emotion-color);
        padding: 5px 12px;
        border-radius: 4px;
        text-transform: uppercase;
        box-shadow: 0 0 5px var(--emotion-color);
        font-family: var(--header-font);
    }
    
    /* 8. DIVIDER & FOOTER */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(255,0,0,0), var(--primary-color), var(--primary-color), rgba(255,0,0,0)); 
        margin: 3rem 0;
    }
    footer, header { visibility: hidden !important; }
    
    /* Caption for the footer */
    .st-emotion-detector-caption {
        text-align: center; 
        font-family: var(--main-font); 
        color: var(--text-color-light); 
        font-size: 1rem;
        text-shadow: 0 0 5px var(--primary-color);
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
st.title("S T R A N G E R   T E X T S")
st.markdown(f'<p style="color: var(--text-color-light); text-align: center; font-family: var(--header-font); font-size: 1.5em; letter-spacing: 0.1em; text-shadow: 0 0 8px var(--primary-color);">A N A L Y S I S   O F   T H E   V O I C E</p>', unsafe_allow_html=True)

st.markdown("---")

# 1. INPUT BLOCK
input_container = st.container()
with input_container:
    
    st.subheader("INPUT: TRANSMISSION RECEIVED >>")
    
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
              analyze = st.button("🔴 OPEN GATE TO EMOTIONS 🔴", use_container_width=True)

# Load the model silently
classifier = initialize_classifier()

st.markdown("---")

# 2. RESULTS BLOCK
if analyze:
    if texts:
        results_container = st.container()
        with results_container:
            st.subheader("OUTPUT: THE UPSIDE DOWN ANALYSIS >>")
            
            # Use two columns to display results cards
            cols = st.columns(2)
            
            with st.spinner("Processing... The lights are flickering..."):
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
        st.warning("SYSTEM ALERT: TEXT INPUT REQUIRED. DANGER IMMINENT.")

# 3. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption">RUN TIME 1983. ALL RIGHTS RESERVED BY HAWKINS LAB</p>', unsafe_allow_html=True)

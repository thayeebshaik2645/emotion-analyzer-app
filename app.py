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

/* ---------------------------------------------- */
/* STRANGER THINGS RETRO UPSIDE-DOWN THEME        */
/* ---------------------------------------------- */

/* Import retro fonts */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;900&family=Roboto+Mono:wght@300;400;700&display=swap');

/* Color palette */
:root {
    --st-red: #e50914;
    --st-red-glow: #ff1d2d;
    --st-black: #0a0a0a;
    --st-dark: #111;
    --st-smoke: rgba(255, 0, 0, 0.15);
    --text-light: #e3e3e3;
}

/* Whole app background */
.main {
    background: radial-gradient(circle at top, #1b0006 0%, #000000 70%);
    background-attachment: fixed;
    color: var(--text-light);
    font-family: 'Roboto Mono', monospace;
    text-shadow: 0px 0px 6px rgba(255,10,25,0.4);
}

/* Fog/Smoke Effect */
.main::before {
    content: "";
    position: fixed;
    left: 0; top: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    background: url("https://i.imgur.com/8fK4hFf.png");
    opacity: 0.09;
    mix-blend-mode: screen;
    animation: drift 60s infinite linear;
}

@keyframes drift {
    0% { transform: translateX(0); }
    100% { transform: translateX(20%); }
}

/* Titles – Stranger Things neon red */
h1 {
    font-family: 'Cinzel', serif;
    color: var(--st-red);
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    text-shadow: 
        0 0 10px var(--st-red-glow),
        0 0 20px var(--st-red-glow),
        0 0 40px var(--st-red-glow);
    letter-spacing: 3px;
}

/* Subtitles */
h3 {
    color: var(--st-red);
    font-family: 'Cinzel', serif;
    font-weight: 700;
    border-bottom: 2px solid var(--st-red);
    padding-bottom: 5px;
    margin-top: 2rem;
}

/* Text area */
textarea {
    background: #0d0d0d !important;
    color: var(--text-light) !important;
    border: 2px solid var(--st-red) !important;
    font-family: 'Roboto Mono', monospace !important;
    box-shadow: 0 0 10px var(--st-red-glow);
}

/* Buttons */
div.stButton > button:first-child {
    background: var(--st-red);
    border: none;
    color: white;
    font-weight: 700;
    padding: 0.7em 1.7em;
    border-radius: 5px;
    font-family: 'Cinzel', serif;
    letter-spacing: 2px;
    box-shadow: 0 0 15px var(--st-red-glow);
    transition: 0.2s ease-in-out;
}

div.stButton > button:first-child:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px var(--st-red-glow);
}

/* Result Cards — Upside-Down look */
.result-card {
    background: rgba(10, 0, 0, 0.85);
    border: 2px solid var(--st-red);
    border-left: 5px solid var(--st-red);
    padding: 16px;
    border-radius: 6px;
    margin-bottom: 20px;
    box-shadow: 
        0 0 10px rgba(255, 0, 0, 0.5),
        inset 0 0 10px rgba(255, 0, 0, 0.3);
}

/* Emotion badge */
.result-emotion {
    background: var(--st-red);
    color: white;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: 900;
    font-family: 'Cinzel', serif;
    letter-spacing: 1px;
    box-shadow: 0 0 10px var(--st-red-glow);
}

/* GIF circle */
.emotion-gif {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    border: 3px solid var(--st-red);
    box-shadow: 0 0 10px var(--st-red-glow);
}

/* Input text in cards */
.result-text {
    font-style: italic;
    color: #ccc;
    margin: 8px 0;
}

/* Confidence meter label */
.result-confidence {
    color: #999;
    font-size: 0.8rem;
    font-family: 'Roboto Mono', monospace;
}

/* Footer */
.st-emotion-detector-caption {
    text-align: center;
    color: #666;
    margin-top: 10px;
}

footer, header { visibility: hidden; }

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


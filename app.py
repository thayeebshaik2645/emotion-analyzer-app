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

# --- CUSTOM CSS (Clean, Light Mode Redesign) ---
st.markdown("""
    <style>
    /* ---------------------------------------------------- */
    /* 1. FONT IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=Playfair+Display:wght@700&display=swap');
    
    /* 2. COLOR PALETTE DEFINITION (CLEAN / PROFESSIONAL LIGHT) */
    :root {
        --primary-color: #173f5f; /* Deep Blue Accent */
        --primary-light: #528cbe; /* Light Blue */
        --background-light: #f7f7f7; /* Off-White Background */
        --surface-color: #ffffff; /* White Cards/Containers */
        --text-color-dark: #333333; 
        --text-color-secondary: #777777; 
        
        --main-font: 'Inter', sans-serif; 
        --header-font: 'Playfair Display', serif; 
    }
    
    /* --- EMOTION SPECIFIC COLOR MAP (Professional Tones) --- */
    .emotion-anger { --emotion-color: #e63946; } /* Red */
    .emotion-joy { --emotion-color: #2a9d8f; } /* Teal */
    .emotion-sadness { --emotion-color: #457b9d; } /* Steel Blue */
    .emotion-fear { --emotion-color: #ffb703; } /* Amber */
    .emotion-neutral { --emotion-color: #adb5bd; } /* Grey */
    .emotion-disgust { --emotion-color: #84a98c; } /* Muted Green */
    .emotion-surprise { --emotion-color: #fca311; } /* Orange */

    
    /* 3. OVERALL LAYOUT & BACKGROUND */
    .main {
        background: var(--background-light);
        padding: 3rem 4rem; 
        font-family: var(--main-font); 
        color: var(--text-color-dark); 
    }
    .stApp .st-emotion-cache-1pxn4ip, .stApp .st-emotion-cache-1v0pmnt {
        background-color: var(--surface-color);
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Soft shadow */
    }

    /* 4. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--primary-color);
        font-weight: 900;
        font-family: var(--header-font);
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    /* --- SECTION SUBHEADER (h3) --- */
    h3 {
        color: var(--primary-color);
        font-family: var(--main-font); 
        font-weight: 700; 
        border-left: 5px solid var(--primary-light);
        padding-left: 10px;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
    }

    /* 5. TEXT AREA EFFECTS */
    textarea {
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
        background-color: #ffffff !important; 
        color: var(--text-color-dark) !important;
        font-size: 16px !important; 
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1); 
    }

    /* 6. BUTTON EFFECTS */
    div.stButton > button:first-child {
        background: var(--primary-color);
        color: #ffffff;
        font-weight: 700;
        font-family: var(--main-font);
        border-radius: 8px;
        padding: 0.6em 1.5em;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: background-color 0.2s ease, transform 0.2s ease;
        border: none;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px); 
        background: var(--primary-light); 
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15); 
    }
    
    /* 7. CUSTOM RESULT CARDS */
    .result-card {
        background-color: var(--surface-color);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid var(--emotion-color, #dddddd); 
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .card-row-1 {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .emotion-label {
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 700;
        color: #ffffff;
        background-color: var(--emotion-color);
        padding: 4px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    
    .result-text {
        color: var(--text-color-dark);
        font-size: 1rem;
        padding: 5px 0;
        border-left: 3px solid var(--emotion-color);
        padding-left: 10px;
        margin-bottom: 8px;
        font-style: italic;
    }

    .result-confidence {
        font-size: 0.8rem;
        font-weight: 400;
        color: var(--text-color-secondary);
        text-align: right;
    }
    
    .emotion-gif-col {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
        background-color: var(--surface-color);
        border-radius: 10px;
        border: 1px dashed var(--emotion-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .emotion-gif {
        width: 80px; 
        height: 80px;
        border-radius: 5px;
        object-fit: cover;
        border: 1px solid #ddd;
    }

    /* 8. DIVIDER & FOOTER */
    hr {
        border: 0;
        height: 1px;
        background: #eeeeee; 
        margin: 2rem 0;
    }
    .st-emotion-detector-caption {
        text-align: center;
        color: var(--text-color-secondary);
        font-family: var(--main-font);
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
        best = max(prediction_list, key=lambda x: x['score'])
        results.append({
            "Input Text": text,
            "Dominant Emotion": best['label'].upper(),
            "Confidence": f"{best['score']:.4f}"
        })
    return results

# =================================================================
# --- APP LAYOUT (REVISED: CLEAN LIGHT MODE) ---
# =================================================================

# MAIN TITLE
st.title("Sentiment & Emotion Analyzer 📊")
st.markdown(f'<p style="color: var(--text-color-secondary); text-align: center; font-family: var(--main-font);">Analyzing Textual Emotion using {MODEL_NAME.split("/")[-1]}</p>', unsafe_allow_html=True)

st.markdown("---")

# 1. INPUT BLOCK 
with st.container():
    
    st.subheader("1. Enter Text for Analysis")
    
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
    
    # Pre-process texts 
    texts = [t.strip() for t in input_text.split("\n") if t.strip()]
    
    # Center the button
    col_btn_l, col_btn, col_btn_r = st.columns([1.5, 2, 1.5])
    with col_btn:
        analyze = st.button("Analyze Emotions", use_container_width=True)

# Load the model silently
classifier = initialize_classifier()

st.markdown("---")

# 2. RESULTS BLOCK 
if analyze:
    if texts:
        results_container = st.container()
        with results_container:
            st.subheader("2. Analysis Results")
            
            with st.spinner("Analyzing text..."):
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
                    col_card, col_gif = st.columns([4, 1])
                    
                    # --- Result Card (Left Column) ---
                    with col_card:
                        st.markdown(f"""
                            <div class="result-card {css_class}">
                                <div class="card-row-1">
                                    <span class="emotion-label">{emotion}</span>
                                    <div class="result-confidence">Confidence: {confidence}</div>
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
        st.warning("Please enter some text in the input box to start the analysis.")

# 3. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption">Built by THAYEEB | Model: Hugging Face Transformers</p>', unsafe_allow_html=True)

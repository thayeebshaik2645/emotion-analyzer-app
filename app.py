import streamlit as st
import pandas as pd
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
PAGE_ICON_URL = "https://cdn-icons-png.flaticon.com/128/10479/10479785.png"
ALL_EMOTIONS = ["ANGER", "HAPPINESS", "SADNESS", "JOY", "FEAR", "NEUTRAL", "DISGUST", "EXCITEMENT", "LONELINESS", "SURPRISE"]

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

# --- CUSTOM CSS (Stranger Things Theme with Flicker and Bar Styles) ---
# Paste the full enhanced CSS block here
st.markdown("""
    <style>
    /* ---------------------------------------------------- */
    /* 1. FONT IMPORTS (Stranger Things 'ITC Benguiat' Lookalikes) */
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
        
        --main-font: 'VCR OSD Mono', monospace; 
        --header-font: 'Cinzel', serif;        
    }
    
    /* --- EMOTION SPECIFIC COLOR MAP --- */
    .emotion-anger, .emotion-disgust { --emotion-color: #ff4444; } 
    .emotion-joy, .emotion-happiness, .emotion-excitement { --emotion-color: #ffdd00; } 
    .emotion-sadness, .emotion-loneliness { --emotion-color: #008cff; } 
    .emotion-fear, .emotion-surprise { --emotion-color: #ff00ff; } 
    .emotion-neutral { --emotion-color: var(--primary-color); }
    
    /* 3. OVERALL LAYOUT & BACKGROUND + SCANLINE/FLICKER */
    .main {
        background: var(--background-dark);
        padding: 2.5rem 4rem; 
        font-family: var(--main-font); 
        color: var(--text-color-light); 
        position: relative;
        /* Overall Flicker Animation */
        animation: flicker 2s infinite alternate;
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
    
    /* Flicker Keyframes */
    @keyframes flicker {
        0% { opacity: 1; }
        20% { opacity: 0.98; }
        40% { opacity: 1; }
        60% { opacity: 0.99; }
        80% { opacity: 1; }
        100% { opacity: 0.97; }
    }

    .stApp .st-emotion-cache-1pxn4ip, .stApp .st-emotion-cache-1v0pmnt {
        background-color: var(--surface-color);
        border-radius: 2px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(255, 0, 0, 0.2); 
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* 4. TITLES & TEXT EFFECTS */
    h1 {
        color: var(--primary-color);
        font-family: var(--header-font);
        font-weight: 900;
        text-align: center;
        letter-spacing: 0.15em; 
        font-size: 4em;
        line-height: 1.2;
        text-shadow: 
            0 0 10px var(--primary-color),
            0 0 20px var(--primary-dark),
            0 0 30px var(--primary-dark);
    }
    
    /* --- SUBHEADER (h3) - Screen Display Text --- */
    h3 {
        color: var(--text-color-light);
        font-family: var(--main-font); 
        padding-left: 10px; 
        border: none;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.4em;
        text-shadow: 0 0 5px var(--primary-color); 
    }
    
    /* 5. TEXT AREA EFFECTS (Data Stream) */
    textarea {
        border-radius: 0 !important;
        border: 2px solid var(--text-color-secondary) !important; 
        padding: 15px !important;
        background-color: var(--background-dark) !important;
        color: var(--primary-color) !important; 
        font-size: 1.2rem !important; 
        font-family: var(--main-font) !important; 
        line-height: 1.6; 
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.8), 0 0 5px var(--primary-color); 
    }
    
    /* 6. BUTTON EFFECTS (Alert/Urgent) */
    div.stButton > button:first-child {
        background: var(--primary-color); 
        color: var(--background-dark);
        font-family: var(--header-font);
        font-weight: 600;
        border-radius: 2px;
        padding: 0.8em 2em;
        box-shadow: 0 0 20px var(--primary-color); 
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
    /* Style for the emotion bars */
    .emotion-bar-container {
        margin-top: 10px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        background-color: #000000;
        box-shadow: inset 0 0 5px rgba(255, 0, 0, 0.5);
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
        # Prepare full scores dictionary
        scores = {p['label'].upper(): p['score'] for p in prediction_list}
        
        # Get dominant emotion
        best = max(prediction_list, key=lambda x: x['score'])
        
        results.append({
            "Input Text": text,
            "Dominant Emotion": best['label'].upper(),
            "Confidence": best['score'],
            "Full Scores": scores
        })
    return results

# =================================================================
# --- APP LAYOUT ---
# =================================================================

# MAIN TITLE
st.title("S T R A N G E R   T E X T S")
st.markdown(f'<p style="color: var(--text-color-light); text-align: center; font-family: var(--header-font); font-size: 1.5em; letter-spacing: 0.1em; text-shadow: 0 0 8px var(--primary-color);">A N A L Y S I S   O F   T H E   V O I C E</p>', unsafe_allow_html=True)

st.markdown("---")

# Use Streamlit Tabs for better segmentation
tab_input, tab_results = st.tabs(["[1] INPUT SEQUENCE", "[2] ANALYSIS REPORT"])

with tab_input:
    # --- CONFIDENCE THRESHOLD SETTING (New Feature 1) ---
    col_settings, col_input_area = st.columns([1, 4])

    with col_settings:
        st.subheader("SYSTEM SETTINGS")
        # Initialize session state for analysis results
        if 'analysis_results' not in st.session_state:
            st.session_state['analysis_results'] = []
            
        confidence_threshold = st.slider(
            "Threshold: Filter Low Confidence (Min: **0%**)",
            min_value=0.0,
            max_value=1.0,
            value=0.5, # Default to 50%
            step=0.05,
            format="%.0f%%",
            key="confidence_slider"
        )

    with col_input_area:
        st.subheader("INPUT: TRANSMISSION RECEIVED >>")
        default_text = """I am so incredibly happy and proud of what we achieved today!
This is confusing; I need someone to clarify the instructions for step three.
My heart is racing, I'm genuinely terrified of what might happen next."""
        
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

with tab_results:
    if analyze:
        if texts:
            with st.spinner("Processing... The lights are flickering..."):
                results = detect_emotions(classifier, texts)
                st.session_state['analysis_results'] = results # Store results in session state
            
            st.subheader("OUTPUT: THE UPSIDE DOWN ANALYSIS >>")

            # Filter results based on the threshold
            filtered_results = [
                r for r in results 
                if r['Confidence'] >= confidence_threshold
            ]

            if not filtered_results:
                st.warning(f"No results meet the **{confidence_threshold*100:.0f}%** confidence threshold.")
            
            cols = st.columns(2)
            
            for i, result in enumerate(filtered_results):
                emotion = result['Dominant Emotion']
                confidence = result['Confidence']
                input_text = result['Input Text']
                full_scores = result['Full Scores']
                
                gif_url = EMOTION_GIFS.get(emotion, EMOTION_GIFS["NEUTRAL"]) 
                
                css_class = ""
                if emotion in ["ANGER", "DISGUST"]: css_class = "emotion-anger"
                elif emotion in ["JOY", "HAPPINESS", "EXCITEMENT"]: css_class = "emotion-joy"
                elif emotion in ["SADNESS", "LONELINESS"]: css_class = "emotion-sadness"
                elif emotion in ["FEAR", "SURPRISE"]: css_class = "emotion-fear"
                elif emotion == "NEUTRAL": css_class = "emotion-neutral"
                
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
                                    CONFIDENCE LEVEL: {confidence:.4f}
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # --- Full Emotion Score Visualization (New Feature 2) ---
                    # Prepare data for Bar Chart
                    # Sort scores and convert to DataFrame for st.bar_chart
                    df_scores = pd.DataFrame(
                        full_scores.items(), 
                        columns=['Emotion', 'Confidence']
                    ).sort_values(by='Confidence', ascending=False).head(5) # Show top 5
                    
                    st.markdown(f'<div class="emotion-bar-container">', unsafe_allow_html=True)
                    st.markdown(f'<p style="font-family: var(--main-font); color: var(--primary-color); font-size: 0.9rem; margin: 0 0 5px 0;">RAW SENSORY DATA (TOP 5)</p>', unsafe_allow_html=True)

                    # Fix the index/label issue for Streamlit charts
                    df_scores = df_scores.set_index('Emotion')
                    
                    st.bar_chart(
                        df_scores, 
                        use_container_width=True, 
                        height=200,
                        color=css_class.replace('emotion-', '#') # Simple attempt to color the bars (Streamlit 1.29+ feature)
                    )
                    st.markdown(f'</div>', unsafe_allow_html=True)

        else:
            st.warning("ERROR: TEXT INPUT REQUIRED. DANGER IMMINENT.")
    else:
        st.info("AWAITING TRANSMISSION... CLICK 'OPEN GATE TO EMOTIONS' TO BEGIN ANALYSIS.")

# 3. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption">RUN TIME 1983. ALL RIGHTS RESERVED BY HAWKINS LAB</p>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
from transformers import pipeline

# --- IMAGE URLS (Fetched from web search) ---
BRAIN_ICON = "http://googleusercontent.com/image_collection/image_retrieval/7984592011731137749"
TERMINAL_ICON = "http://googleusercontent.com/image_collection/image_retrieval/4165680859544906968"
SCAN_ICON = "http://googleusercontent.com/image_collection/image_retrieval/10382102162479750995"
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Emotion Detector [Futuristic]",
    page_icon="🤖",  # Page icon must be emoji or file path
    layout="wide",
)

# --- CUSTOM CSS (DARK/NEON THEME & ICON STYLING) ---
st.markdown(f"""
    <style>
    /* 1. COLOR PALETTE DEFINITION (DARK/NEON) */
    :root {{
        --primary-color: #00ffc8; /* Neon Cyan/Green Accent */
        --primary-dark: #00b38c;  /* Darker Neon */
        --background-dark: #121212; /* Very Dark Background */
        --surface-color: #1e1e1e; /* Card/Container Background */
        --text-color-light: #f0f0f0; /* Light Text */
        --text-color-secondary: #aaaaaa; /* Gray Text */
        --mono-font: 'Consolas', 'Courier New', monospace; /* Futuristic Monospace Font */
    }}
    
    /* 2. OVERALL LAYOUT & BACKGROUND */
    .main {{
        background: var(--background-dark);
        padding: 2.5rem 4rem; 
        font-family: 'Inter', sans-serif;
        color: var(--text-color-light); 
    }}
    .stText, .stMarkdown {{
        color: var(--text-color-light) !important;
    }}
    
    /* Apply surface color to major containers for "block" look */
    .stApp .st-emotion-cache-1pxn4ip, .stApp .st-emotion-cache-1v0pmnt {{
        background-color: var(--surface-color);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.4);
    }}

    /* 3. TITLES & ICONS */
    h1 {{
        color: var(--primary-color);
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 5px var(--primary-color), 0 0 10px var(--primary-dark); 
        letter-spacing: 2px;
    }}
    h2, h3 {{
        color: var(--text-color-light);
        font-weight: 700;
        padding-left: 5px; 
        border-left: 5px solid var(--primary-color); 
        margin-top: 2rem;
        margin-bottom: 1rem;
    }}
    
    /* Global image styling for the futuristic icons */
    .futuristic-icon {{
        height: 28px;
        width: 28px;
        vertical-align: middle;
        margin-right: 10px;
        margin-left: 5px;
        filter: drop-shadow(0 0 3px var(--primary-color));
    }}

    /* 4. TEXT AREA EFFECTS */
    .stTextArea label {{
        font-weight: 600;
        color: var(--primary-color);
    }}
    textarea {{
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
    }}

    /* 5. BUTTON EFFECTS */
    div.stButton > button:first-child {{
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
    }}
    div.stButton > button:first-child:hover {{
        transform: translateY(-2px); 
        background: var(--primary-color); 
        box-shadow: 0 0 25px var(--primary-color), 0 0 5px var(--primary-color); 
    }}
    
    /* 6. METRIC & DATAFRAME STYLING */
    /* Target for st.metric containers */
    [data-testid="stMetric"] {{
        background-color: var(--surface-color);
        border: 1px solid var(--primary-dark);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 0 8px rgba(0, 255, 200, 0.2);
    }}
    /* Metric label color */
    [data-testid="stMetricLabel"] {{
        color: var(--text-color-secondary) !important;
        font-family: var(--mono-font);
        text-transform: uppercase;
    }}
    /* Metric value color (use primary neon) */
    [data-testid="stMetricValue"] {{
        color: var(--primary-color) !important;
        text-shadow: 0 0 5px rgba(0, 255, 200, 0.5);
    }}

    /* Dataframe and emotion coloring (kept from previous step) */
    .stDataFrame {{border-radius: 10px !important; box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.7); border: 1px solid var(--primary-dark); overflow: hidden; background-color: var(--surface-color);}}
    .stDataFrame .data-row, .stDataFrame th, .stDataFrame td {{color: var(--text-color-light) !important; font-family: var(--mono-font) !important;}}
    [data-testid="stDataframe"] div:nth-child(3) > div:not(:first-child) {{ font-weight: 800 !important; padding: 6px 10px !important; border-radius: 6px; text-align: center; margin: 4px 0; display: inline-block; min-width: 110px; transition: all 0.3s ease; text-shadow: 0 0 2px black; font-family: 'Inter', sans-serif !important;}}
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("ANGER"), div[data-testid="stDataframe"] .css-1r6cnx6:contains("DISGUST") {{ background-color: #ff3366; color: var(--background-dark); box-shadow: 0 0 8px #ff3366;}}
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("JOY"), div[data-testid="stDataframe"] .css-1r6cnx6:contains("HAPPINESS"), div[data-testid="stDataframe"] .css-1r6cnx6:contains("EXCITEMENT") {{ background-color: #fffb00; color: var(--background-dark); box-shadow: 0 0 8px #fffb00;}}
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("SADNESS"), div[data-testid="stDataframe"] .css-1r6cnx6:contains("LONELINESS") {{ background-color: #00aaff; color: var(--background-dark); box-shadow: 0 0 8px #00aaff;}}
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("FEAR"), div[data-testid="stDataframe"] .css-1r6cnx6:contains("SURPRISE") {{ background-color: #ff00ff; color: var(--background-dark); box-shadow: 0 0 8px #ff00ff;}}
    div[data-testid="stDataframe"] .css-1r6cnx6:contains("NEUTRAL") {{ background-color: var(--primary-color); color: var(--background-dark); box-shadow: 0 0 8px var(--primary-color);}}


    /* 7. DIVIDER & FOOTER */
    hr {{ border: 0; height: 2px; background: linear-gradient(to right, rgba(0,0,0,0), var(--primary-dark), rgba(0,0,0,0)); margin: 3rem 0;}}
    .st-emotion-detector-caption {{ text-align: center; color: var(--text-color-secondary); font-style: italic; margin-top: 1rem; animation: fadeIn 3s;}}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    footer, header {{ visibility: hidden !important; }}

    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def initialize_classifier():
    """Load and cache the transformer model."""
    try:
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

# --- HEADER (Title with Image) ---
st.markdown(
    f'<h1 style="text-align: center;">'
    f'<img src="{BRAIN_ICON}" class="futuristic-icon" alt="Brain Icon"> '
    f'EMOTION DETECTOR FROM TEXT'
    f'</h1>',
    unsafe_allow_html=True
)
st.markdown(f'<p style="color: var(--text-color-secondary); text-align: center; font-family: var(--mono-font);">ANALYSIS PROTOCOL V1.2.0 | EMOTIONAL RESPONSE LOGGING INTERFACE</p>', unsafe_allow_html=True)

st.markdown("---")

# 1. INPUT BLOCK (Terminal style)
input_container = st.container()
with input_container:
    # Subheader with Terminal Icon
    st.markdown(
        f'<h3>'
        f'<img src="{TERMINAL_ICON}" class="futuristic-icon" alt="Terminal Icon"> '
        f'INPUT TERMINAL'
        f'</h3>',
        unsafe_allow_html=True
    )
    
    col_input, _ = st.columns([4, 1])

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
        
        # Center the button using columns and add the Scan Icon
        col_btn_l, col_btn, col_btn_r = st.columns([1.5, 2, 1.5])
        with col_btn:
             analyze = st.button(
                f'<img src="{SCAN_ICON}" class="futuristic-icon" style="height: 18px; width: 18px;" alt="Scan Icon"> INITIATE ANALYSIS',
                use_container_width=True,
                unsafe_allow_html=True
             )

# Initialize classifier
classifier = initialize_classifier()

st.markdown("---")

# 2. RESULTS BLOCK (Structured Dashboard)
if analyze:
    if texts:
        results_container = st.container()
        with results_container:
            st.subheader("📊 OUTPUT DATA LOG")
            
            # --- STRUCTURE: [Summary Metrics] | [Detailed Data Log] ---
            col_metrics, col_data = st.columns([1, 2])
            
            with st.spinner("Processing data... Stand by."):
                results = detect_emotions(classifier, texts)
                df = pd.DataFrame(results)
                
                # Summary Metrics (Left Column)
                with col_metrics:
                    dominant_emotion = df['Dominant Emotion'].mode()[0]
                    total_sentences = len(df)
                    
                    st.metric(
                        label="TOTAL LOG ENTRIES", 
                        value=f"{total_sentences}"
                    )
                    st.metric(
                        label="DOMINANT RESPONSE", 
                        value=f"{dominant_emotion}"
                    )
                    
                    st.markdown('<p style="color: var(--text-color-secondary); font-size: 14px; margin-top: 20px;">KEY RESPONSE SIGNALS DETECTED.</p>', unsafe_allow_html=True)
                
                # Detailed Log Table (Right Column)
                with col_data:
                    st.markdown("#### Detailed Analysis Log:")
                    st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.warning("Input required. Please provide text before initiating analysis.")

# 3. FOOTER
st.markdown("---")
st.markdown('<p class="st-emotion-detector-caption">SYSTEM ONLINE | BUILD BY CSE-A</p>', unsafe_allow_html=True)

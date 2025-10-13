import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Configuration ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- MINIONS COLOR PALETTE ---
# Based on Minions (Yellow, Blue, Gray)
MINION_YELLOW = "#FCE029"  # Minion Skin/Body
MINION_BLUE = "#0A75BC"     # Overalls
MINION_GRAY = "#949699"     # Goggles/Metals
MINION_DARK = "#231F20"     # Raisin Black (Glove/Shadows)

# Mapping for UI Enhancement: Icons and Emojis
EMOTION_ICONS = {
    "JOY": "😄",
    "SADNESS": "😢",
    "ANGER": "😡",
    "FEAR": "😨",
    "SURPRISE": "😮",
    "DISGUST": "🤢",
    "NEUTRAL": "😐"
}

# --- MINIONS CUSTOM CSS STYLING ---
def inject_custom_css():
    """Injects custom CSS for a bright, Minions-themed UI."""
    st.markdown(
        f"""
        <style>
        /* BASE THEME OVERRIDES (Light background, Minions colors) */
        .stApp {{
            background-color: #FFFFFF;
            color: {MINION_DARK};
        }}
        
        /* 1. Typography and Headings */
        h1, h2, h3, h4 {{
            color: {MINION_BLUE}; /* Blue for headings */
        }}
        
        /* 2. Main Title Styling */
        h1 {{
            border-bottom: 3px solid {MINION_YELLOW}; /* Thick yellow line */
            padding-bottom: 15px;
            font-family: 'Comic Sans MS', cursive, sans-serif; /* Playful font */
            text-shadow: 2px 2px {MINION_GRAY};
        }}
        
        /* 3. Streamlit Metrics (Emotion Summary Cards) */
        div[data-testid="stMetric"] {{
            background-color: {MINION_YELLOW}; /* Yellow background for cards */
            padding: 15px;
            border-radius: 15px; /* Rounded and chunky */
            border: 3px solid {MINION_BLUE}; /* Blue border */
            box-shadow: 5px 5px 0px {MINION_DARK}; /* Fun, thick shadow effect */
            transition: all 0.1s ease-in-out;
            transform: skew(-2deg); /* SLIGHT playful tilt */
        }}
        div[data-testid="stMetric"]:hover {{
            background-color: #FFEB63; 
            box-shadow: 2px 2px 0px {MINION_DARK};
            transform: skew(0deg); /* Straightens on hover */
        }}
        
        /* Adjust metric value/label colors */
        div[data-testid="stMetricLabel"] > div {{
            color: {MINION_DARK} !important;
            font-weight: 800;
        }}
        div[data-testid="stMetricValue"] {{
            color: {MINION_BLUE} !important;
            font-size: 2.2em;
            font-weight: 900;
        }}
        
        /* 4. Text Area and Input Styling */
        textarea {{
            background-color: #FFFFF0;
            color: {MINION_DARK};
            border-radius: 8px;
            border: 2px solid {MINION_GRAY};
        }}
        
        /* 5. Primary Button Styling (The big overall blue button!) */
        .stButton button {{
            background-color: {MINION_BLUE}; 
            color: {MINION_YELLOW}; /* Yellow text on blue */
            border-radius: 10px;
            border: 2px solid {MINION_DARK};
            padding: 10px 20px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: background-color 0.2s;
        }}
        .stButton button:hover {{
            background-color: #005F99;
            color: white;
            border-color: {MINION_YELLOW};
        }}

        /* 6. Tabs Styling */
        div[role="tablist"] button {{
            font-weight: 800;
            color: {MINION_DARK};
        }}
        div[data-testid="stTabs"] div[aria-selected="true"] button {{
            border-bottom-color: {MINION_YELLOW} !important; /* Active tab underline */
            color: {MINION_BLUE} !important;
        }}
        
        /* 7. Expander (Input Section) Styling */
        .streamlit-expanderHeader {{
            background-color: #F8F8F8;
            border-radius: 10px;
            border: 1px solid {MINION_GRAY};
            padding: 10px;
            font-weight: 600;
            color: {MINION_DARK};
        }}
        
        /* 8. Dataframe (Table) Styling */
        .stDataFrame {{
            border-radius: 8px;
            border: 2px solid {MINION_GRAY};
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Core Functions (Based on the original code, adapted for UI data) ---

@st.cache_resource
def initialize_classifier():
    """Initializes the Hugging Face emotion classification pipeline and caches it."""
    try:
        with st.spinner(f"Loading emotion classification model: {MODEL_NAME}... (This may take a moment on first run)"):
            classifier = pipeline(
                "text-classification",
                model=MODEL_NAME,
                return_all_scores=True
            )
        st.success("Model loaded successfully!")
        return classifier
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()
        return None

def detect_emotions(classifier, texts):
    """Analyzes a list of texts and returns the results for Streamlit."""
    if not texts:
        return []

    predictions = classifier(texts)
    results = []

    for text, prediction_list in zip(texts, predictions):
        best_prediction = max(prediction_list, key=lambda x: x['score'])
        
        dominant_emotion = best_prediction['label'].upper()
        icon = EMOTION_ICONS.get(dominant_emotion, "❓")
        
        # NOTE: Confidence Score must be a float for st.column_config.ProgressColumn
        row = {
            'Input Text': text,
            # UI Enhancement: Prepend icon to the emotion
            'Dominant Emotion': f"{icon} {dominant_emotion}",
            # Use float here for the progress bar
            'Confidence Score': float(f"{best_prediction['score']:.4f}"),
            # Added for internal count tracking
            'Raw Emotion Label': dominant_emotion 
        }

        # Include all scores for the Advanced tab
        for item in prediction_list:
            row[f"Score - {item['label'].upper()}"] = f"{item['score']:.4f}"

        results.append(row)

    return results

# --- Streamlit Application Layout (Utilizing the Minions Theme) ---

# Inject CSS at the very start
inject_custom_css()

# 1. Page Configuration and Title
st.set_page_config(
    page_title="Minions Emotion Detector",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("🍌 Minion Language Detector: POOPAYE!")
st.markdown(f"""
<span style='color:{MINION_BLUE}; font-weight: bold;'>Analyze the emotional tone of your text</span> using the **`{MODEL_NAME}`** model.
""", unsafe_allow_html=True)

# 2. Model Initialization (called once and cached)
classifier = initialize_classifier()
st.markdown("---")

# 3. Input Area (Styled by CSS)
with st.expander("📝 **BEE DO! Enter Text(s) for Analysis**", expanded=True):
    default_text = (
        "I am so incredibly happy and proud of what we achieved today!\n"
        "This is confusing; I need someone to clarify the instructions for step three.\n"
        "My heart is racing, I'm genuinely terrified of what might happen next."
    )

    col1, col2 = st.columns([4, 1])

    with col1:
        input_text = st.text_area(
            "Paste one or more sentences (each on a new line):",
            value=default_text,
            height=150,
            key="text_input",
            label_visibility="collapsed"
        )
    
    input_texts = [text.strip() for text in input_text.split('\n') if text.strip()]

    with col2:
        st.markdown("<br>", unsafe_allow_html=True) 
        analyze_button = st.button(
            "Analyze Texts", 
            type="primary", 
            use_container_width=True,
            help="Click to run the emotion detection model."
        )

st.markdown("---")

# 4. Analysis Logic and Results Display (Styled by CSS)
if analyze_button:
    if input_texts:
        st.subheader("🍌 Results: KAI-FU-RU!")
        with st.spinner(f"Analyzing {len(input_texts)} sentence(s)..."):
            detection_results = detect_emotions(classifier, input_texts)

            if detection_results:
                df = pd.DataFrame(detection_results)
                
                # --- Emotion Count Metrics (Utilizing the styled stMetric cards) ---
                st.markdown("#### Dominant Emotions Summary")
                
                emotion_counts = df['Raw Emotion Label'].value_counts()
                
                metric_container = st.container()
                cols = metric_container.columns(min(len(emotion_counts), 7)) 

                for i, (emotion, count) in enumerate(emotion_counts.items()):
                    if i < 7:
                        icon = EMOTION_ICONS.get(emotion, "")
                        cols[i].metric(
                            label=f"{icon} {emotion.capitalize()}", 
                            value=count
                        )

                st.markdown("---")
                
                # --- Tabular Results with Progress Bar ---
                st.markdown("#### Detailed Analysis Table")
                
                # Using tabs for cleaner organization
                tab_simple, tab_advanced = st.tabs(["Simple Results", "All Confidence Scores"])

                with tab_simple:
                    # Keep only essential columns for the simple view
                    simple_df = df.drop(columns=['Raw Emotion Label']) 
                    simple_df = simple_df[['Input Text', 'Dominant Emotion', 'Confidence Score']]
                    
                    st.dataframe(
                        simple_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Dominant Emotion": st.column_config.Column("Dominant Emotion", width="small"),
                            "Confidence Score": st.column_config.ProgressColumn(
                                "Confidence Score",
                                format="%.2f",
                                min_value=0.0,
                                max_value=1.0,
                            ),
                            "Input Text": st.column_config.TextColumn("Input Text", width="large")
                        }
                    )
                
                with tab_advanced:
                    # Drop UI helper columns for the raw data view
                    advanced_df = df.drop(columns=['Dominant Emotion', 'Confidence Score', 'Raw Emotion Label'])
                    st.dataframe(
                        advanced_df,
                        hide_index=True,
                        use_container_width=True
                    )

            else:
                st.warning("No valid text found to analyze.")
    else:
        st.warning("Please enter some text to analyze before clicking the button.")

st.markdown("---")
st.caption("Powered by Streamlit and Hugging Face Transformers. TULALILOO TI AMO!")

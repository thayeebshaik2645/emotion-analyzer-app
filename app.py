import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Configuration ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- RETRO CALM COLOR PALETTE ---
# (Kept for visual consistency)
COLOR_AQUA = "#81D8D0"     # Primary Button/Accent
COLOR_PEACH = "#D99E82"    # Secondary Accent/Card Background
COLOR_KHAKI = "#D7D982"    # Subtle Highlight/Light Border
COLOR_PURPLE = "#AE82D9"   # Main Title/Text Highlight
COLOR_DARK = "#333333"     # Text/Shadows

# --- GIF MAPPING (New Feature) ---
# Using external, common GIF URLs for animated expression
GIF_ICONS = {
    "JOY": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOHp1eXR6bW5qNjBna25rYjYxc2xwdmwwOXQ4c3RzYjB4OXVwMXd4dCZlcD12MV9pbnRlcm5hbF9naWYmY3Q9cw/WlDW2jYx9N5z2V1u60/giphy.gif", # Happy Face
    "SADNESS": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ2dpcjZzd2F5Z254enJ4b3Z5dG5wY2o3b284eXh5NWx0MWRwMXU4ZCZlcD12MV9pbnRlcm5hbF9naWYmY3Q9cw/wI8kC8eD124GvKjD0p/giphy.gif", # Crying Face
    "ANGER": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOXV0OHc2eGozN3Z0cW05Z2ZlYzZ0a2w3Nzhia2EwNWlvdG5sZmF1ZCZlcD12MV9pbnRlcm5hbF9naWYmY3Q9cw/Q7yH1m0c313zO/giphy.gif", # Angry Red Face
    "FEAR": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdTBqc3V0cm5td3V5YXF0czRtc2R0eW12MDBkdG1pZWcyY3Q0d3gxdSZlcD12MV9pbnRlcm5hbF9naWYmY3Q9cw/26FL35n484B6F00jC/giphy.gif", # Scared/Sweating
    "SURPRISE": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOWQ4eWVwM2Nwa204Y3h0N3Vob2U3NXBxZnd0eTR4ZHZ2Yjd1aWw4YSZlcD12MV9pbnRlcm5hbF9naWYmY3Q9cw/3o7TKM74MxC02VWrks/giphy.gif", # Mind Blown/Surprised
    "DISGUST": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExc29lYnQycm95bXN0bmFnb21sMXN4d3g0bmFhMW8xcjNyN3lwNDJ3biZlcD12MV9pbnRlcm5hbF9naWYmY3Q9cw/PWHKk7D4xH6U/giphy.gif", # Green Sick Face
    "NEUTRAL": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2R4d3N1OHY3N2k1eWpjaWdvbnY1eHp1Z281N2Q3NG41b2kwaWc3eCZlcD12MV9pbnRlcm5hbF9naWYmY3Q9cw/6u31D8I2aHjVq/giphy.gif" # Meh/Shrugging
}

# --- STATIC EMOJI MAPPING (Still needed for the main table) ---
EMOTION_EMOJIS = {
    "JOY": "😊",
    "SADNESS": "😔",
    "ANGER": "😠",
    "FEAR": "😨",
    "SURPRISE": "😲",
    "DISGUST": "🤢",
    "NEUTRAL": "😶"
}

# --- RETRO CALM CUSTOM CSS STYLING (The core UI remains the same) ---
def inject_custom_css():
    """Injects custom CSS for a Retro Calm, professional theme."""
    st.markdown(
        f"""
        <style>
        /* Base Styling */
        .stApp {{
            background-color: #FFFFFF;
            color: {COLOR_DARK};
            font-family: serif; 
        }}
        
        /* Headings */
        h1, h2, h3, h4 {{
            color: {COLOR_DARK}; 
            font-weight: 700;
        }}
        h1 {{
            color: {COLOR_PURPLE}; 
            border-bottom: 2px solid {COLOR_AQUA}; 
            padding-bottom: 15px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            letter-spacing: -1px; 
        }}
        
        /* Streamlit Metrics (Emotion Summary Cards) */
        div[data-testid="stMetric"] {{
            background-color: {COLOR_PEACH}; 
            padding: 15px;
            border-radius: 8px;
            border: 1px solid {COLOR_KHAKI}; 
            box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1); 
            transition: all 0.3s ease-in-out;
            /* Added margin-bottom to separate metric from the GIF */
            margin-bottom: 5px; 
        }}
        div[data-testid="stMetric"]:hover {{
            background-color: #E6B5A1;
            border-color: {COLOR_PURPLE};
        }}
        
        /* Metric Text Contrast */
        div[data-testid="stMetricLabel"] > div {{
            color: {COLOR_DARK} !important;
            font-weight: 600;
        }}
        div[data-testid="stMetricValue"] {{
            color: {COLOR_DARK} !important;
            font-size: 2em;
            font-weight: 700;
        }}
        
        /* Primary Button Styling (Aqua emphasis) */
        .stButton button {{
            background-color: {COLOR_AQUA}; 
            color: {COLOR_DARK}; 
            border-radius: 6px;
            border: none;
            padding: 10px 20px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
            transition: background-color 0.2s;
        }}
        .stButton button:hover {{
            background-color: #63C7BC;
            color: white;
        }}

        /* Tabs Styling */
        div[role="tablist"] button {{
            font-weight: 600;
            color: {COLOR_DARK};
        }}
        div[data-testid="stTabs"] div[aria-selected="true"] button {{
            border-bottom-color: {COLOR_AQUA} !important; 
            color: {COLOR_PURPLE} !important; 
        }}
        
        /* General Box Styling */
        .streamlit-expanderHeader, .stDataFrame {{
            border-radius: 6px;
            border: 1px solid {COLOR_KHAKI};
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Core Functions ---

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
        # Use static emoji for the clean table view
        emoji = EMOTION_EMOJIS.get(dominant_emotion, "❓")
        
        row = {
            'Input Text': text,
            'Dominant Emotion': f"{emoji} {dominant_emotion}",
            'Confidence Score': float(f"{best_prediction['score']:.4f}"),
            'Raw Emotion Label': dominant_emotion 
        }

        for item in prediction_list:
            row[f"Score - {item['label'].upper()}"] = f"{item['score']:.4f}"

        results.append(row)

    return results

# --- Streamlit Application Layout (Applying GIF Enhancement) ---

# Inject CSS
inject_custom_css()

# 1. Page Configuration and Title
st.set_page_config(
    page_title="Retro Calm Emotion Detector",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("Retro Calm Emotion Analyzer")
st.markdown(f"""
<span style='color:{COLOR_DARK};'>This tool uses the pre-trained Hugging Face model **`{MODEL_NAME}`** to classify emotions in text.</span>
""", unsafe_allow_html=True)

# 2. Model Initialization
classifier = initialize_classifier()
st.markdown("---")

# 3. Input Area
with st.expander("📝 **Enter Text(s) for Analysis**", expanded=True):
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

# 4. Analysis Logic and Results Display
if analyze_button:
    if input_texts:
        st.subheader("Analysis Results")
        with st.spinner(f"Analyzing {len(input_texts)} sentence(s)..."):
            detection_results = detect_emotions(classifier, input_texts)

            if detection_results:
                df = pd.DataFrame(detection_results)
                
                # --- GIF EMOTION COUNT SUMMARY (Enhanced UI with GIFs) ---
                st.markdown("#### Dominant Emotions Summary (Animated)")
                
                emotion_counts = df['Raw Emotion Label'].value_counts()
                
                metric_container = st.container()
                cols = metric_container.columns(min(len(emotion_counts), 7)) 

                for i, (emotion, count) in enumerate(emotion_counts.items()):
                    if i < 7:
                        # 1. Display the count metric (styled card)
                        cols[i].metric(
                            label=f"{emotion.capitalize()}", 
                            value=count
                        )
                        # 2. Display the GIF beneath the metric (NEW FEATURE)
                        gif_url = GIF_ICONS.get(emotion, GIF_ICONS['NEUTRAL'])
                        cols[i].image(
                            gif_url,
                            width=50, # Set a small, fixed size for the GIF
                            caption=None # Hide the caption
                        )

                st.markdown("---")
                
                # --- Tabular Results with Progress Bar ---
                st.markdown("#### Detailed Analysis Table")
                
                tab_simple, tab_advanced = st.tabs(["Simple Results", "All Confidence Scores"])

                with tab_simple:
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
st.caption("Application powered by Streamlit and Hugging Face Transformers. Animated icons via Giphy.")

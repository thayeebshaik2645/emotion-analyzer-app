import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Configuration ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

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

# --- CUSTOM CSS STYLING ---
def inject_custom_css():
    """Injects custom CSS to enhance the Streamlit UI."""
    st.markdown(
        """
        <style>
        /* 1. Base Font and Background */
        html, body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        }
        
        /* 2. Main Title Styling */
        .stApp > header {
            background-color: transparent; /* Remove header background */
        }
        h1 {
            color: #1E90FF; /* Dodger Blue for the title */
            border-bottom: 2px solid #EEEEEE;
            padding-bottom: 10px;
        }
        
        /* 3. Streamlit Metrics (for Emotion Summary) */
        div[data-testid="stMetric"] {
            background-color: #F8F9FA;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #E9ECEF;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #1E90FF; /* Highlight on hover */
        }
        
        /* 4. Text Area Styling */
        textarea {
            border-radius: 8px;
            border: 1px solid #CED4DA;
        }
        
        /* 5. Primary Button Styling */
        .stButton button {
            background-color: #1E90FF;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.2s;
        }
        .stButton button:hover {
            background-color: #0077CC;
        }

        /* 6. Tabs Styling */
        div[role="tablist"] button {
            font-weight: bold;
            color: #495057;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Core Functions (kept the same) ---

@st.cache_resource
def initialize_classifier():
    """Initializes the Hugging Face emotion classification pipeline and caches it."""
    try:
        with st.spinner(f"Loading emotion classification model: {MODEL_NAME}..."):
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
        
        row = {
            'Input Text': text,
            'Dominant Emotion': f"{icon} {dominant_emotion}",
            'Confidence Score': float(f"{best_prediction['score']:.4f}"), # Convert to float for ProgressColumn
            'Raw Emotion Label': dominant_emotion 
        }

        for item in prediction_list:
            row[f"Score - {item['label'].upper()}"] = f"{item['score']:.4f}"

        results.append(row)

    return results

# --- Streamlit Application Layout (Incorporating CSS) ---

# Inject CSS at the very start
inject_custom_css()

# 1. Page Configuration and Title
st.set_page_config(
    page_title="AI Emotion Detector",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={'About': "Emotion Analyzer App using Hugging Face Transformers and Streamlit."}
)

st.title("🗣️ AI Emotion Analyzer")
st.markdown(f"""
Analyze the emotional tone of your text using the **`{MODEL_NAME}`** model.
""")

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
        st.subheader("📊 Analysis Results")
        with st.spinner(f"Analyzing {len(input_texts)} sentence(s)..."):
            detection_results = detect_emotions(classifier, input_texts)

            if detection_results:
                df = pd.DataFrame(detection_results)
                
                # --- Emotion Count Metrics (Styled by CSS) ---
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
                
                tab_simple, tab_advanced = st.tabs(["Simple Results", "All Confidence Scores"])

                with tab_simple:
                    simple_df = df.drop(columns=['Raw Emotion Label']) 
                    simple_df = simple_df[['Input Text', 'Dominant Emotion', 'Confidence Score']]
                    
                    st.dataframe(
                        simple_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Dominant Emotion": st.column_config.Column(
                                "Dominant Emotion",
                                width="small"
                            ),
                            "Confidence Score": st.column_config.ProgressColumn(
                                "Confidence Score",
                                format="%.2f",
                                min_value=0.0,
                                max_value=1.0,
                            ),
                            "Input Text": st.column_config.TextColumn(
                                "Input Text",
                                width="large"
                            )
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
st.caption("Powered by Streamlit and Hugging Face Transformers.")

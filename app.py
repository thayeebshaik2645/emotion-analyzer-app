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

# --- AGGRESSIVE CUSTOM CSS STYLING (Dark Theme Focus) ---
def inject_custom_css():
    """Injects custom CSS for a dark, professional theme."""
    st.markdown(
        """
        <style>
        /* BASE DARK THEME OVERRIDES */
        /* Set the main background to a dark color */
        .stApp {
            background-color: #121212; /* Very dark gray */
            color: #E0E0E0; /* Light text color */
        }
        
        /* Ensure main content background matches */
        section.main {
            background-color: #121212;
        }

        /* 1. Typography and Headings */
        h1, h2, h3, h4 {
            color: #BB86FC; /* Primary accent color (Purple) */
        }
        
        /* 2. Main Title Styling */
        h1 {
            border-bottom: 2px solid #2C2C2C; /* Darker line for separation */
            padding-bottom: 15px;
        }
        
        /* 3. Streamlit Metrics (Emotion Summary Cards) */
        div[data-testid="stMetric"] {
            background-color: #1E1E1E; /* Darker background for cards */
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #2C2C2C;
            box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4); /* Stronger shadow */
            transition: all 0.3s ease-in-out;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #BB86FC; /* Highlight on hover */
            background-color: #242424;
        }
        
        /* Adjust metric label/value colors for dark theme contrast */
        div[data-testid="stMetricLabel"] > div {
            color: #90CAF9 !important; /* Light Blue for labels */
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-size: 2em;
        }
        
        /* 4. Text Area and Input Styling */
        textarea {
            background-color: #1E1E1E;
            color: #E0E0E0;
            border-radius: 8px;
            border: 1px solid #3A3A3A;
        }
        
        /* 5. Primary Button Styling */
        .stButton button {
            background-color: #03DAC6; /* Secondary accent color (Teal) */
            color: #121212; /* Dark text for contrast */
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.2s;
        }
        .stButton button:hover {
            background-color: #018786;
            color: white;
        }

        /* 6. Tabs Styling */
        div[role="tablist"] button {
            font-weight: bold;
            color: #E0E0E0;
        }
        div[data-testid="stTabs"] div[aria-selected="true"] button {
            border-bottom-color: #BB86FC !important; /* Active tab underline */
            color: #BB86FC !important;
        }
        
        /* 7. Expander (Input Section) Styling */
        .streamlit-expanderHeader {
            background-color: #1E1E1E;
            border-radius: 8px;
            border: 1px solid #2C2C2C;
            padding: 10px;
        }
        
        /* 8. Dataframe (Table) Styling */
        .stDataFrame {
            border-radius: 8px;
            border: 1px solid #2C2C2C;
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

# --- Streamlit Application Layout (Applying New CSS) ---

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
<span style='color:#BB86FC;'>Analyze the emotional tone of your text</span> using the **`{MODEL_NAME}`** model.
""", unsafe_allow_html=True) # Use span for a little color accent

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
                # Ensure the number of columns doesn't exceed 7 for clean layout
                cols = metric_container.columns(min(len(emotion_counts), 7)) 

                for i, (emotion, count) in enumerate(emotion_counts.items()):
                    if i < 7:
                        icon = EMOTION_ICONS.get(emotion, "")
                        # The CSS targets stMetric, making these boxes look great
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
                                # Use the accent color for the progress bar
                                # Note: ProgressColumn color can only be set globally in the theme or using HTML/custom components, 
                                # but the dark background improves its visibility significantly.
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

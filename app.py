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

# --- Core Functions (kept mostly the same for functionality) ---

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
        
        # Prepare the output row, including the icon for UI
        dominant_emotion = best_prediction['label'].upper()
        icon = EMOTION_ICONS.get(dominant_emotion, "❓")
        
        row = {
            'Input Text': text,
            'Dominant Emotion': f"{icon} {dominant_emotion}", # Enhanced UI column
            'Confidence Score': f"{best_prediction['score']:.4f}",
            'Raw Emotion Label': dominant_emotion # Keep raw label for filtering/internal logic
        }

        # Add all scores for the advanced view
        for item in prediction_list:
            row[f"Score - {item['label'].upper()}"] = f"{item['score']:.4f}"

        results.append(row)

    return results

# --- Streamlit Application Layout (UI CHANGES) ---

# 1. Page Configuration and Title
st.set_page_config(
    page_title="Text Emotion Detector",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "Emotion Analyzer App using Hugging Face Transformers and Streamlit."
    }
)

st.title("🗣️ AI Emotion Analyzer")
st.markdown(f"""
Analyze the emotional tone of your text using the **`{MODEL_NAME}`** model.
""")

# 2. Model Initialization
classifier = initialize_classifier()
st.markdown("---")

# 3. Input Area (Organized using an expander)
with st.expander("📝 **Enter Text(s) for Analysis**", expanded=True):
    default_text = (
        "I am so incredibly happy and proud of what we achieved today!\n"
        "This is confusing; I need someone to clarify the instructions for step three.\n"
        "My heart is racing, I'm genuinely terrified of what might happen next."
    )

    # Use st.columns for better button placement
    col1, col2 = st.columns([4, 1])

    with col1:
        input_text = st.text_area(
            "Paste one or more sentences (each on a new line):",
            value=default_text,
            height=150,
            key="text_input",
            label_visibility="collapsed" # Hide label for cleaner look
        )
    
    # Prepare texts list, filtering out empty lines
    input_texts = [text.strip() for text in input_text.split('\n') if text.strip()]

    with col2:
        # Align the button nicely with the text area
        st.markdown("<br>", unsafe_allow_html=True) # Small vertical space
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
                
                # --- UI Enhancement: Emotion Count Metrics (More Visually appealing) ---
                st.markdown("#### Dominant Emotions Summary")
                
                # Get the counts from the raw label column for better grouping/sorting
                emotion_counts = df['Raw Emotion Label'].value_counts()
                
                # Use a container to hold the metrics
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
                
                # --- UI Enhancement: Tabular Results with two views ---
                st.markdown("#### Detailed Analysis Table")
                
                # Use tabs to organize the two result views
                tab_simple, tab_advanced = st.tabs(["Simple Results", "All Confidence Scores"])

                with tab_simple:
                    # Drop the raw label column before displaying the final table
                    simple_df = df.drop(columns=['Raw Emotion Label']) 
                    simple_df = simple_df[['Input Text', 'Dominant Emotion', 'Confidence Score']]
                    
                    st.dataframe(
                        simple_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Dominant Emotion": st.column_config.Column(
                                "Dominant Emotion",
                                help="The emotion with the highest confidence score (includes emoji for quick identification).",
                                width="small" # Make this column smaller
                            ),
                            "Confidence Score": st.column_config.ProgressColumn(
                                "Confidence Score",
                                help="Model's certainty (0.00 to 1.00)",
                                format="%.2f",
                                min_value=0.0,
                                max_value=1.0,
                            ),
                            "Input Text": st.column_config.TextColumn(
                                "Input Text",
                                help="The text that was analyzed.",
                                width="large"
                            )
                        }
                    )
                
                with tab_advanced:
                    # Drop the UI-specific columns and keep all score columns
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

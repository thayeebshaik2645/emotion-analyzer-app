import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Configuration ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Use Streamlit's caching to load the model only once
# This is crucial for performance in a web app
@st.cache_resource(show_spinner=False)
def initialize_classifier():
    """Initializes the Hugging Face emotion classification pipeline and caches it."""
    try:
        # Show a loading message while the model is downloading/loading
        with st.spinner(f"Loading emotion model: **{MODEL_NAME}**..."):
            classifier = pipeline(
                "text-classification",
                model=MODEL_NAME,
                return_all_scores=True
            )
        st.success("✅ Model loaded successfully!")
        return classifier
    except Exception as e:
        st.error(f"❌ Error loading the model: {e}")
        st.warning("Ensure you have the required libraries installed (`pip install transformers torch`).")
        st.stop()
        return None

def detect_emotions(classifier, texts):
    """Analyzes a list of texts and returns the results for Streamlit."""
    if not texts:
        return []

    # Process texts using the pipeline
    predictions = classifier(texts)
    results = []

    for text, prediction_list in zip(texts, predictions):
        # Find the highest scoring emotion
        best_prediction = max(prediction_list, key=lambda x: x['score'])

        # Prepare the output row for a DataFrame
        results.append({
            'Input Text': text,
            'Dominant Emotion': best_prediction['label'].capitalize(), # Use capitalize for a cleaner look
            'Confidence Score': f"{best_prediction['score']:.4f}"
        })

    return results

# --- Streamlit Application Layout ---

# 1. Page Configuration and Title
st.set_page_config(
    page_title="Emotion Analyzer",
    layout="wide"
)

st.title("🗣️ Simple Text Emotion Analyzer")
st.markdown(f"""
This tool uses a powerful **Transformer model** to classify the dominant emotion in your text.
Model used: **`{MODEL_NAME}`** (cached for fast repeated use).
""")

st.divider()

# 2. Model Initialization (called once and cached)
classifier = initialize_classifier()

# 3. Input Area
st.subheader("1. Enter Your Text(s)")
st.caption("Enter each sentence on a new line.")

default_text = (
    "I am so incredibly happy and proud of what we achieved today!\n"
    "This is confusing; I need someone to clarify the instructions.\n"
    "My heart is racing, I'm genuinely terrified of what might happen next."
)

input_text = st.text_area(
    "Text Input:",
    value=default_text,
    height=150,
    key="text_input",
    label_visibility="collapsed" # Hide the label to simplify the look
)

# Prepare texts list, filtering out empty lines
input_texts = [text.strip() for text in input_text.split('\n') if text.strip()]

# 4. Trigger Button
# The use of 'use_container_width=True' and centering helps the simple aesthetic
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("✨ Analyze Emotions ✨", type="primary", use_container_width=True)

st.markdown("---")

# 5. Analysis Logic and Results Display
if analyze_button:
    if input_texts:
        st.subheader("2. Analysis Results")
        with st.spinner(f"Processing {len(input_texts)} sentence(s)..."):
            # Perform Detection
            detection_results = detect_emotions(classifier, input_texts)

            # Display Results in a Streamlit DataFrame
            if detection_results:
                df = pd.DataFrame(detection_results)
                
                # Simple Styling for Dominant Emotion column
                def color_emotion(val):
                    color_map = {
                        'joy': '#C3E6CB', # light green
                        'sadness': '#B8DAFF', # light blue
                        'anger': '#F5C6CB', # light red
                        'fear': '#FFEBA8', # light yellow/gold
                        'surprise': '#B3CDE3', # a different light blue
                        'love': '#F8D7DA', # pinkish
                        'default': 'white'
                    }
                    lower_val = val.lower()
                    background = color_map.get(lower_val, color_map['default'])
                    return f'background-color: {background}'

                # Apply color styling only to the Dominant Emotion column (Streamlit does not support
                # this type of styling directly on st.dataframe without using st.table or st.data_editor
                # with custom CSS/styling functions, but a simple DataFrame view is cleaner)
                # We'll stick to a clean, un-styled st.dataframe for simplicity.

                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.warning("⚠️ No valid text found to analyze. Please check your input.")
    else:
        st.warning("💡 Please enter some text to analyze before clicking the button.")

st.divider()
st.info("Powered by Streamlit and Hugging Face Transformers. ")

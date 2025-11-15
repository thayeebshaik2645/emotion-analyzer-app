import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Configuration ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Use Streamlit's caching to load the model only once
# This is crucial for performance in a web app
@st.cache_resource
def initialize_classifier():
    """Initializes the Hugging Face emotion classification pipeline and caches it."""
    try:
        # Show a loading message while the model is downloading/loading
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
        st.warning("Please check your internet connection and library installations (`pip install transformers torch`).")
        st.stop() # Stop the app if model fails to load
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
            'Dominant Emotion': best_prediction['label'].upper(),
            'Confidence Score': f"{best_prediction['score']:.4f}"
        })

    return results

# --- Streamlit Application Layout ---

# 1. Page Configuration and Title
st.set_page_config(
    page_title="Text Emotion Detector",
    layout="wide"
)

st.title("🧠 Transformer-Based Text Emotion Analyzer")
st.markdown(f"""
This tool uses the pre-trained Hugging Face model **`{MODEL_NAME}`** to classify emotions in text.
""")

# 2. Model Initialization (called once and cached)
classifier = initialize_classifier()

# 3. Input Area
st.subheader("1. Enter Text(s) to Analyze")

default_text = (
    "I am so incredibly happy and proud of what we achieved today!\n"
    "This is confusing; I need someone to clarify the instructions for step three.\n"
    "My heart is racing, I'm genuinely terrified of what might happen next."
)

input_text = st.text_area(
    "Paste one or more sentences (each on a new line):",
    value=default_text,
    height=200,
    key="text_input"
)

# Prepare texts list, filtering out empty lines
input_texts = [text.strip() for text in input_text.split('\n') if text.strip()]

# 4. Trigger Button
analyze_button = st.button("Analyze Emotions", type="primary")

# 5. Analysis Logic and Results Display
if analyze_button:
    if input_texts:
        st.subheader("2. Analysis Results")
        with st.spinner(f"Analyzing {len(input_texts)} sentence(s)..."):
            # Perform Detection
            detection_results = detect_emotions(classifier, input_texts)

            # Display Results in a Streamlit DataFrame (like a nice table)
            if detection_results:
                df = pd.DataFrame(detection_results)
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    # Optional: Add color to the Emotion column
                    column_config={
                        "Dominant Emotion": st.column_config.Column(
                            "Dominant Emotion",
                            # Customize styling based on content if desired
                        ),
                    }
                )
            else:
                st.warning("No valid text found to analyze.")
    else:
        st.warning("Please enter some text to analyze before clicking the button.")

st.markdown("---")
st.caption("Application powered by Streamlit and Hugging Face Transformers.") 

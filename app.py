import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Configuration ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Placeholder GIF/Image URLs (Using placehold.co to represent dynamic content)
# In a real app, this would be populated by a Giphy/Tenor API call based on the detected emotion.
IMAGE_MAP = {
    'JOY': "https://placehold.co/150x150/5cb85c/ffffff?text=JOY+%28GIF+Placeholder%29",      # Green for happiness
    'SADNESS': "https://placehold.co/150x150/337ab7/ffffff?text=SADNESS+%28GIF+Placeholder%29",  # Blue for sadness
    'ANGER': "https://placehold.co/150x150/d9534f/ffffff?text=ANGER+%28GIF+Placeholder%29",      # Red for anger
    'FEAR': "https://placehold.co/150x150/f0ad4e/ffffff?text=FEAR+%28GIF+Placeholder%29",        # Yellow/Orange for fear
    'SURPRISE': "https://placehold.co/150x150/5bc0de/ffffff?text=SURPRISE+%28GIF+Placeholder%29",# Light Blue for surprise
    'LOVE': "https://placehold.co/150x150/8a39a0/ffffff?text=LOVE+%28GIF+Placeholder%29",        # Purple for love
    'DEFAULT': "https://placehold.co/150x150/cccccc/000000?text=Unknown+Emotion"
}

# --- Model Initialization and Caching ---
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
        st.stop()
        return None

def detect_emotions(classifier, texts):
    """Analyzes a list of texts and returns the results for Streamlit."""
    if not texts:
        return []

    # Process texts using the pipeline
    predictions = classifier(texts)
    results = []

    # Track the overall dominant emotion across all sentences to display one main GIF
    overall_emotions = {}
    
    for text, prediction_list in zip(texts, predictions):
        # Find the highest scoring emotion
        best_prediction = max(prediction_list, key=lambda x: x['score'])
        dominant_label = best_prediction['label'].upper()
        
        # Count the occurrences of each emotion
        overall_emotions[dominant_label] = overall_emotions.get(dominant_label, 0) + 1

        # Prepare the output row for a DataFrame
        results.append({
            'Input Text': text,
            'Dominant Emotion': dominant_label,
            'Confidence Score': f"{best_prediction['score']:.4f}"
        })

    # Determine the most frequently occurring emotion for the main GIF
    if overall_emotions:
        most_frequent_emotion = max(overall_emotions, key=overall_emotions.get)
        main_image_url = IMAGE_MAP.get(most_frequent_emotion, IMAGE_MAP['DEFAULT'])
    else:
        main_image_url = IMAGE_MAP['DEFAULT']
        
    return results, main_image_url

# --- Streamlit Application Layout ---

# 1. Page Configuration and Title
st.set_page_config(
    page_title="Text Emotion Detector with Visuals",
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
        st.subheader("2. Analysis Results and Visual Feedback")
        
        # Create columns for the results table and the GIF/Image
        col_results, col_gif = st.columns([3, 1])
        
        with st.spinner(f"Analyzing {len(input_texts)} sentence(s)..."):
            # Perform Detection
            detection_results, main_image_url = detect_emotions(classifier, input_texts)

            # Display Results in the left column (Table)
            with col_results:
                if detection_results:
                    df = pd.DataFrame(detection_results)
                    st.dataframe(
                        df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Dominant Emotion": st.column_config.Column(
                                "Dominant Emotion",
                            ),
                            # Adding progress bar for confidence for better visual representation
                            "Confidence Score": st.column_config.ProgressColumn(
                                "Confidence Score",
                                format="%.4f",
                                min_value=0,
                                max_value=1.0,
                            ),
                        }
                    )
                else:
                    st.warning("No valid text found to analyze.")

            # Display the main GIF/Image in the right column
            with col_gif:
                st.markdown("##### Visual Feedback")
                st.image(
                    main_image_url, 
                    caption="Most Frequent Emotion Visual", 
                    use_column_width=True
                )
                st.caption("*(This URL is a placeholder simulating a GIF based on the most common emotion detected.)*")


    else:
        st.warning("Please enter some text to analyze before clicking the button.")

st.markdown("---")
st.caption("Application powered by Streamlit and Hugging Face Transformers.")

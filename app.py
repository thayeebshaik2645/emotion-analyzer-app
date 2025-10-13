import streamlit as st
import pandas as pd
from transformers import pipeline
import io # Needed for handling file uploads

# --- Configuration ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Mapping for simple color-coding in the results
# You can expand this to all your model's labels
EMOTION_COLORS = {
    "JOY": "green",
    "SADNESS": "blue",
    "ANGER": "red",
    "FEAR": "purple",
    "SURPRISE": "orange",
    "DISGUST": "brown",
    "NEUTRAL": "gray"
}

# Use Streamlit's caching to load the model only once
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

    # Create a column for all scores (new feature for advanced analysis)
    all_scores = {}
    
    for text, prediction_list in zip(texts, predictions):
        # Find the highest scoring emotion
        best_prediction = max(prediction_list, key=lambda x: x['score'])
        
        # Initialize the dictionary for the row
        row = {
            'Input Text': text,
            'Dominant Emotion': best_prediction['label'].upper(),
            'Confidence Score': f"{best_prediction['score']:.4f}"
        }
        
        # Populate the score for all emotions for the new advanced view
        for item in prediction_list:
            score_key = f"Score - {item['label'].upper()}"
            row[score_key] = f"{item['score']:.4f}"
            
        results.append(row)

    return results

# --- NEW FEATURE: File Handling ---
def process_uploaded_file(uploaded_file):
    """Reads texts from a supported uploaded file (TXT or CSV)."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    texts = []

    try:
        if file_extension == 'txt':
            # Read all lines from the text file
            string_data = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            texts = [line.strip() for line in string_data.split('\n') if line.strip()]
        elif file_extension == 'csv':
            # Read the CSV. Assume the text is in a column named 'text' or 'sentence'
            df = pd.read_csv(uploaded_file)
            
            # Auto-detect the text column
            potential_columns = [col for col in df.columns if 'text' in col.lower() or 'sentence' in col.lower()]
            if not potential_columns:
                st.error("CSV file must contain a column named 'text' or 'sentence' for analysis.")
                return []
            
            # Use the first matching column
            text_column = potential_columns[0]
            texts = df[text_column].astype(str).tolist()
            # Filter out non-string/empty entries
            texts = [t.strip() for t in texts if t and str(t).strip()]
        else:
            st.error("Unsupported file type. Please upload a **.txt** or **.csv** file.")
            return []
            
        return texts
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return []

# --- Streamlit Application Layout (ENHANCED UI) ---

# 1. Page Configuration and Title
st.set_page_config(
    page_title="Advanced Text Emotion Detector",
    layout="wide", # Use full width for a cleaner look
    initial_sidebar_state="auto"
)

st.title("🧠 Advanced Transformer-Based Emotion Analyzer")
st.markdown(f"""
This tool uses the pre-trained Hugging Face model **`{MODEL_NAME}`** to classify emotions in text.
""")
st.markdown("---")

# 2. Model Initialization (called once and cached)
classifier = initialize_classifier()

# --- Input Section (Tabs for UI enhancement) ---

st.subheader("1. Input Data")

# Use tabs to organize the two input methods
tab_text, tab_file = st.tabs(["📝 Manual Text Input", "📁 Bulk File Upload"])

input_texts = []
analyze_button_key = "analyze_text_button"

with tab_text:
    st.markdown("Enter one or more sentences below (each on a new line).")
    default_text = (
        "I am so incredibly happy and proud of what we achieved today!\n"
        "This is confusing; I need someone to clarify the instructions for step three.\n"
        "My heart is racing, I'm genuinely terrified of what might happen next."
    )

    input_text = st.text_area(
        "Text Input:",
        value=default_text,
        height=200,
        key="text_input"
    )

    # Prepare texts list from manual input
    input_texts = [text.strip() for text in input_text.split('\n') if text.strip()]
    analyze_button = st.button("Analyze Manual Text", type="primary", key="manual_analyze")

with tab_file:
    st.markdown("Upload a **.txt** file (one sentence per line) or a **.csv** file (with a 'text' or 'sentence' column).")
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'], key="file_uploader")
    
    # Process file input if available
    if uploaded_file is not None:
        input_texts = process_uploaded_file(uploaded_file)
        # Display a summary of the uploaded data
        if input_texts:
            st.info(f"Successfully loaded **{len(input_texts)}** lines/rows from the file.")
            analyze_button = st.button("Analyze Uploaded File", type="primary", key="file_analyze")
        else:
             # Prevent analysis if file failed to process
            analyze_button = st.button("Analyze Uploaded File", type="primary", key="file_analyze_disabled", disabled=True)
    else:
        # Placeholder for the button when no file is uploaded
        analyze_button = st.button("Analyze Uploaded File", type="primary", key="file_analyze_placeholder", disabled=True)

st.markdown("---")

# 3. Analysis Logic and Results Display
if analyze_button:
    if input_texts:
        st.subheader("2. Analysis Results")
        with st.spinner(f"Analyzing {len(input_texts)} sentence(s)..."):
            # Perform Detection
            detection_results = detect_emotions(classifier, input_texts)

            if detection_results:
                df = pd.DataFrame(detection_results)
                
                # --- UI Enhancement: Metric for Emotion Count ---
                emotion_counts = df['Dominant Emotion'].value_counts()
                st.markdown("#### Summary Metrics")
                
                # Create a row of columns for emotion counts
                cols = st.columns(min(len(emotion_counts), 5)) 
                
                for i, (emotion, count) in enumerate(emotion_counts.items()):
                    if i < 5: # Limit to 5 for the metric display
                        cols[i].metric(
                            label=emotion.capitalize(), 
                            value=count, 
                            delta=None
                        )
                
                st.markdown("---")
                
                # --- UI Enhancement: Tabular Results and Advanced View ---
                st.markdown("#### Detailed Analysis")
                
                # Create two tabs for the results
                tab_simple, tab_advanced = st.tabs(["Simple View", "Advanced Scores"])

                with tab_simple:
                    # Keep only the essential columns for the simple view
                    simple_df = df[['Input Text', 'Dominant Emotion', 'Confidence Score']]
                    st.dataframe(
                        simple_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Dominant Emotion": st.column_config.Column(
                                "Dominant Emotion",
                                help="The emotion with the highest score.",
                                # Use color coding in the simple view (if supported by Streamlit version)
                                # The coloring is not directly supported in st.dataframe column_config by emotion content, 
                                # so we rely on the clean table view. 
                                # A better visualization might be a dedicated chart.
                            ),
                        }
                    )
                    
                with tab_advanced:
                    # Show all scores for advanced users
                    st.dataframe(
                        df,
                        hide_index=True,
                        use_container_width=True
                    )

                # --- NEW FEATURE: Download Button ---
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Results as CSV",
                    data=csv,
                    file_name='emotion_analysis_results.csv',
                    mime='text/csv',
                    help="Download the full table, including all confidence scores."
                )

            else:
                st.warning("No valid text found to analyze after processing.")
    else:
        st.warning("Please enter some text or upload a valid file to analyze before clicking the button.")

st.markdown("---")
st.caption("Application powered by Streamlit and Hugging Face Transformers.")

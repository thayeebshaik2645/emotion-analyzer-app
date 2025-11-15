import streamlit as st
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
PAGE_ICON_URL = "https://cdn-icons-png.flaticon.com/128/10479/10479785.png"


# --- EMOTION GIF MAPPING ---
EMOTION_GIFS = {
    "ANGER": "https://media0.giphy.com/media/jsNiI5nMGQurggwpkN/giphy.webp",
    "HAPPINESS": "https://media4.giphy.com/media/USR9bpLz899PYVHk7C/giphy.webp",
    "SADNESS": "https://media0.giphy.com/media/StAnQV9TUCuys/giphy.webp",
    "JOY": "https://media0.giphy.com/media/LN5bH1r7UEpSRbcN7M/giphy.webp",
    "FEAR": "https://media0.giphy.com/media/Gl7mfimOjkkGl5mMDS/giphy.webp",
    "NEUTRAL": "https://media3.giphy.com/media/7CXIO53h5YciXOp505/giphy.webp",
    "DISGUST": "https://media0.giphy.com/media/jsNiI5nMGQurggwpkN/giphy.webp",
    "EXCITEMENT": "https://media0.giphy.com/media/LN5bH1r7UEpSRbcN7M/giphy.webp",
    "LONELINESS": "https://media0.giphy.com/media/StAnQV9TUCuys/giphy.webp",
    "SURPRISE": "https://media0.giphy.com/media/Gl7mfimOjkkGl5mMDS/giphy.webp",
}


# --- PAGE SETUP ---
st.set_page_config(
    page_title="STRANGER THINGS ANALYZER",
    page_icon=PAGE_ICON_URL,
    layout="wide",
)


# ===========================
#     STRANGER THINGS CSS
# ===========================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;900&family=Roboto+Mono:wght@300;400;700&display=swap');

:root {
    --st-red: #e50914;
    --st-red-glow: #ff1d2d;
    --st-bg-dark: #000000;
    --text-light: #e3e3e3;
}

/* Background */
.main {
    background: radial-gradient(circle at top, #1b0006 0%, #000000 70%);
    color: var(--text-light);
    font-family: 'Roboto Mono', monospace;
}

/* Fog layer */
.main::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: url("https://i.imgur.com/8fK4hFf.png");
    opacity: 0.08;
    animation: drift 60s linear infinite;
}

@keyframes drift {
    from { transform: translateX(0); }
    to   { transform: translateX(20%); }
}

/* Title */
h1 {
    font-family: 'Cinzel', serif;
    color: var(--st-red);
    font-size: 4rem;
    font-weight: 900;
    text-shadow: 0 0 20px var(--st-red-glow), 0 0 40px var(--st-red-glow);
    letter-spacing: 4px;
}

/* Subtitle */
h3 {
    font-family: 'Cinzel', serif;
    color: var(--st-red);
    text-shadow: 0 0 10px var(--st-red-glow);
}

/* Text area */
textarea {
    background: #0d0d0d !important;
    color: var(--text-light) !important;
    border: 2px solid var(--st-red) !important;
    box-shadow: 0 0 10px var(--st-red-glow);
}

/* Buttons */
div.stButton > button {
    background: var(--st-red);
    border: none;
    color: white;
    font-family: 'Cinzel', serif;
    font-weight: 700;
    padding: 0.7em 1.7em;
    letter-spacing: 2px;
    box-shadow: 0 0 20px var(--st-red-glow);
    transition: 0.2s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 35px var(--st-red-glow);
}

/* Result container */
.result-card {
    background: rgba(15, 0, 0, 0.75);
    border-left: 5px solid var(--st-red);
    padding: 16px;
    margin-bottom: 20px;
    border-radius: 6px;
    box-shadow: 0 0 25px rgba(255,0,0,0.4), inset 0 0 10px rgba(255,0,0,0.2);
    backdrop-filter: blur(3px);
}

/* Emotion Label */
.result-emotion {
    background: var(--st-red);
    padding: 5px 12px;
    border-radius: 4px;
    color: white;
    font-family: 'Cinzel', serif;
    font-weight: 900;
    text-shadow: 0 0 10px var(--st-red-glow);
}

/* GIF bubble */
.emotion-gif {
    width: 48px;
    height: 48px;
    border: 3px solid var(--st-red);
    border-radius: 50%;
    box-shadow: 0 0 12px var(--st-red-glow);
}

footer, header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)


# ======================================
#       MODEL LOADING
# ======================================

@st.cache_resource
def initialize_classifier():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        return_all_scores=True
    )

classifier = initialize_classifier()


def detect_emotions(classifier, texts):
    predictions = classifier(texts)
    results = []
    for text, prediction_list in zip(texts, predictions):
        best = max(prediction_list, key=lambda x: x["score"])
        results.append({
            "Input Text": text,
            "Dominant Emotion": best["label"].upper(),
            "Confidence": f"{best['score']:.4f}",
        })
    return results


# ======================================
#            STRANGER THINGS UI
# ======================================

st.markdown("""
<div style='text-align:center; padding-top:20px;'>
    <h1>STRANGER THINGS ANALYZER</h1>
    <p style="
        color:#bbb; 
        font-family:Roboto Mono;
        letter-spacing:2px;
        text-shadow:0 0 10px red;
    ">
        Emotion Detection From The Upside-Down
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #700000;'>", unsafe_allow_html=True)


# Input block
st.markdown("""
<h3>ENTER YOUR MESSAGE</h3>
""", unsafe_allow_html=True)

col_l, col_mid, col_r = st.columns([1, 2, 1])

with col_mid:
    input_text_raw = st.text_area(
        "",
        "Something feels wrong… deeply wrong.\nMy heart is racing for no reason.",
        height=180
    )

    run = st.button("🔻 ANALYZE EMOTION 🔻", use_container_width=True)

texts = [t.strip() for t in input_text_raw.split("\n") if t.strip()]


# Display results
if run and texts:

    results = detect_emotions(classifier, texts)

    st.markdown("""
    <h3 style='margin-top:40px;'>UPSIDE-DOWN RESULTS</h3>
    """, unsafe_allow_html=True)

    cols = st.columns(2)

    for i, result in enumerate(results):
        emotion = result["Dominant Emotion"]
        confidence = result["Confidence"]
        user_input = result["Input Text"]
        gif = EMOTION_GIFS.get(emotion, EMOTION_GIFS["NEUTRAL"])

        with cols[i % 2]:
            st.markdown(f"""
            <div class="result-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="result-emotion">{emotion}</span>
                    <img src="{gif}" class="emotion-gif">
                </div>

                <div style="color:#ccc; margin-top:12px; font-style:italic;">
                    "{user_input}"
                </div>

                <div style="margin-top:10px; color:#aaa; text-align:right;">
                    CONFIDENCE: {confidence}
                </div>
            </div>
            """, unsafe_allow_html=True)

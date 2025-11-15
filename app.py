import streamlit as st
from transformers import pipeline


# ------------------------------
# CONFIG
# ------------------------------
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
PAGE_ICON_URL = "https://cdn-icons-png.flaticon.com/128/10479/10479785.png"

EMOTION_GIFS = {
    "ANGER": "https://media0.giphy.com/media/jsNiI5nMGQurggwpkN/giphy.webp",
    "HAPPINESS": "https://media4.giphy.com/media/USR9bpLz899PYVHk7C/giphy.webp",
    "SADNESS": "https://media0.giphy.com/media/StAnQV9TUCuys/giphy.webp",
    "JOY": "https://media0.giphy.com/media/LN5bH1r7UEpSRbcN7M/giphy.webp",
    "FEAR": "https://media0.giphy.com/media/Gl7mfimOjkkGl5mMDS/giphy.webp",
    "NEUTRAL": "https://media3.giphy.com/media/7CXIO53h5YciXOp505/giphy.webp",
    "DISGUST": "https://media0.giphy.com/media/jsNiI5nMGQurggwpkN/giphy.webp",
    "SURPRISE": "https://media0.giphy.com/media/Gl7mfimOjkkGl5mMDS/giphy.webp",
}


# ------------------------------
# PAGE SETTINGS
# ------------------------------
st.set_page_config(
    page_title="Stranger Things Analyzer",
    page_icon=PAGE_ICON_URL,
    layout="wide"
)


# ------------------------------
# FULL STRANGER THINGS UI THEME 
# ------------------------------

st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background: #000;
}

/* Fullscreen Upside-Down Atmosphere */
body {
    background-image: url('https://i.imgur.com/zY8z1DP.jpeg');
    background-size: cover;
    background-attachment: fixed;
}

/* Fog Layer */
body::before {
    content: "";
    position: fixed;
    inset: 0;
    background: url("https://i.imgur.com/8fK4hFf.png");
    opacity: 0.07;
    pointer-events: none;
    animation: fogMove 80s infinite linear;
}

@keyframes fogMove {
    from { transform: translateX(0); }
    to { transform: translateX(20%); }
}

/* Floating Spores */
body::after {
    content: "";
    position: fixed;
    inset: 0;
    background: url("https://i.imgur.com/7NkZyHf.png");
    background-size: cover;
    opacity: 0.18;
    pointer-events: none;
    animation: sporesFloat 12s infinite ease-in-out alternate;
}

@keyframes sporesFloat {
    from { opacity: 0.1; transform: translateY(0); }
    to { opacity: 0.22; transform: translateY(-30px); }
}

/* VHS Scanline Effect */
.scanlines {
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        to bottom,
        rgba(255,0,0,0.05) 0px,
        rgba(0,0,0,0) 2px,
        rgba(0,0,0,0) 4px
    );
    mix-blend-mode: screen;
    pointer-events:none;
    opacity:0.25;
}

/* Title */
h1 {
    font-family: 'Cinzel', serif;
    font-size: 4.6rem;
    text-align:center;
    margin-top:30px;
    color:#e50914;
    text-shadow: 
        0 0 20px #ff1d2d,
        0 0 40px #ff1d2d,
        0 0 80px #700000;
    letter-spacing: 6px;
}

/* Subtitle */
.subtext {
    text-align:center;
    color:#ccc;
    font-family: 'Roboto Mono', monospace;
    font-size:1.1rem;
    letter-spacing:3px;
    margin-top:-15px;
    margin-bottom:40px;
}

/* Input Panel */
.stTextArea textarea {
    background:#0a0a0a !important;
    color:white !important;
    border: 2px solid #e50914 !important;
    box-shadow:0 0 25px red;
    border-radius:10px;
    font-family:'Roboto Mono', monospace !important;
    font-size:1rem !important;
}

/* Button */
div.stButton > button {
    background:#e50914;
    color:white;
    font-size:1.3rem;
    padding:12px 20px;
    border:none;
    border-radius:6px;
    font-family:'Cinzel', serif;
    letter-spacing:3px;
    width:100%;
    box-shadow:0 0 25px #ff1d2d;
    transition:0.2s;
}

div.stButton > button:hover {
    transform:scale(1.05);
    box-shadow:0 0 40px #ff1d2d;
}

/* Result Card */
.result-card {
    background:rgba(0,0,0,0.7);
    padding:20px;
    margin-bottom:25px;
    border-radius:10px;
    border-left:5px solid #e50914;
    box-shadow:
        0 0 20px rgba(255,0,0,0.5),
        inset 0 0 20px rgba(255,0,0,0.2);
    backdrop-filter: blur(4px);
}

/* Emotion Label */
.result-emotion {
    color:white;
    font-family:'Cinzel', serif;
    font-size:1.1rem;
    padding:5px 12px;
    background:#e50914;
    border-radius:4px;
    text-shadow:0 0 10px red;
}

/* GIF bubble */
.emotion-gif {
    width:55px;
    height:55px;
    border-radius:50%;
    border:3px solid #e50914;
    box-shadow:0 0 12px red;
}

</style>

<div class="scanlines"></div>
""", unsafe_allow_html=True)


# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        return_all_scores=True
    )

classifier = load_model()


# ------------------------------
# TITLE
# ------------------------------
st.markdown("<h1>STRANGER THINGS ANALYZER</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Emotion Detection From The Upside-Down</p>", unsafe_allow_html=True)


# ------------------------------
# INPUT
# ------------------------------
input_text = st.text_area(
    "",
    "Something feels wrong... I can sense it breathing in the dark.",
    height=180
)

run = st.button("ENTER THE UPSIDE DOWN")


# ------------------------------
# EMOTION FUNCTION
# ------------------------------
def detect_emotions(texts):
    predictions = classifier(texts)
    results = []
    for text, p_list in zip(texts, predictions):
        top = max(p_list, key=lambda x: x["score"])
        results.append({
            "text": text,
            "emotion": top["label"].upper(),
            "score": round(top["score"], 4)
        })
    return results


# ------------------------------
# OUTPUT
# ------------------------------
if run:
    lines = [x.strip() for x in input_text.split("\n") if x.strip()]
    results = detect_emotions(lines)

    st.markdown("<h3 style='color:#e50914; margin-top:40px;'>UPSIDE-DOWN RESULTS</h3>", unsafe_allow_html=True)

    cols = st.columns(2)

    for i, res in enumerate(results):
        emotion = res["emotion"]
        gif = EMOTION_GIFS.get(emotion, EMOTION_GIFS["NEUTRAL"])

        with cols[i % 2]:
            st.markdown(f"""
            <div class="result-card">

                <div style="display:flex; justify-content:space-between;">
                    <span class="result-emotion">{emotion}</span>
                    <img class="emotion-gif" src="{gif}">
                </div>

                <div style="color:#ccc; margin-top:14px; font-style:italic;">
                    "{res['text']}"
                </div>

                <div style="color:#aaa; text-align:right; margin-top:10px;">
                    CONFIDENCE: {res['score']}
                </div>
            </div>
            """, unsafe_allow_html=True)

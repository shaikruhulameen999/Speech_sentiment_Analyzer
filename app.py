import streamlit as st
import speech_recognition as sr
import tempfile
import os

from transformers import pipeline


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Speech Sentiment Analyzer",
    page_icon="🎤",
    layout="centered"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.35), transparent 30%),
        radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.30), transparent 30%),
        linear-gradient(135deg, #0f172a, #1e1b4b, #111827);
    color: white;
}

/* Main container */
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 22px;
    padding: 28px;
    margin: 20px 0;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.30);
}

/* Section heading */
.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 15px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 12px 20px;
    font-size: 17px;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    box-shadow: 0 6px 20px rgba(99,102,241,0.35);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(139,92,246,0.55);
}

/* Radio buttons */
.stRadio > div {
    background: rgba(255,255,255,0.06);
    padding: 15px;
    border-radius: 15px;
}

/* Slider */
.stSlider {
    padding: 10px 0;
}

/* Result boxes */
.result-box {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    margin-top: 15px;
    border: 1px solid rgba(255,255,255,0.12);
}

/* Sentiment */
.positive {
    color: #4ade80;
    font-size: 30px;
    font-weight: 800;
}

.negative {
    color: #fb7185;
    font-size: 30px;
    font-weight: 800;
}

.neutral {
    color: #facc15;
    font-size: 30px;
    font-weight: 800;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# TITLE
# =====================================================

st.markdown(
    '<div class="main-title">🎤 Speech Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Convert your voice into text and analyze its sentiment using DistilBERT 🤖'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return sentiment_pipeline

sentiment_pipeline = load_model()


# =====================================================
# OPTION CARD
# =====================================================

st.markdown(
    '<div class="glass-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🎯 Choose Input Method</div>',
    unsafe_allow_html=True
)

option = st.radio(
    "Select an option:",
    ["🎤 Talk", "📁 Upload Audio"],
    horizontal=True
)

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# TALK
# =====================================================

if option == "🎤 Talk":

    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🎤 Record Your Voice</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Speak naturally and the system will convert your voice "
        "into text and predict its sentiment."
    )

    duration = st.slider(
        "⏱️ Recording duration (seconds)",
        3,
        10,
        5
    )

    if st.button("🎙️ Start Recording"):

      
        import scipy.io.wavfile as wav

        sample_rate = 16000

        st.info("🎤 Speak now...")

        wav.write(
            "voice_input.wav",
            sample_rate,
            audio
        )

        st.success("✅ Recording completed!")

        recognizer = sr.Recognizer()

        with sr.AudioFile("voice_input.wav") as source:

            recorded_audio = recognizer.record(source)

        try:

            text = recognizer.recognize_google(
                recorded_audio
            )

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.markdown("### 📝 Transcribed Text")

            st.write(f"**{text}**")

            st.markdown("</div>", unsafe_allow_html=True)

            # Sentiment prediction

            result = sentiment_pipeline(text)[0]

            label = result["label"]
            confidence = result["score"] * 100

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.markdown("### 🎯 Sentiment")

            if label.upper() == "POSITIVE":

                st.markdown(
                    '<div class="positive">😊 POSITIVE</div>',
                    unsafe_allow_html=True
                )

            elif label.upper() == "NEGATIVE":

                st.markdown(
                    '<div class="negative">😞 NEGATIVE</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="neutral">😐 NEUTRAL</div>',
                    unsafe_allow_html=True
                )

            st.markdown("### 📊 Confidence")

            st.progress(
                int(confidence)
            )

            st.write(
                f"**{confidence:.2f}%**"
            )

            st.markdown("</div>", unsafe_allow_html=True)

        except sr.UnknownValueError:

            st.error(
                "❌ Could not understand the audio."
            )

        except sr.RequestError:

            st.error(
                "❌ Speech recognition service unavailable."
            )

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# UPLOAD AUDIO
# =====================================================

else:

    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">📁 Upload Your Audio</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a WAV audio file and analyze the sentiment."
    )

    uploaded_file = st.file_uploader(
        "🎵 Choose an audio file",
        type=["wav"]
    )

    if uploaded_file is not None:

        st.audio(
            uploaded_file,
            format="audio/wav"
        )

        if st.button("🔍 Analyze Audio"):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as temp:

                temp.write(
                    uploaded_file.getbuffer()
                )

                audio_path = temp.name

            recognizer = sr.Recognizer()

            with sr.AudioFile(audio_path) as source:

                recorded_audio = recognizer.record(
                    source
                )

            try:

                text = recognizer.recognize_google(
                    recorded_audio
                )

                st.markdown(
                    '<div class="result-box">',
                    unsafe_allow_html=True
                )

                st.markdown("### 📝 Transcribed Text")

                st.write(
                    f"**{text}**"
                )

                st.markdown("</div>", unsafe_allow_html=True)

                # Sentiment

                result = sentiment_pipeline(text)[0]

                label = result["label"]

                confidence = result["score"] * 100

                st.markdown(
                    '<div class="result-box">',
                    unsafe_allow_html=True
                )

                st.markdown("### 🎯 Sentiment")

                if label.upper() == "POSITIVE":

                    st.markdown(
                        '<div class="positive">😊 POSITIVE</div>',
                        unsafe_allow_html=True
                    )

                elif label.upper() == "NEGATIVE":

                    st.markdown(
                        '<div class="negative">😞 NEGATIVE</div>',
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        '<div class="neutral">😐 NEUTRAL</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("### 📊 Confidence")

                st.progress(
                    int(confidence)
                )

                st.write(
                    f"**{confidence:.2f}%**"
                )

                st.markdown("</div>", unsafe_allow_html=True)

            except sr.UnknownValueError:

                st.error(
                    "❌ Could not understand the audio."
                )

            except sr.RequestError:

                st.error(
                    "❌ Speech recognition service unavailable."
                )

            finally:

                if os.path.exists(audio_path):

                    os.remove(audio_path)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    '<div class="footer">'
    '🤖 Powered by DistilBERT • Speech Recognition • Streamlit'
    '</div>',
    unsafe_allow_html=True
)

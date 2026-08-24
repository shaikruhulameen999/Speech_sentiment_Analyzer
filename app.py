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

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.35), transparent 30%),
        radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.30), transparent 30%),
        linear-gradient(135deg, #0f172a, #1e1b4b, #111827);
    color: white;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 22px;
    padding: 28px;
    margin: 20px 0;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.30);
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 15px;
}

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

.stRadio > div {
    background: rgba(255,255,255,0.06);
    padding: 15px;
    border-radius: 15px;
}

.result-box {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    margin-top: 15px;
    border: 1px solid rgba(255,255,255,0.12);
}

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
# LOAD DISTILBERT MODEL
# =====================================================

@st.cache_resource
def load_model():

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return sentiment_pipeline


# Load model
with st.spinner("🤖 Loading DistilBERT model..."):
    sentiment_pipeline = load_model()


# =====================================================
# SENTIMENT DISPLAY FUNCTION
# =====================================================

def show_sentiment(text):

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

    st.progress(min(int(confidence), 100))

    st.write(
        f"**{confidence:.2f}%**"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# SPEECH-TO-TEXT FUNCTION
# =====================================================

def convert_speech_to_text(audio_path):

    recognizer = sr.Recognizer()

    try:

        with sr.AudioFile(audio_path) as source:

            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        return text, None

    except sr.UnknownValueError:

        return None, "❌ Could not understand the audio."

    except sr.RequestError:

        return None, "❌ Google Speech Recognition service is unavailable."

    except Exception as e:

        return None, f"❌ Error processing audio: {str(e)}"


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
# TALK / MICROPHONE
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
        "Click the microphone button below and speak naturally. "
        "Your voice will be converted into text and analyzed."
    )

    # Browser microphone
    audio_value = st.audio_input(
        "🎙️ Click here to record your voice"
    )

    if audio_value is not None:

        st.success("✅ Recording completed!")

        # Get audio bytes
        audio_bytes = audio_value.getvalue()

        # Play recorded audio
        st.audio(
            audio_bytes,
            format="audio/wav"
        )

        # Save temporary WAV file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(audio_bytes)

            audio_path = temp_file.name

        # Convert speech to text
        with st.spinner("🎧 Converting speech to text..."):

            text, error = convert_speech_to_text(
                audio_path
            )

        # Remove temporary file
        if os.path.exists(audio_path):

            os.remove(audio_path)

        if error:

            st.error(error)

        else:

            # Show transcription
            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.markdown("### 📝 Transcribed Text")

            st.write(
                f"**{text}**"
            )

            st.markdown("</div>", unsafe_allow_html=True)

            # Sentiment prediction
            show_sentiment(text)

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
        "Upload a WAV audio file and analyze its sentiment."
    )

    uploaded_file = st.file_uploader(
        "🎵 Choose an audio file",
        type=["wav"]
    )

    if uploaded_file is not None:

        # Display audio
        st.audio(
            uploaded_file,
            format="audio/wav"
        )

        if st.button("🔍 Analyze Audio"):

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                audio_path = temp_file.name

            # Convert speech to text
            with st.spinner("🎧 Converting speech to text..."):

                text, error = convert_speech_to_text(
                    audio_path
                )

            # Remove temporary file
            if os.path.exists(audio_path):

                os.remove(audio_path)

            if error:

                st.error(error)

            else:

                # Show transcription
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
                show_sentiment(text)

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

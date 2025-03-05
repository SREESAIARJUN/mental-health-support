import os
import streamlit as st
import torch
import numpy as np
import google.generativeai as genai
from faster_whisper import WhisperModel
from nltk.sentiment import SentimentIntensityAnalyzer
from TTS.api import TTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load Whisper for real-time STT
whisper_model = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

# Load TTS model
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

# Emotion-based voice adjustments
emotion_voices = {
    "positive": {"speed": 1.2, "pitch": 1.1},
    "negative": {"speed": 0.9, "pitch": 0.8},
    "neutral": {"speed": 1.0, "pitch": 1.0},
}

# Function to detect emotion
def detect_emotion(text):
    sentiment = sia.polarity_scores(text)
    if sentiment["compound"] >= 0.05:
        return "positive"
    elif sentiment["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Function to get chatbot response
def get_chat_response(user_input):
    response = genai.chat(model="gemini-pro", messages=[{"role": "user", "content": user_input}])
    return response.text if response else "I'm not sure how to respond."

# Function to generate speech output
def generate_speech(text, emotion):
    config = emotion_voices.get(emotion, emotion_voices["neutral"])
    tts.tts_to_file(text=text, file_path="response.wav", speed=config["speed"], pitch=config["pitch"])
    return "response.wav"

# WebRTC audio processor class
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv_audio(self, frame: av.AudioFrame):
        audio_np = np.array(frame.to_ndarray()).flatten()
        self.audio_buffer.append(audio_np)
        return None  # No need to process, just collect data

# Streamlit UI
st.title("🗣️ Real-Time AI Chatbot with Emotional Voice Response")

# Chat via Text
user_input = st.text_input("Type your message here:")
if st.button("Send"):
    if user_input:
        emotion = detect_emotion(user_input)
        response_text = get_chat_response(user_input)
        response_audio = generate_speech(response_text, emotion)

        st.markdown(f"**🤖 Chatbot ({emotion} emotion):** {response_text}")
        st.audio(response_audio, format="audio/wav")

# Chat via Voice
st.write("🎤 **Speak in real-time using WebRTC:**")
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"video": False, "audio": True},
)

if webrtc_ctx.audio_processor:
    audio_data = np.concatenate(webrtc_ctx.audio_processor.audio_buffer, axis=0)

    if len(audio_data) > 0:
        segments, _ = whisper_model.transcribe(audio_data)
        transcribed_text = " ".join(segment.text for segment in segments)

        if transcribed_text:
            emotion = detect_emotion(transcribed_text)
            response_text = get_chat_response(transcribed_text)
            response_audio = generate_speech(response_text, emotion)

            st.markdown(f"**🤖 Chatbot ({emotion} emotion):** {response_text}")
            st.audio(response_audio, format="audio/wav")

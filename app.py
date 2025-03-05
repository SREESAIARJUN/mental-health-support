import os
import streamlit as st
import sounddevice as sd
import numpy as np
import torch
import nltk
import google.generativeai as genai
from faster_whisper import WhisperModel
from scipy.io.wavfile import write
from nltk.sentiment import SentimentIntensityAnalyzer
from TTS.api import TTS

# Download necessary NLTK data
nltk.download("vader_lexicon")

# Set up Google Gemini API Key (store in Streamlit Cloud Secrets for deployment)
GEMINI_API_KEY = os.getenv("AIzaSyBCZ6l4qRGHq9UqPej0MsEC2q4aBOCdXYY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load TTS model for voice output (Coqui-TTS)
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

# Load real-time whisper model for speech-to-text
whisper_model = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

# Emotion-based TTS voice mapping
emotion_voices = {
    "positive": {"speed": 1.2, "pitch": 1.1},
    "negative": {"speed": 0.9, "pitch": 0.8},
    "neutral": {"speed": 1.0, "pitch": 1.0},
}

# Function to detect emotion from text
def detect_emotion(text):
    sentiment = sia.polarity_scores(text)
    if sentiment["compound"] >= 0.05:
        return "positive"
    elif sentiment["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Function to generate chatbot response using Gemini
def get_chat_response(user_input):
    response = genai.chat(model="gemini-pro", messages=[{"role": "user", "content": user_input}])
    return response.text if response else "I'm not sure how to respond."

# Function to generate speech output
def generate_speech(text, emotion):
    config = emotion_voices.get(emotion, emotion_voices["neutral"])
    tts.tts_to_file(text=text, file_path="response.wav", speed=config["speed"], pitch=config["pitch"])
    return "response.wav"

# Function to record real-time audio and transcribe it
def record_audio(duration=5, samplerate=16000):
    st.write("🎙️ Recording... Speak now!")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    write("input_audio.wav", samplerate, audio)  # Save recorded audio
    return "input_audio.wav"

# Function to transcribe audio using Whisper
def transcribe_audio(filename):
    segments, _ = whisper_model.transcribe(filename)
    text = " ".join(segment.text for segment in segments)
    return text

# Streamlit UI
st.title("🗣️ Emotion-Aware AI Chatbot")
st.write("Chat with an AI that understands emotions and responds with voice!")

# Chat interface
user_input = st.text_input("Type your message here:")

if st.button("Send"):
    if user_input:
        emotion = detect_emotion(user_input)
        response_text = get_chat_response(user_input)
        response_audio = generate_speech(response_text, emotion)

        # Display chatbot response
        st.markdown(f"**🤖 Chatbot ({emotion} emotion):** {response_text}")
        st.audio(response_audio, format="audio/wav")

# Voice input button
if st.button("🎤 Speak"):
    audio_file = record_audio()
    transcribed_text = transcribe_audio(audio_file)
    
    st.write(f"**You said:** {transcribed_text}")

    emotion = detect_emotion(transcribed_text)
    response_text = get_chat_response(transcribed_text)
    response_audio = generate_speech(response_text, emotion)

    # Display chatbot response
    st.markdown(f"**🤖 Chatbot ({emotion} emotion):** {response_text}")
    st.audio(response_audio, format="audio/wav")

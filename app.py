import os
import streamlit as st
import torch
import numpy as np
import sounddevice as sd
import google.generativeai as genai
from queue import Queue
from faster_whisper import WhisperModel
from nltk.sentiment import SentimentIntensityAnalyzer
from TTS.api import TTS

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load Whisper for real-time STT (Optimized for fast transcription)
whisper_model = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

# Load TTS model for voice output
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

# Emotion-based voice adjustments
emotion_voices = {
    "positive": {"speed": 1.2, "pitch": 1.1},
    "negative": {"speed": 0.9, "pitch": 0.8},
    "neutral": {"speed": 1.0, "pitch": 1.0},
}

# Real-time Audio Queue
audio_queue = Queue()

# Function to detect emotion from text
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

# Real-time audio callback function
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Function to start real-time speech recognition
def real_time_transcription():
    samplerate = 16000  # Whisper expects 16kHz audio
    with sd.InputStream(samplerate=samplerate, channels=1, dtype=np.float32, callback=callback):
        st.write("🎙️ **Listening... Speak now!**")
        audio_data = []
        while True:
            chunk = audio_queue.get()
            audio_data.append(chunk)

            # Convert to numpy array for Whisper processing
            audio_array = np.concatenate(audio_data, axis=0)

            # Transcribe with Whisper
            segments, _ = whisper_model.transcribe(audio_array)
            transcribed_text = " ".join(segment.text for segment in segments)

            # Display partial transcription
            st.write(f"**You:** {transcribed_text}")

            # Break if user stops speaking (adjust as needed)
            if len(transcribed_text) > 10:  # Stop on long enough speech
                break

        return transcribed_text

# Streamlit UI
st.title("🗣️ Real-Time Emotion-Aware AI Chatbot")
st.write("Chat with AI that understands emotions & responds with voice!")

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
if st.button("🎤 Speak"):
    transcribed_text = real_time_transcription()

    if transcribed_text:
        emotion = detect_emotion(transcribed_text)
        response_text = get_chat_response(transcribed_text)
        response_audio = generate_speech(response_text, emotion)

        st.markdown(f"**🤖 Chatbot ({emotion} emotion):** {response_text}")
        st.audio(response_audio, format="audio/wav")

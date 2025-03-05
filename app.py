import streamlit as st
import google.generativeai as genai
import whisper
import tempfile
import os
import gtts
from textblob import TextBlob
import nest_asyncio
import pygame

# Fix asyncio issues in Streamlit
nest_asyncio.apply()

# Configure Google Gemini API
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-2.0-pro")

# Streamlit UI Config
st.set_page_config(page_title="Mental Health Support", page_icon="🧠", layout="wide")
st.title("🧠 Mental Health Support Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0.3:
        return "positive"
    elif sentiment_score < -0.3:
        return "negative"
    else:
        return "neutral"

# Function to generate response based on sentiment
def get_response(user_input):
    sentiment = analyze_sentiment(user_input)
    prompt = f"You are a supportive AI therapist. The user is feeling {sentiment}. Respond with empathy and encouragement.\nUser: {user_input}\nAI:"
    response = model.generate_content(prompt).text
    return response

# Function to generate speech from text
def text_to_speech(text):
    tts = gtts.gTTS(text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        return temp_audio.name

# Function to transcribe voice input using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# Audio input feature
uploaded_audio = st.file_uploader("🎙️ Upload an audio file (MP3, WAV) for voice input", type=["mp3", "wav"])
if uploaded_audio is not None:
    with st.spinner("Transcribing..."):
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
            temp_audio.write(uploaded_audio.getvalue())
            audio_text = transcribe_audio(temp_audio.name)
        st.write("🗣️ You said:", audio_text)
        user_query = audio_text
else:
    user_query = st.chat_input("Type your message...")

# Chatbot response handling
if user_query:
    with st.chat_message("user"):
        st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        response_text = get_response(user_query)
        st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Generate and play speech output
        audio_file = text_to_speech(response_text)
        st.audio(audio_file, format="audio/mp3")

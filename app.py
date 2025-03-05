import streamlit as st
import google.generativeai as genai
import whisper
import pyttsx3
from gtts import gTTS
import torch
from transformers import pipeline
import time
import os
import random

# Configure Gemini API
GEMINI_API_KEY = "your_gemini_api_key_here"
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"

# Load AI Sentiment Model (Deep Learning)
sentiment_model = pipeline("sentiment-analysis")

# Initialize TTS engine
engine = pyttsx3.init()

# Create a directory for saving journals
if not os.path.exists("journals"):
    os.makedirs("journals")

# Function: Advanced Sentiment Analysis
def get_sentiment(text):
    result = sentiment_model(text)[0]
    label = result["label"]
    score = result["score"]
    
    if label == "POSITIVE" and score > 0.7:
        return "positive 😊"
    elif label == "NEGATIVE" and score > 0.7:
        return "negative 😔"
    else:
        return "neutral 😐"

# Function: Convert text to speech
def speak(text):
    tts = gTTS(text, lang='en')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3" if os.name != "nt" else "start response.mp3")

# Function: Convert speech to text
def voice_input():
    model = whisper.load_model("base")
    with st.spinner("🎙️ Listening..."):
        audio_path = "voice_input.wav"
        os.system(f"arecord -d 5 -f cd {audio_path}" if os.name != "nt" else "Recording not supported")
        result = model.transcribe(audio_path)
        return result["text"]

# Function: Generate Motivational Image
def generate_motivation_image():
    motivation_list = [
        "Stay strong, you got this!",
        "Every day is a fresh start!",
        "You are worthy of happiness!"
    ]
    return random.choice(motivation_list)

# Function: Save Journal Entry
def save_journal(entry):
    filename = f"journals/journal_{time.time()}.txt"
    with open(filename, "w") as file:
        file.write(entry)
    st.sidebar.success("📝 Journal saved successfully!")

# System Prompt for AI
system_prompt = """
You are a compassionate AI designed to provide mental health support.
- Be empathetic, understanding, and encouraging.
- Analyze emotions from user input and respond accordingly.
- Offer relaxation techniques, mindfulness exercises, and self-care tips.
- Encourage seeking professional help when necessary.
- Avoid diagnosing medical conditions or offering treatment.
"""

# Streamlit UI
st.set_page_config(page_title="Mental Health Support System", page_icon="💙", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧘 Mental Health Support System</h1>", unsafe_allow_html=True)
st.write("Hello! I'm here to support you. Feel free to share your thoughts.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "Bot", "message": "How are you feeling today?"}]

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["message"])

# User Input Options
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.chat_input("Type your thoughts...")
with col2:
    if st.button("🎙️ Speak"):
        user_input = voice_input()

# Process User Input
if user_input:
    sentiment = get_sentiment(user_input)  # Analyze emotion

    # Append user message
    st.session_state.messages.append({"role": "USER", "message": f"{user_input} ({sentiment})"})

    # Display user message
    with st.chat_message("USER"):
        st.write(user_input)

    # Generate AI Response
    full_prompt = f"{system_prompt}\nUser: {user_input} (Mood: {sentiment})\nAI:"
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(full_prompt)
    
    bot_reply = response.text.strip() if response else "I'm here to listen. Can you tell me more?"
    
    # Append AI response
    st.session_state.messages.append({"role": "Bot", "message": bot_reply})

    # Display AI response
    with st.chat_message("Bot"):
        st.write(bot_reply)

    # Speak AI response
    speak(bot_reply)

# Sidebar: Well-being Tips & Mood Tracker
st.sidebar.markdown("### 🌿 Mental Well-being Tips")
st.sidebar.info("✔ Deep breathing & mindfulness\n✔ Stay hydrated & rest well\n✔ Talk to someone you trust\n✔ Engage in hobbies\n✔ Seek professional help if needed")

st.sidebar.markdown("### 📈 Your Mood Tracker")
moods = [msg["message"].split("(")[-1].strip(")") for msg in st.session_state.messages if "message" in msg]
mood_counts = {mood: moods.count(mood) for mood in set(moods)}
st.sidebar.bar_chart(mood_counts)

st.sidebar.markdown("### 🎨 AI Motivation")
st.sidebar.image(f"https://dummyimage.com/300x200/000/fff&text={generate_motivation_image()}")

# Journal Section
st.sidebar.markdown("### ✍ Journal Your Thoughts")
journal_entry = st.sidebar.text_area("Write here...")
if st.sidebar.button("Save Entry"):
    save_journal(journal_entry)

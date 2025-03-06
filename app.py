import streamlit as st
import google.generativeai as genai
from audio_recorder_streamlit import audio_recorder
import numpy as np
import io
import re
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

# Configure page
st.set_page_config(page_title="Mental Health Assistant", layout="wide")

# Add CSS for fixed input positioning
st.markdown("""
    <style>
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Container styles */
        .stApp {
            margin: 0;
            padding: 0;
        }
        
        /* Chat container */
        .chat-container {
            margin-bottom: 120px;  /* Space for input container */
            padding: 20px;
            overflow-y: auto;
        }
        
        /* Input container */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            border-top: 1px solid #ddd;
            padding: 20px;
            z-index: 100;
        }
        
        /* Remove default Streamlit padding */
        .block-container {
            padding: 0 !important;
        }
        
        div[data-testid="stVerticalBlock"] {
            padding: 0 !important;
            gap: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Configure Gemini API
genai.configure(api_key="AIzaSyB0x0Fv6jiluu8JdFToe4QKXQRHK8SMmrA")

# Initialize the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite",
    generation_config=generation_config,
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ],
    system_instruction="""
    You are a compassionate and empathetic AI trained to provide mental health support.
    - Always respond in a calm and soothing manner.
    - Use positive reinforcement and validate emotions.
    - Avoid medical diagnoses but encourage seeking professional help.
    - Offer mindfulness tips, breathing exercises, and self-care suggestions.
    Note: Be concise in your responses don't give overwhelming responses unnecessarily. Yet, be detailed wherever requied.
    """
)

def clean_markdown(text):
    # Remove headers
    text = re.sub(r'#+\s*', '', text)
    # Remove bold and italic markers
    text = re.sub(r'\*+', '', text)
    # Remove bullet points
    text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)
    # Remove code blocks
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    # Remove links
    text = re.sub(r'$$([^$$]+)\]$$[^$$]+\)', r'\1', text)
    return text.strip()

# Initialize chat session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.title("Voice AI Mental Health Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.markdown('</div>', unsafe_allow_html=True)

# Fixed input container at bottom
st.markdown('<div class="input-container">', unsafe_allow_html=True)
col1, col2 = st.columns([8, 1])
with col1:
    user_text_input = st.text_input("Type your message:", "", key="user_input", label_visibility="collapsed")
with col2:
    audio = audio_recorder()
st.markdown('</div>', unsafe_allow_html=True)

# Process voice input
if audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio)
        tmp_filename = tmp_file.name

    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_filename) as source:
        audio_data = recognizer.record(source)
        try:
            user_input = recognizer.recognize_google(audio_data)
            st.session_state.messages.append({"role": "user", "content": user_input})
        except sr.UnknownValueError:
            st.write("Could not understand audio")
        except sr.RequestError:
            st.write("Speech recognition service unavailable")

    os.remove(tmp_filename)

# Process text input
if user_text_input:
    st.session_state.messages.append({"role": "user", "content": user_text_input})

if st.session_state.messages:
    # Send entire chat history for context
    conversation_history = [msg["content"] for msg in st.session_state.messages]
    response = model.generate_content(conversation_history)
    response_text = response.text
    response_text = clean_markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Convert response to speech
    tts = gTTS(text=response_text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
        tts.save(tts_file.name)
        st.audio(tts_file.name, format='audio/mp3')
        os.remove(tts_file.name)

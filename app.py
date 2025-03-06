import streamlit as st
import google.generativeai as genai
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import re

# Page configuration
st.set_page_config(page_title="🧘 Mental Health Assistant", layout="wide")

# Custom CSS to fix input at bottom
st.markdown("""
    <style>
        .stApp {
            margin: 0;
            padding: 0;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .fixed-input {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 20px;
            z-index: 1000;
        }
        .chat-container {
            margin-bottom: 100px;  /* Space for fixed input */
        }
        .audio-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1001;
        }
    </style>
""", unsafe_allow_html=True)

# Define style constants
USER_COLOR = "#E8E8E8"
ASSISTANT_COLOR = "#036165"
USER_TEXT_COLOR = "#000000"
ASSISTANT_TEXT_COLOR = "#FFFFFF"
BORDER_RADIUS = "8px"
FONT_FAMILY = "Arial, sans-serif"
USER_LOGO = "🧑"
ASSISTANT_LOGO = "🧘"

# Configure Gemini API
genai.configure(api_key="AIzaSyB0x0Fv6jiluu8JdFToe4QKXQRHK8SMmrA")

# Initialize the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
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
    Note: Be concise in your responses don't give overwhelming responses unnecessarily. Yet, be detailed wherever required.
    """
)

def clean_markdown(text):
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    text = re.sub(r'$$([^$$]+)\]$$[^$$]+\)', r'\1', text)
    return text.strip()

def display_message(message):
    """Display a message with custom styling"""
    if message["role"] == "user":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: {USER_COLOR}; color: {USER_TEXT_COLOR}; padding: 10px; 
                border-radius: {BORDER_RADIUS}; margin: 5px; max-width: 70%; font-family: {FONT_FAMILY};'>
                    <span>{message["content"]}</span> {USER_LOGO}
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start;'>
                <div style='background-color: {ASSISTANT_COLOR}; color: {ASSISTANT_TEXT_COLOR}; padding: 10px; 
                border-radius: {BORDER_RADIUS}; margin: 5px; max-width: 70%; font-family: {FONT_FAMILY};'>
                    {ASSISTANT_LOGO} <span>{message["content"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

# Initialize chat session
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to support you. How are you feeling today?"}]

# Sidebar
with st.sidebar:
    st.title("🧘 Mental Health Assistant")
    st.write("A safe space for emotional support and mental wellness.")
    
    # Clear chat button
    if st.button('Clear Chat History'):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to support you. How are you feeling today?"}]
        st.rerun()

# Main chat container
st.title("Voice AI Mental Health Assistant")

# Chat container with bottom margin
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        display_message(message)
    st.markdown('</div>', unsafe_allow_html=True)

# Fixed input container at bottom
with st.container():
    st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
    cols = st.columns([8, 1])
    
    with cols[0]:
        user_input = st.text_input("Type your message:", key="user_text_input")
    
    with cols[1]:
        st.markdown('<div class="audio-button">', unsafe_allow_html=True)
        audio = audio_recorder(key="audio_recorder")
        st.markdown('</div>', unsafe_allow_html=True)
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
            transcribed_text = recognizer.recognize_google(audio_data)
            st.session_state.messages.append({"role": "user", "content": transcribed_text})
            
            # Generate response
            conversation_history = [msg["content"] for msg in st.session_state.messages]
            response = model.generate_content(conversation_history)
            response_text = clean_markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Generate audio response
            tts = gTTS(text=response_text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
                tts.save(tts_file.name)
                st.audio(tts_file.name, format='audio/mp3')
                os.remove(tts_file.name)
            
            st.rerun()
            
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError:
            st.error("Speech recognition service unavailable")

    os.remove(tmp_filename)

# Process text input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        conversation_history = [msg["content"] for msg in st.session_state.messages]
        response = model.generate_content(conversation_history)
        response_text = clean_markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Generate audio response
        tts = gTTS(text=response_text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
            tts.save(tts_file.name)
            st.audio(tts_file.name, format='audio/mp3')
            os.remove(tts_file.name)
    
    st.rerun()

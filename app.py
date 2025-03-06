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

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            margin: 0;
            padding: 0;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Chat container */
        .chat-messages {
            margin-bottom: 70px;
            padding-bottom: 100px;
        }
        
        /* Input container styling */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 1rem;
            z-index: 1000;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
            align-items: center;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        }
        
        /* Chat input styling */
        .stChatInput {
            bottom: 0;
            background-color: white;
            padding: 20px 140px 20px 20px !important;
        }
        
        /* Audio button styling */
        .audio-button {
            position: fixed;
            bottom: 24px;
            right: 30px;
            z-index: 1001;
        }
        
        /* Audio recorder custom styling */
        .audio-recorder {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        
        .audio-recorder:hover {
            background: #f0f0f0 !important;
            border-radius: 50%;
        }
        
        /* Message styling */
        .stMarkdown {
            min-height: 0;
        }
        
        /* Hide default streamlit elements */
        .block-container {
            padding-bottom: 0px;
            padding-top: 0px;
            margin-top: 0px;
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

def generate_response(prompt):
    """Generate response using the Gemini model"""
    conversation_history = [msg["content"] for msg in st.session_state.messages]
    response = model.generate_content(conversation_history + [prompt])
    return clean_markdown(response.text)

def text_to_speech(text):
    """Convert text to speech and play audio"""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
        tts.save(tts_file.name)
        st.audio(tts_file.name, format='audio/mp3')
        os.remove(tts_file.name)

# Initialize chat session
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to support you. How are you feeling today?"}]

# Sidebar
with st.sidebar:
    st.title("🧘 Mental Health Assistant")
    st.write("A safe space for emotional support and mental wellness.")
    
    if st.button('Clear Chat History'):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to support you. How are you feeling today?"}]
        st.rerun()

# Main chat container
st.title("Voice AI Mental Health Assistant")

# Chat messages
st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
for message in st.session_state.messages:
    display_message(message)
st.markdown('</div>', unsafe_allow_html=True)

# Input container with integrated mic
st.markdown('<div class="input-container">', unsafe_allow_html=True)
col1, col2 = st.columns([20, 1])
with col1:
    prompt = st.chat_input("Type your message here...")
with col2:
    st.markdown('<div class="audio-button">', unsafe_allow_html=True)
    audio = audio_recorder(
        key="audio_recorder",
        pause_threshold=2.0,
        sample_rate=44100
    )
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Handle audio input
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
            
            with st.spinner("Thinking..."):
                response_text = generate_response(transcribed_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                text_to_speech(response_text)
            
            st.rerun()
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError:
            st.error("Speech recognition service unavailable")

    os.remove(tmp_filename)

# Handle text input
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        response_text = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        text_to_speech(response_text)
    
    st.rerun()

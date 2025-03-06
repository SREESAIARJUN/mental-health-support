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
            overflow: hidden;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Chat container styles */
        .chat-container {
            height: calc(100vh - 180px);
            overflow-y: auto;
            padding-bottom: 80px;
        }
        
        /* Fixed input container styles */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 20px;
            border-top: 1px solid #ddd;
            z-index: 1000;
        }
        
        /* Hide Streamlit's default elements */
        .stTextInput, .stAudio {
            margin-bottom: 0 !important;
        }
        
        div[data-testid="stVerticalBlock"] {
            padding-bottom: 0px;
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

def process_audio_to_text(audio_data):
    """Convert audio to text using speech recognition"""
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError:
        st.error("Speech recognition service unavailable")
        return None

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

# Chat messages container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    display_message(message)
st.markdown('</div>', unsafe_allow_html=True)

# Fixed input container
st.markdown('<div class="input-container">', unsafe_allow_html=True)
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("Type your message:", key="user_text_input", label_visibility="collapsed")
with col2:
    audio = audio_recorder(key="audio_recorder")
st.markdown('</div>', unsafe_allow_html=True)

# Handle audio input
if audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio)
        tmp_filename = tmp_file.name

    with sr.AudioFile(tmp_filename) as source:
        audio_data = sr.Recognizer().record(source)
        transcribed_text = process_audio_to_text(audio_data)
        
        if transcribed_text:
            st.session_state.messages.append({"role": "user", "content": transcribed_text})
            
            with st.spinner("Thinking..."):
                response_text = generate_response(transcribed_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                text_to_speech(response_text)
            
            st.rerun()

    os.remove(tmp_filename)

# Handle text input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        response_text = generate_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        text_to_speech(response_text)
    
    # Clear the input field
    st.session_state.user_text_input = ""
    st.rerun()

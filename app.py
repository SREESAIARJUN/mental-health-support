import streamlit as st
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
import os
import google.generativeai as genai
import tempfile
import speech_recognition as sr
import re

# Configure Gemini API
genai.configure(api_key="AIzaSyB0x0Fv6jiluu8JdFToe4QKXQRHK8SMmrA")

# Initialize the model
generation_config = {
    "temperature": 0.1,
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
    """
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Start chat session
chat_session = model.start_chat(
    history=[
        {"role": msg["role"], "parts": [msg["content"]]}
        for msg in st.session_state.messages
    ]
)

def recognize_speech(audio_bytes):
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError:
            return "Error connecting to Google Speech Recognition."
    return "No audio input detected."

def get_gemini_response(prompt):
    response = chat_session.send_message(prompt)
    return response.text if response else "Error generating response."

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        return temp_audio.name  # Return file path

def clean_markdown(text):
    text = re.sub(r'#+\s*', '', text)  # Remove headers
    text = re.sub(r'\*+', '', text)  # Remove bold and italic markers
    text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)  # Remove bullet points
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove links
    return text.strip()

# Streamlit UI
st.title("Voice AI Mental Support Chat")
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        padding: 10px;
        font-size: 18px;
    }
    .stAudio {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Mic button centered at the bottom
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
audio_bytes = audio_recorder()
st.markdown("</div>", unsafe_allow_html=True)

if audio_bytes:
    st.write("Processing audio...")
    user_input = recognize_speech(audio_bytes)
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        response = get_gemini_response(user_input)
        response = clean_markdown(response)
        
        # Store AI response in history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Convert response to speech
        audio_file = text_to_speech(response)
        st.audio(audio_file, format="audio/mp3")

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
import os
import google.generativeai as genai
import tempfile
import speech_recognition as sr

# Configure Gemini API
genai.configure(api_key="AIzaSyCj7X_uJVs2wxNlISNoYv8clt-Vq7u0aiM")

# Initialize the model
generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-pro-exp-02-05",
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

chat_session = model.start_chat(history=[])

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
    tts.save("response.mp3")
    os.system("mpg321 response.mp3" if os.name != "nt" else "start response.mp3")

# Streamlit UI
st.title("Voice AI Assistant using Gemini API")

audio_bytes = audio_recorder()
if audio_bytes:
    st.write("Processing audio...")
    user_input = recognize_speech(audio_bytes)
    st.write("You said:", user_input)
    
    if user_input:
        response = get_gemini_response(user_input)
        st.write("AI Response:", response)
        text_to_speech(response)

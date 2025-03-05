import streamlit as st
import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBCZ6l4qRGHq9UqPej0MsEC2q4aBOCdXYY"
genai.configure(api_key=GEMINI_API_KEY)

# Model Selection
MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"

# System Prompt for Mental Health Chatbot
system_prompt = """
You are a compassionate AI designed to provide mental health support.
- Always be empathetic, supportive, and non-judgmental.
- Listen actively and validate emotions.
- Offer mindfulness exercises, relaxation techniques, and well-being advice.
- Encourage seeking professional help when needed.
- Do NOT diagnose medical conditions or provide medical treatment.
"""

# Streamlit App Configuration
st.set_page_config(page_title="Mental Health Support System", page_icon="💙")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧘 Mental Health Support System</h1>", unsafe_allow_html=True)
st.write("Hello! I'm here to support you. Feel free to share your thoughts.")

# Chat History Handling
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "Bot", "message": "How are you feeling today?"}]

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["message"])

# User Input
user_input = st.chat_input("Share your thoughts...")

if user_input:
    # Append user message to session state
    st.session_state.messages.append({"role": "USER", "message": user_input})
    
    # Display user message
    with st.chat_message("USER"):
        st.write(user_input)
    
    # Generate AI Response
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAI:"
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(full_prompt)
    
    # Extract AI response
    bot_reply = response.text.strip() if response else "I'm here to listen. Can you tell me more?"
    
    # Append AI response to session state
    st.session_state.messages.append({"role": "Bot", "message": bot_reply})
    
    # Display AI response
    with st.chat_message("Bot"):
        st.write(bot_reply)

# Sidebar with Well-being Resources
st.sidebar.markdown("### 🌿 Mental Well-being Tips")
st.sidebar.info("✔ Practice deep breathing & mindfulness\n✔ Stay hydrated & get enough sleep\n✔ Talk to someone you trust\n✔ Engage in activities that make you happy\n✔ Seek professional help when needed")

# Motivational Quotes Section
st.sidebar.markdown("### 💡 Inspirational Quote")
st.sidebar.success("“You are braver than you believe, stronger than you seem, and smarter than you think.” – A.A. Milne")

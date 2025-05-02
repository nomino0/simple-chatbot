import streamlit as st
import os
from dotenv import load_dotenv

# Configure page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Chatbot UI", page_icon=":robot_face:", layout="centered")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Import Groq after ensuring the API key is available
from groq import Groq


Model_options = {
    "gemma2-9b-it": "Gemma 2 9B (Google)",
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (Meta)",
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant (Meta)",
    "llama-guard-3-8b": "Llama Guard 3 8B (Meta)",
    "llama3-70b-8192": "Llama 3 70B 8192 (Meta)",
    "llama3-8b-8192": "Llama 3 8B 8192 (Meta)",
    "whisper-large-v3": "Whisper Large V3 (OpenAI)",
    "whisper-large-v3-turbo": "Whisper Large V3 Turbo (OpenAI)",
    "distil-whisper-large-v3-en": "Distil Whisper Large V3 English (OpenAI)"
}


def llm_answer(history, model_name):
    # Use the model name passed as a parameter
    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=history,
        model=model_name,
    )
    return chat_completion.choices[0].message.content

st.title("Chatbot")

# Select the model 
selected_model = st.selectbox("Select a model", options=list(Model_options.keys()), format_func=lambda x: Model_options[x])
st.write(f"Selected model: {Model_options[selected_model]}")

if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.history.append(
        {
            'role': 'system',
            'content': 'You are Thabet, a tunisian assistant . You will answear evry question while sounding fun and always shouting.'
        }
    )

user_input = st.chat_input("Ask me anything:", key="user_input")

if user_input:
    user_prompt = {
        "role": "user",
        "content": user_input,
    }
    st.session_state.history.append(user_prompt)
    
    with st.spinner("Thinking..."):
        # Pass the selected model key to the llm_answer function
        answer = llm_answer(st.session_state.history, selected_model)
    
    st.session_state.history.append(
        {
          "role": "assistant",
          "content": answer
        }
    )

# Display all messages in chat history
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
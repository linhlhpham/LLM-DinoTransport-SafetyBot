import streamlit as st
from openai import OpenAI
import openai
import os


os.environ["OPENAI_API_KEY"] = 'example_key'

st.title("Hello, I am HODAML Team 1 Chatbot")
# Set OpenAI API key from Streamlit secrets
client = OpenAI()

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": 'You are a helpful assistant'}]

# Initialize temperature
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0

if "system_message_switch" not in st.session_state:
    st.session_state["system_message_switch"] = False
# Display chat messages from history on app rerun
def display_previous_messages():
    messages = st.session_state.messages
    for i in range(1,len(messages)):
        with st.chat_message(messages[i]["role"]):
            st.markdown(messages[i]["content"])

def side_bar():
    # Sidebar for temperature selection
    st.session_state["temperature"] = st.sidebar.select_slider(
       'Select the Model Temperature:',
       options=[0.0, 0.4, 0.9],
       value=0.4
    )

    st.session_state["openai_model"] = st.sidebar.selectbox(
        'Select Models',
        ('gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-3.5-turbo-16k')
    )

    st.session_state["system_message"] = st.sidebar.selectbox(
        'Select System Message',
        ('You are a helpful assistant', 'You are an unhelpful assistant.', 'You are a PhD student at Harvard University'),
        on_change=system_message_switch
    )  

    if st.session_state["system_message_switch"]:
        st.session_state["system_message_switch"] = False
        st.session_state.messages = [{"role": "system", "content": st.session_state["system_message"]}]

def system_message_switch():
    st.session_state["system_message_switch"] = True

def ask_openai():
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages],
        temperature=st.session_state["temperature"]
    )

    return response.choices[0].message.content

def main_chat():
    # Accept user input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("system"):
            response = ask_openai()
            st.markdown(response)
            st.session_state.messages.append({"role": "system", "content": response})

if __name__ == '__main__':
    side_bar()
    display_previous_messages()
    main_chat()

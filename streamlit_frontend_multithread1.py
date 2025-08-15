import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

import re
from collections import Counter

stopwords = set(["the","is","a","of","to","in","for","and","on","with","how","do","i"])

# **************************************** utility functions *************************

def generate_thread_id():
    return str(uuid.uuid4())

def generate_thread_name(user_input):
    # Take first 5 words of user input as name (fallback to "New Chat")
    #return " ".join(user_input.strip().split()[:5]) if user_input.strip() else "New Chat"
    words = re.findall(r'\w+', user_input.lower())
    filtered_words = [w for w in words if w not in stopwords]
    most_common = [w for w, _ in Counter(filtered_words).most_common(3)]
    return " ".join(most_common).title() or "New Chat"

def reset_chat(name="New Chat"):
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id, name)
    st.session_state['message_history'] = []

def add_thread(thread_id, name):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        st.session_state['thread_names'][thread_id] = name

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}

add_thread(st.session_state['thread_id'], "New Chat")


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    thread_name = st.session_state['thread_names'].get(thread_id, str(thread_id))
    if st.sidebar.button(thread_name, key=f"thread_btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # If this is the first message in this thread, update its name
    if len(st.session_state['message_history']) == 0:
        st.session_state['thread_names'][st.session_state['thread_id']] = generate_thread_name(user_input)

    # add the user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # stream assistant message
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

import streamlit as st
from flipkart.data_ingestion import data_ingestion
from flipkart.retrieval_generation import build_chain

st.set_page_config(
    page_title="Flipkart Chatbot",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Flipkart Product Chatbot")

# Load chain once
@st.cache_resource
def load_chain():
    vstore = data_ingestion("done")
    return build_chain(vstore)

chain = load_chain()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user"

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    response = chain.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    answer = response.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

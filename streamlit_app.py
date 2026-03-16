import streamlit as st
import uuid
from flipkart.data_ingestion import data_ingestion
from flipkart.retrieval_generation import build_chain, clear_session_history

st.set_page_config(page_title="Flipkart AI Assistant", page_icon="🛒")

st.title("🛒 Flipkart Product Chatbot")
st.caption("Ask me about electronics, clothing, or home appliances based on real reviews.")

@st.cache_resource
def load_rag_system():
    # Ensure data_ingestion returns a valid vector store
    vstore = data_ingestion("done")
    return build_chain(vstore)

try:
    chain = load_rag_system()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar controls
with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        clear_session_history(st.session_state.session_id)
        st.rerun()

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What are people saying about the latest iPhone?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching reviews..."):
            try:
                # Invoke the chain
                response = chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                # If you use StrOutputParser, response is a string. 
                # If not, it's an AIMessage object.
                answer = response.content if hasattr(response, 'content') else response
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"I encountered an error: {e}")
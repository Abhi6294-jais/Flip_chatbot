import streamlit as st
from flipkart.data_ingestion import data_ingestion
from flipkart.retrieval_generation import build_chain

# Page configuration
st.set_page_config(
    page_title="Flipkart Chatbot",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Flipkart Product Chatbot")

# Load chain once (cached)
@st.cache_resource
def load_chain():
    vstore = data_ingestion("done")
    return build_chain(vstore)

chain = load_chain()

# Initialize session state
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
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    try:
        response = chain.invoke(
            {"input": prompt},
            config={
                "configurable": {
                    "session_id": st.session_state.session_id
                }
            }
        )
        
        # Handle response from our guardrail chain
        # Our chain returns AIMessage objects with content attribute
        if hasattr(response, 'content'):
            answer = response.content
        elif isinstance(response, dict):
            answer = response.get("output", str(response))
        else:
            answer = str(response)

    except Exception as e:
        # Log error for debugging (can be removed in production)
        print(f"Debug - Error details: {str(e)}")
        # User-friendly error message
        answer = "I encountered an issue. Please try asking in a different way."

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

# Optional: Add a clear chat button in sidebar
with st.sidebar:
    st.markdown("### Chat Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        # Also clear the session history in the chain
        if "session_id" in st.session_state:
            # You might want to add a function to clear history from store
            pass
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot helps you find products on Flipkart. Ask about:")
    st.markdown("- Product recommendations")
    st.markdown("- Product features")
    st.markdown("- Comparisons")
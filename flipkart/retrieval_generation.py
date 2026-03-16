import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from flipkart.data_ingestion import data_ingestion

# Load environment variables
load_dotenv()

# Initialize LLM
# Using llama-3.3-70b-versatile as requested in your correct code snippet
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

# Session Store for Chat History
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def clear_session_history(session_id: str):
    """Utility to clear history if needed (e.g., from Streamlit)"""
    if session_id in store:
        store.pop(session_id)

def build_chain(vstore):
    """
    Constructs a conversational RAG chain using high-level LangChain factory methods.
    """
    # 1. Setup Retriever
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    # 2. Contextualize Question (History Aware Retriever)
    # This reformulates the user question based on chat history to make it standalone.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    # 3. Answer Question (Stuff Documents Chain)
    # This takes the retrieved context and the reformulated question to generate an answer.
    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", PRODUCT_BOT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    # 4. Final RAG Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 5. Add Conversational Memory Wrapper
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

if __name__ == "__main__":
    # Integration test for the module
    vstore = data_ingestion("done")
    chain = build_chain(vstore)
    
    config = {"configurable": {"session_id": "test_user"}}
    
    print("--- Query 1 ---")
    res1 = chain.invoke({"input": "can you tell me the best bluetooth buds?"}, config=config)
    print("Answer:", res1["answer"])
    
    print("\n--- Query 2 (Testing Memory) ---")
    res2 = chain.invoke({"input": "what was my previous question?"}, config=config)
    print("Answer:", res2["answer"])
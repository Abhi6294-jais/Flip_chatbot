import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# Initialize LLM
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
llm = ChatGroq(model=GROQ_MODEL, temperature=0.1) # Lower temperature for consistency

# Session Store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def clear_session_history(session_id: str):
    store.pop(session_id, None)

# --- ENHANCED PROMPT ---
PRODUCT_BOT_TEMPLATE = """You are a specialized Flipkart Product Expert.
You provide recommendations based on the customer reviews provided in the CONTEXT.

INSTRUCTIONS:
1. Mention specific product names found in the context.
2. If multiple products are found, compare them briefly based on user reviews (e.g., "Users liked the battery of X but preferred the sound of Y").
3. If the context is empty or unrelated to the question, say: "I couldn't find specific customer reviews for that product in our database. Would you like me to look for something else?"
4. Avoid generic "I am an AI" intros. Get straight to the product details.

CONTEXT:
{context}

QUESTION:
{input}

EXPERT RESPONSE:"""

# --- UTILITY FUNCTIONS ---

def format_docs(docs):
    """Formats retrieved docs and prints a debug log to the console."""
    if not docs:
        return "No relevant product reviews found."
    
    formatted = []
    for i, d in enumerate(docs):
        # We include metadata if available to help the LLM identify the specific product
        title = d.metadata.get('product_name', 'Unknown Product')
        formatted.append(f"--- Product: {title} ---\nReview Snippet: {d.page_content}")
    
    context_str = "\n\n".join(formatted)
    # Debug: See what is actually being sent to the LLM
    print(f"\n[DEBUG] Context Length: {len(docs)} snippets retrieved.")
    return context_str

def build_chain(vstore):
    # Use Similarity Search with Score to filter out 'random' junk
    # 'k: 5' provides a better variety for recommendations
    retriever = vstore.as_retriever(search_kwargs={"k": 5})

    # 1. Safety & Sanitization
    def sanitize_and_guard(inputs: dict) -> dict:
        text = inputs.get("input", "")
        banned = ["hack", "exploit", "jailbreak"]
        if any(w in text.lower() for w in banned):
            raise ValueError("Inappropriate content detected.")
        
        # Redact PII
        text = re.sub(r"[\w\.-]+@[\w\.-]+", "[EMAIL]", text)
        inputs["input"] = text
        return inputs

    # 2. Main RAG Logic
    # We've removed the contextualization step. 
    # The 'input' is now passed directly to the retriever and the final prompt.
    rag_chain = (
        RunnableLambda(sanitize_and_guard)
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | ChatPromptTemplate.from_messages([
            ("system", PRODUCT_BOT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )



# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    vstore = data_ingestion("done")
    chain = build_chain(vstore)

    res1 = chain.invoke(
        {"input": "Can you tell me the best bluetooth buds?"},
        config={"configurable": {"session_id": "abhishek"}}
    )

    print("\nAnswer 1:\n", res1.content)

    res2 = chain.invoke(
        {"input": "What was my previous question?"},
        config={"configurable": {"session_id": "abhishek"}}
    )

    print("\nAnswer 2:\n", res2.content)

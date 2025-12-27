from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory

from flipkart.data_ingestion import data_ingestion

# Load env variables
load_dotenv()

# Optional safety check
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("❌ GROQ_API_KEY missing from .env")

# Model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)



# -------------------------------------------------
# MODEL
# -------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

# -------------------------------------------------
# MEMORY STORE
# -------------------------------------------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# -------------------------------------------------
# PROMPT
# -------------------------------------------------
PRODUCT_BOT_TEMPLATE = """
You are an ecommerce chatbot expert.
Answer ONLY using the given context.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PRODUCT_BOT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

# -------------------------------------------------
# BUILD RAG CHAIN (NO langchain.chains)
# -------------------------------------------------
def build_chain(vstore):

    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | prompt
        | llm
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

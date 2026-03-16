from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from groq import BadRequestError as GroqBadRequestError

from flipkart.data_ingestion import data_ingestion

# Load env variables
load_dotenv()

# Optional safety check
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("❌ GROQ_API_KEY missing from .env")

# Optional model selection via environment variable (e.g., GROQ_MODEL=llama-3.1-8b-instant)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Groq-only model initialization
try:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.5)
except Exception as e:
    raise RuntimeError(
        f"Failed to initialize Groq LLM. Ensure GROQ_API_KEY is set and your Groq org/model permissions allow {GROQ_MODEL}."
        f" Details: {e}"
    )

# -------------------------------------------------
# MEMORY STORE
# -------------------------------------------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def clear_session_history(session_id: str):
    store.pop(session_id, None)

# -------------------------------------------------
# PROMPT
# -------------------------------------------------
PRODUCT_BOT_TEMPLATE = """You are a helpful and polite Flipkart ecommerce expert chatbot.
Your primary goal is to provide accurate product information and recommendations based STRICTLY on the real customer reviews provided in the context below.

CRITICAL RULES:
1. ONLY use the information provided in the CONTEXT.
2. If the answer is not contained within the CONTEXT, you must explicitly say "I'm sorry, but I don't have enough information about that based on the current reviews." Do NOT make up, guess, or infer facts, prices, or specifications that are not explicitly stated.
3. If discussing a product, try to mention the product name using the context's metadata if available.
4. Keep your answers concise, helpful, and directly address the user's question.
5. Do not hallucinate URLs, links, or contact numbers.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PRODUCT_BOT_TEMPLATE),
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

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    def get_contextualized_question(inputs):
        chat_history = inputs.get("chat_history", [])
        if chat_history:
            return contextualize_q_chain.invoke(inputs)
        return inputs["input"]

    def format_final_inputs(x):
        return {
            "context": x["context"],
            "input": x["contextualized_query"]
        }

    def check_banned_keywords(inputs: dict) -> dict:
        """
        Deterministic guardrail: Block requests containing banned keywords.
        Runs BEFORE anything else -- zero LLM cost for blocked requests.
        """
        banned_keywords = ["hack", "exploit", "malware", "jailbreak", "bypass"]
        content = inputs.get("input", "")
        
        if isinstance(content, str):
            content_lower = content.lower()
            for keyword in banned_keywords:
                if keyword in content_lower:
                    print(f"Blocked -- keyword detected: '{keyword}'")
                    raise ValueError(
                        "I cannot process requests containing inappropriate content. "
                        "Please rephrase your request."
                    )
        return inputs

    def check_safety_guardrail(response):
        """
        Model-based guardrail: Use an LLM to evaluate response safety.
        Runs AFTER the principal LLM produces a response, before it reaches the user.
        """
        # Using ChatGroq with a fast model for the safety check (equivalent to lightweight GPT-4o-mini)
        safety_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        
        content = response.content if hasattr(response, 'content') else str(response)

        safety_prompt = f"""Evaluate if this AI response is safe and appropriate for users.
Respond with only 'SAFE' or 'UNSAFE'.

Response to evaluate:
{content}"""

        result = safety_model.invoke(
            [{"role": "user", "content": safety_prompt}]
        )

        if "UNSAFE" in result.content.upper():
            print("Output flagged as UNSAFE -- replacing with safe fallback")
            safe_fallback = (
                "I'm unable to provide that response. "
                "Please rephrase your request or contact support."
            )
            if hasattr(response, 'content'):
                response.content = safe_fallback
            elif isinstance(response, str):
                response = safe_fallback
                
        return response

    def sanitize_input(inputs: dict) -> dict:
        text = inputs.get("input", "")

        # Simple deterministic sanitization for common sensitive patterns
        import re
        email_pattern = r"[\w\.-]+@[\w\.-]+"
        cc_pattern = r"(?:\d[ -]*?){13,16}"
        api_key_pattern = r"sk-[A-Za-z0-9]{32}"

        text = re.sub(email_pattern, "[REDACTED_EMAIL]", text)
        text = re.sub(cc_pattern, "[REDACTED_CREDIT_CARD]", text)
        text = re.sub(api_key_pattern, "[REDACTED_API_KEY]", text)

        inputs["input"] = text
        return inputs

    rag_chain = (
        RunnableLambda(check_banned_keywords)
        | RunnableLambda(sanitize_input)
        | RunnablePassthrough.assign(
            contextualized_query=get_contextualized_question
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["contextualized_query"]))
        )
        | format_final_inputs
        | prompt
        | llm
        | RunnableLambda(check_safety_guardrail)
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

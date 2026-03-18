# import os
# import re
# from dotenv import load_dotenv

# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain_community.chat_message_histories import ChatMessageHistory
# from flipkart.data_ingestion import data_ingestion
# # -------------------------------
# # CONFIG & MODELS
# # -------------------------------
# load_dotenv()

# def get_llm(model_name="llama-3.1-8b-instant", temp=0.3):
#     return ChatGroq(model=model_name, temperature=temp)

# llm = get_llm()
# evaluator = get_llm(temp=0)

# # -------------------------------
# # MEMORY STORE
# # -------------------------------
# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# # -------------------------------
# # GUARDRAILS & ROUTING
# # -------------------------------
# def preprocess_input(inputs: dict) -> dict:
#     text = inputs.get("input", "").strip()
#     low_text = text.lower()

#     # 1. Banned Keywords
#     if any(word in low_text for word in ["hack", "exploit", "malware", "jailbreak"]):
#         return {
#             "input": text,
#             "output": "❌ Security Alert: Inappropriate request detected.",
#             "context": "NO_CONTEXT"
#         }

#     # 2. Sanitize (PII Redaction)
#     text = re.sub(r"[\w\.-]+@[\w\.-]+", "[REDACTED_EMAIL]", text)
#     text = re.sub(r"(?:\d[ -]*?){13,16}", "[REDACTED_CARD]", text)

#     # 3. Intent Detection / Routing
#     greetings = ["hi", "hello", "hey", "yo", "hola"]
#     vague_patterns = ["what", "why", "how", "help", "?", "tell me", "suggest", "show me", "anything", "something"]
    
#     # Check for inventory queries FIRST (more specific)
#     inventory_queries = ["what products", "list items", "what do you have", "show products", "categories", "products do you have"]
#     if any(q in low_text for q in inventory_queries):
#         return {**inputs, "input": text, "intent": "inventory_query"}
    
#     # Then check for greetings (less specific)
#     if any(greet in low_text.split() for greet in greetings):  # Check whole words only
#         return {
#             "input": text,
#             "output": "Hi! 👋 I'm your Flipkart assistant. How can I help you find products today?",
#             "context": "NO_CONTEXT"
#         }

#     # Check for vague queries last
#     if len(low_text) < 3 or any(p in low_text for p in vague_patterns):
#         return {
#             "input": text,
#             "output": "Please ask a specific product-related question.\n\nTry asking:\n• 'Best earbuds under ₹2000'\n• 'Top gaming laptops'\n• 'Compare iPhone 15 and Samsung S24'",
#             "context": "NO_CONTEXT"
#         }

#     return {**inputs, "input": text}

# def check_grounding(chain_output: dict) -> dict:
#     """Evaluates if the LLM response is supported by the retrieved context."""
#     if "output" in chain_output and "context" in chain_output:
#         context = chain_output["context"]
#         answer = chain_output["output"]
        
#         if not context or context in ["NO_CONTEXT", ""]: 
#             return chain_output

#         eval_prompt = f"Context: {context}\nAnswer: {answer}\nRespond only: GROUNDED or NOT_GROUNDED"
#         result = evaluator.invoke(eval_prompt)

#         if "NOT_GROUNDED" in result.content.upper():
#             return {
#                 **chain_output, # Preserve original input and metadata
#                 "output": "I don't know based on the available data."
#             }
            
#     return chain_output

# # -------------------------------
# # PROMPT & RAG CHAIN
# # -------------------------------
# SYSTEM_PROMPT = """You are a strict ecommerce assistant.
# 1. Answer ONLY using the provided CONTEXT.
# 2. If the user asks for products or categories, summarize what is found in the CONTEXT.
# 3. If the answer is NOT present, say: "I don't know based on the available data."
# 4. Do NOT guess. Keep answers concise.

# CONTEXT:
# {context}"""

# prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}")
# ])

# def generate_response(x: dict) -> dict:
#     """Handles LLM generation with intent-based overrides."""
#     payload = x.copy()
    
#     # Check for different query types
#     input_lower = payload["input"].lower()
    
#     if x.get("intent") == "inventory_query":
#         payload["input"] = "List all available products and categories from the context."
#     elif "recommend" in input_lower or "good" in input_lower or "best" in input_lower:
#         # For recommendation queries, preserve the user's request
#         payload["input"] = f"Based on the context, {payload['input']}"
    
#     # Add instruction to use context
#     if "context" in x and x["context"] != "NO_CONTEXT":
#         payload["input"] = f"Using ONLY the product information in the context, {payload['input']}"
    
#     response = llm.invoke(prompt.format_messages(**payload))
#     return {
#         **x,
#         "output": response.content
#     }

# def build_chain(vstore):
#     retriever = vstore.as_retriever(search_kwargs={"k": 5})

#     # Improved query reformulation with better context awareness
#     rephrase_prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are a query reformulation assistant for a product search system.
#         Given the conversation history and the current user message, generate a standalone search query.
        
#         Guidelines:
#         - If the user is asking about something mentioned before (like "earbuds"), include that context
#         - If the user asks for recommendations ("recommend me", "good", "best"), include those keywords
#         - Output ONLY the search query, nothing else
#         """),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "Current message: {input}\n\nGenerate a search query:")
#     ])
    
#     rephrase_chain = rephrase_prompt | llm | (lambda x: x.content)

#     def format_docs(docs):
#         if not docs:
#             return "NO_CONTEXT"
#         return "\n\n".join([f"Product: {doc.page_content}" for doc in docs])

#     # Core logic
#     rag_pipeline = (
#         RunnableLambda(preprocess_input) 
#         | RunnablePassthrough.assign(
#             search_query=lambda x: rephrase_chain.invoke({
#                 "input": x["input"], 
#                 "chat_history": x.get("chat_history", [])
#             }) if "output" not in x else ""
#         )
#         | RunnablePassthrough.assign(
#             context=lambda x: format_docs(retriever.invoke(x["search_query"])) if "output" not in x and x.get("search_query") else x.get("context", "NO_CONTEXT")
#         )
#         | RunnableLambda(lambda x: x if "output" in x else generate_response(x))
#         | RunnableLambda(check_grounding)
#     )

#     return RunnableWithMessageHistory(
#         rag_pipeline,
#         get_session_history,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#         output_messages_key="output"
#     )

# # -------------------------------
# # EXECUTION
# # -------------------------------
# if __name__ == "__main__":
   
    
#     vstore = data_ingestion("done")
#     chain = build_chain(vstore)
#     config = {"configurable": {"session_id": "user_final_verification"}}

#     test_suite = ["Hi!", "What products do you have?", "Tell me more about the first one."]
    
#     for q in test_suite:
#         print(f"\nUser: {q}")
#         try:
#             response = chain.invoke({"input": q}, config=config)
#             final_output = response.get("output") if isinstance(response, dict) else response
#             print(f"Bot: {final_output}")
#         except Exception as e:
#             print(f"Bot Error: Something went wrong. ({str(e)})")



from dotenv import load_dotenv
import os
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
# MEMORY STORE
# -------------------------------------------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# -------------------------------------------------
# GUARDRAILS
# -------------------------------------------------
def safety_guard(inputs: dict) -> dict:
    """Basic safety checks for banned content and PII"""
    text = inputs.get("input", "").strip()
    low_text = text.lower()

    # 1. Banned Keywords (security)
    banned_words = ["hack", "exploit", "malware", "jailbreak", "crack", "bypass"]
    if any(word in low_text for word in banned_words):
        return {
            "input": text,
            "output": "I can't assist with that request. Please ask about products instead.",
            "guardrail_triggered": True
        }

    # 2. Basic PII Redaction (privacy)
    # Email redaction
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
    # Phone number redaction (simple pattern)
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
    # Credit card redaction (simple 16-digit pattern)
    text = re.sub(r'\b\d{16}\b', '[CARD]', text)

    return {**inputs, "input": text}

# -------------------------------------------------
# PROMPT
# -------------------------------------------------
PRODUCT_BOT_TEMPLATE = """ You are a friendly Flipkart ecommerce assistant.

CONVERSATION GUIDELINES:
- If the user says hello, greet them back as a Flipkart assistant
- If the user asks about products, use the CONTEXT provided
- If the user gives feedback like "Good product nice ❣️", acknowledge it warmly
- Be helpful, enthusiastic, and conversational
CONTEXT: {context} 
QUESTION: {input} 
ANSWER: """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PRODUCT_BOT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

# -------------------------------------------------
# BUILD RAG CHAIN WITH GUARDRAILS
# -------------------------------------------------
def build_chain(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 5})
    
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
    
    # Add safety check at the beginning
    def route_with_safety(inputs):
        # Apply safety guard first
        guarded_inputs = safety_guard(inputs)
        
        # If guardrail triggered, return early output
        if guarded_inputs.get("guardrail_triggered"):
            return guarded_inputs["output"]
        
        # Otherwise proceed with normal RAG flow
        context = format_docs(retriever.invoke(guarded_inputs["input"]))
        return prompt.invoke({
            "context": context,
            "input": guarded_inputs["input"],
            "chat_history": guarded_inputs.get("chat_history", [])
        })
    
    # Create the chain
    rag_chain = RunnableLambda(route_with_safety) | llm
    
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
    
    # Test normal query
    res1 = chain.invoke(
        {"input": "Can you tell me the best bluetooth buds?"},
        config={"configurable": {"session_id": "abhishek"}}
    )
    print("\nAnswer 1:\n", res1.content)
    
    # Test memory
    res2 = chain.invoke(
        {"input": "What was my previous question?"},
        config={"configurable": {"session_id": "abhishek"}}
    )
    print("\nAnswer 2:\n", res2.content)
    
    # Test banned keyword
    res3 = chain.invoke(
        {"input": "How to hack a product?"},
        config={"configurable": {"session_id": "abhishek"}}
    )
    print("\nAnswer 3 (should show guardrail):\n", res3.content)
    
    # Test PII redaction
    res4 = chain.invoke(
        {"input": "My email is test@example.com and phone is 1234567890"},
        config={"configurable": {"session_id": "abhishek"}}
    )
    print("\nAnswer 4 (PII should be redacted before processing):\n", res4.content)




from langchain_astradb import AstraDBVectorStore
try:
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
except ImportError as e:
    raise ImportError(
        "Missing HuggingFace embedding packages. Install sentence-transformers and transformers from requirements.txt."
    ) from e

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
from flipkart.data_converter import dataconverter
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
HF_TOKEN = os.getenv("HF_TOKEN")

required_vars = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "ASTRA_DB_API_ENDPOINT": ASTRA_DB_API_ENDPOINT,
    "ASTRA_DB_APPLICATION_TOKEN": ASTRA_DB_APPLICATION_TOKEN,
    "ASTRA_DB_KEYSPACE": ASTRA_DB_KEYSPACE,
}

missing_vars = [name for name, val in required_vars.items() if not val]
if missing_vars:
    raise RuntimeError(
        f"Missing required environment variables for deployment: {', '.join(missing_vars)}"
    )

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="BAAI/bge-base-en-v1.5"
)


def data_ingestion(status):

    vstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="flipkart",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_KEYSPACE
    )


    storage = status

    if storage is None:
        docs = dataconverter()
        insert_ids = vstore.add_documents(docs)
    
    else:
        return vstore
    return vstore, insert_ids

if __name__ == "__main__":

    vstore, insert_ids = data_ingestion(None)
    print(f"\n Inserted {len(insert_ids)} documents.")
    results = vstore.similarity_search("Can you tell me the low budget sound basshead?")
    for res in results:
        print(f"\n {res.page_content} [{res.metadata}]")





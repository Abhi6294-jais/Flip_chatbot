from flask import Flask, render_template, request
from flipkart.retrieval_generation import build_chain
from flipkart.data_ingestion import data_ingestion

# Load vector store (no re-ingestion)
vstore = data_ingestion("done")

# Build RAG + memory chain
chain = build_chain(vstore)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST", "GET"])
def chat():
    if request.method == "POST":
        msg = request.form["msg"]

        result = chain.invoke(
            {"input": msg},
            config={
                "configurable": {"session_id": "dhruv"}
            }
        )

        return str(result.content)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

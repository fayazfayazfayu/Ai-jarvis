from flask import Flask, request, jsonify, render_template
import ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
import os

app = Flask(__name__)

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

VECTOR_PATH = "memory"

# Load or create vector DB
if os.path.exists(VECTOR_PATH):
    vector_db = FAISS.load_local(VECTOR_PATH, embeddings)
else:
    vector_db = FAISS.from_documents([Document(page_content="AI Assistant initialized")], embeddings)
    vector_db.save_local(VECTOR_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json["message"]

    # Retrieve relevant memory
    docs = vector_db.similarity_search(user_message, k=3)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {user_message}
    """

    response = ollama.generate(
        model="codellama",
        prompt=prompt
    )

    return jsonify({"reply": response["response"]})


@app.route("/store", methods=["POST"])
def store():

    text = request.json["text"]

    vector_db.add_documents([Document(page_content=text)])
    vector_db.save_local(VECTOR_PATH)

    return jsonify({"status": "stored"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

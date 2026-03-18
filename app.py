"""
Flask server - take PDF, build index, answer questions
"""

from flask import Flask, request, jsonify, render_template
import pdfplumber
import io

from rag import build_index, find_relevant_chunks
from claude_client import ask_claude

app = Flask(__name__)

# Maximum size of file - 20MB
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# Routes

@app.route("/")
def index():
    """ Main page """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    take PDF, extract text, build RAG index
    return amount of chunks
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        # read PDF and extract text with pdfplumber
        pdf_bytes = file.read()
        text = extract_text(pdf_bytes)

        if not text.strip():
            return jsonify({"error": "Could not extract text from PDF"}), 400
        
        # build RAG index (chunks + embeddings)
        chunk_count = build_index(text)

        return jsonify({
            "success": True,
            "chunk_count": chunk_count,
            "filename": file.filename,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    take question, find relevant chunks,
    send to Claude, get answer
    """
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"].strip()

    if not question:
        return jsonify({"error": "Question is empty"}), 400

    try:
        # find top-4 relevant chunks
        relevant_chunks = find_relevant_chunks(question, top_k=4)

        if not relevant_chunks:
            return jsonify({"error": "No document indexed. Please upload a PDF first."}), 400

        # get answer from Claude
        answer = ask_claude(question, relevant_chunks)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Functions

def extract_text(pdf_bytes: bytes) -> str:
    """Using pdfplumber to extract text from PDF"""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# app run

if __name__ == "__main__":
    app.run(debug=True, port=5000)

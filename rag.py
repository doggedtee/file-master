# from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
import numpy as np
import anthropic
import pdfplumber
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()
model = SentenceTransformer("all-MiniLM-L6-v2") # multi-qa-mpnet-base-dot-v1
# model = TextEmbedding("BAAI/bge-small-en-v1.5")
_chunks = []
_embeddings = []

def split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 30):
    words = text.split()
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        _chunks.append(chunk)

def build_index(text):
    _chunks.clear()
    global _embeddings
    split_into_chunks(text)
    _embeddings = model.encode(_chunks)
    # _embeddings = list(model.embed(_chunks))
    return len(_chunks)


def find_relevant_chunks(question, top_k=4):
    question_embedding = model.encode([question])
    # question_embedding = list(model.embed([question]))
    scores = np.dot(_embeddings, question_embedding.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [_chunks[i] for i in top_indices]

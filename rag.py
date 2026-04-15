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
_index = {} # {filename: {"chunks": [...], "embeddings": [...]}}

def split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 30):
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i: i + chunk_size]))
    return chunks

def build_index(text, filename):
    chunks = split_into_chunks(text)
    embeddings = model.encode(chunks)
    # _embeddings = list(model.embed(_chunks))
    _index[filename] = {"chunks": chunks, "embeddings": embeddings}
    return len(chunks)


def find_relevant_chunks(question, top_k=4):
    question_embedding = model.encode([question])
    # question_embedding = list(model.embed([question]))
    all_chunks = []
    all_embeddings = []
    for filename, data in _index.items():
        all_chunks.extend(data["chunks"])
        all_embeddings.extend(data["embeddings"])
        
    all_embeddings = np.array(all_embeddings)
    scores = np.dot(all_embeddings, question_embedding.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [all_chunks[i] for i in top_indices]
        
def get_documents():
    return list(_index.keys())


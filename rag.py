# from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2") # multi-qa-mpnet-base-dot-v1
# model = TextEmbedding("BAAI/bge-small-en-v1.5")
_chunks = []
_embeddings = []
_bm25 = None

def split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 30):
    words = text.split()
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        _chunks.append(chunk)

def build_index(text):
    _chunks.clear()
    global _embeddings
    global _bm25
    split_into_chunks(text)
    _embeddings = model.encode(_chunks)
    tokenized_chunks = [chunk.lower().split() for chunk in _chunks]
    _bm25 = BM25Okapi(tokenized_chunks)
    # _embeddings = list(model.embed(_chunks))
    return len(_chunks)


def find_relevant_chunks(question, top_k=4):
    # Vector search
    question_embedding = model.encode([question])
    vector_scores = np.dot(_embeddings, question_embedding.T).flatten()
    vector_ranks = np.argsort(vector_scores)[::-1]

    # BM25 search
    tokenized_question = question.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_question)
    bm25_ranks = np.argsort(bm25_scores)[::-1]

    # Reciprocal Rank Fusion
    k = 60
    rrf_scores = np.zeros(len(_chunks))
    for rank, idx in enumerate(vector_ranks):
        rrf_scores[idx] += 1 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranks):
        rrf_scores[idx] += 1 / (k + rank + 1)
    
    top_indices = np.argsort(rrf_scores)[::-1][:top_k]
    return [_chunks[i] for i in top_indices]

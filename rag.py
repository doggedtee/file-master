"""
RAG logic: split text into chunks, store embeddings, search
"""

import math
from claude_client import get_embedding

# Storage (in-memory, one document per time)

_chunks: list[str] = []
_embeddings: list[list[float]] = []

# Chunking
def split_into_chunks(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """
    Splits text into overlapping chunks by words.
    chunk_size — how many words in one chunk
    overlap    — how many words are repeated in the next chunk
                 (to avoid losing context at boundaries)
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

# Vector math
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Measures similarity between two vectors (from -1 to 1).
    The closer to 1 — the more similar in meaning.
    """
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# Index
def build_index(text: str) -> int:
    """
    Accepts the full document text:
    1. Splits it into chunks
    2. Gets an embedding for each chunk
    3. Stores them in memory
    Returns the number of chunks.
    """
    global _chunks, _embeddings
    _chunks = split_into_chunks(text)
    _embeddings = []
    for chunk in _chunks:
        embedding = get_embedding(chunk)
        _embeddings.append(embedding)
    return len(_chunks)

def find_relevant_chunks(question: str, top_k: int = 4) -> list[str]:
    """
    Converts the question into a vector and finds the top_k
    most similar chunks using cosine similarity.
    """
    if not _chunks:
        return []
    question_embedding = get_embedding(question)

    # Calculate similarity with each chunk
    scores = [
        (i, cosine_similarity(question_embedding, emb))
        for i, emb in enumerate(_embeddings)
    ]

    # Sort by descending similarity

    scores.sort(key=lambda x: x[1], reverse=True)

    # Return the text of the top-k chunks

    return [_chunks[i] for i, _ in scores[:top_k]]
"""
All requests to Claude API - embeddings and answers
"""

import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-sonnet-4-20250514"

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
}


def get_embedding(text: str) -> list[float]:
    """
    Convert text into a vector of 128 numbers using Claude.
    Claude does not have separate embedding endpoint, so 
    we ask it to generate JSON
    """
    prompt = (
        "Return ONLY a JSON array of 128 floats between -1 and 1 "
        "that semantically represents the following text. "
        "No explanation, no markdown, just the raw JSON array.\n\n"
        f"Text: {text[:800]}"
    )

    response = httpx.post(
        API_URL,
        headers=HEADERS,
        json={
            "model": MODEL,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()

    raw = response.json()["content"][0]["text"].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: simple hash-vector if Claude returns something extra
        vec = [0.0] * 128
        for i, ch in enumerate(text):
            vec[i % 128] += ord(ch) / 1000
        mag = sum(v ** 2 for v in vec) ** 0.5 or 1
        return [v / mag for v in vec]


def ask_claude(question: str, chunks: list[str]) -> str:
    """
    send question + relevant chunks to Claude,
    return string type answer
    """
    context = "\n\n".join(
        f"[Chunk {i + 1}]: {chunk}" for i, chunk in enumerate(chunks)
    )

    response = httpx.post(
        API_URL,
        headers=HEADERS,
        json={
            "model": MODEL,
            "max_tokens": 1000,
            "system": (
                "You are a precise document analyst. "
                "Answer questions based ONLY on the provided document chunks. "
                "If the answer isn't in the chunks, say so clearly. "
                "Be concise and cite which chunk supports your answer."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": f"Document chunks:\n\n{context}\n\nQuestion: {question}",
                }
            ],
        },
        timeout=60,
    )
    response.raise_for_status()

    return response.json()["content"][0]["text"]

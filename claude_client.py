"""
All requests to Claude API (answers) and Voyage AI (embeddings)
"""

import os
from dotenv import load_dotenv
import anthropic
import voyageai

load_dotenv()

# Validate API keys at startup
_anthropic_key = os.getenv("ANTHROPIC_API_KEY")
_voyage_key = os.getenv("VOYAGE_API_KEY")

if not _anthropic_key:
    raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
if not _voyage_key:
    raise RuntimeError("VOYAGE_API_KEY is not set. Add it to your .env file.")

# Clients
_claude = anthropic.Anthropic(api_key=_anthropic_key)
_voyage = voyageai.Client(api_key=_voyage_key)

MODEL = "claude-sonnet-4-6"


def get_embedding(text: str) -> list[float]:
    """Convert text into a real semantic vector using Voyage AI."""
    result = _voyage.embed([text], model="voyage-3")
    return result.embeddings[0]


def ask_claude(question: str, chunks: list[str]) -> str:
    """Send question + relevant chunks to Claude, return answer."""
    context = "\n\n".join(
        f"[Chunk {i + 1}]: {chunk}" for i, chunk in enumerate(chunks)
    )

    message = _claude.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=(
            "You are a precise document analyst. "
            "Answer questions based ONLY on the provided document chunks. "
            "If the answer isn't in the chunks, say so clearly. "
            "Be concise and cite which chunk supports your answer."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Document chunks:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    return message.content[0].text

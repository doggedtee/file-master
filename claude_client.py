import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Validate API keys at startup
_anthropic_key = os.getenv("ANTHROPIC_API_KEY")

if not _anthropic_key:
    raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

_claude = anthropic.Anthropic(api_key=_anthropic_key)

MODEL="claude-sonnet-4-6"

TOOLS = [
    {
        "name": "search_document",
        "description": "Search the document for relevant information. Use this to find answers",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
]

def run_agent(question: str, search_fn, history) -> str:
    history.append({"role": "user", "content": question})

    while True:
        response = _claude.messages.create(
            model=MODEL,
            max_tokens=1000,
            system="You are a document analyst. Use the search_document tool ONLY when you need to find specific information from the document. If the question can be answered without searching, answer directly",
            tools=TOOLS,
            messages=history
        )

        if response.stop_reason == "tool_use":
            history.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if block.type == "tool_use":
                    query = block.input["query"]
                    chunks = search_fn(query)
                    result = "\n\n".join(chunks)

                    history.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id, 
                            "content": result
                        }]
                    })
        else:
            history.append({"role": "assistant", "content": response.content[0].text})
            return response.content[0].text

def ask_claude(question: str, search_fn, history) -> str:
    return run_agent(question, search_fn, history)

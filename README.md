# Document Oracle — PDF Chat (FastAPI + Claude)
Chat with your PDF documents using RAG and Claude API.

## Project Structure
```
file-master/
├── app.py              # FastAPI server, routes
├── rag.py              # Chunks, embeddings, semantic search
├── claude_client.py    # Claude API client
├── requirements.txt
├── .env                # Your API key
├── static/
│   ├── css/style.css   # Styles
│   └── js/main.js      # UI logic
└── templates/
    └── index.html      # HTML template
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/dam1r/file-master.git
cd file-master
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create .env file and add your Anthropic API key
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Start the server
```bash
python -m uvicorn app:app --reload --port 5000
```

### 5. Open browser
```
http://localhost:5000
```

## How it works (RAG pipeline)
```
PDF is uploaded to the server (FastAPI)
  ↓
pdfplumber extracts the text
  ↓
rag.py splits it into chunks of ~200 words
  ↓
Each chunk → vector via sentence-transformers (multi-qa-mpnet-base-dot-v1)
  ↓
User asks a question
  ↓
Question also → vector
  ↓
Dot product similarity finds top-4 relevant chunks
  ↓
Claude answers based on those chunks
```
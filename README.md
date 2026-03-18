# Document Oracle — PDF Chat (Flask + Claude)
Chat with your PDF documents using RAG and Claude API.
## Project Structure
```
pdf-chat/
├── app.py              # Flask server, routes
├── rag.py              # Chunks, embeddings, cosine search
├── claude_client.py    # Claude API requests
├── requirements.txt
├── .env                # Your API key (don't commit!)
├── static/
│   ├── css/style.css   # Styles
│   └── js/main.js      # UI logic
└── templates/
    └── index.html      # HTML template
```
## Setup
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Create .env file
```bash
cp .env.example .env
```
Open `.env` and paste your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...
```
### 3. Start the server
```bash
python app.py
```
### 4. Open browser
```
http://localhost:5000
```
## How it works (RAG pipeline)
```
PDF is uploaded to the server (Flask)
  ↓
pdfplumber extracts the text
  ↓
rag.py splits it into chunks of ~400 words
  ↓
Each chunk → vector via Claude API (claude_client.py)
  ↓
User asks a question
  ↓
Question also → vector
  ↓
cosine_similarity finds top-4 similar chunks
  ↓
Claude answers based on those chunks
```
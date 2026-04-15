from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pdfplumber
import io

from rag import build_index, find_relevant_chunks
from claude_client import ask_claude

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

_chat_history = []

@app.get("/")
async def index(request: Request):
    """ Main page """
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            return JSONResponse({"error": "Only PDF files are supported"}, status_code=400)

        try:

            # read PDF and extract text with pdfplumber
            pdf_bytes = await file.read()
            text = extract_text(pdf_bytes)

            if not text.strip():
                continue
        
            # build RAG index (chunks + embeddings)
            chunk_count = build_index(text, file.filename)
            results.append({"filename": file.filename, "chunk_count": chunk_count})

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
        
    return JSONResponse({
        "success": True,
        "documents": results
    })


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()

    if not data or "question" not in data:
        return JSONResponse({"error": "No question provided"}, status_code=400)

    question = data["question"].strip()

    if not question:
        return JSONResponse({"error": "Question is empty"}, status_code=400)

    try:
        # get answer from Claude
        answer = ask_claude(question, find_relevant_chunks, _chat_history)

        return JSONResponse({"answer": answer})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def extract_text(pdf_bytes: bytes) -> str:
    """Using pdfplumber to extract text from PDF"""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


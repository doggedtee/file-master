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

@app.get("/")
async def index(request: Request):
    """ Main page """
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported"}, status_code=400)

    try:

        # read PDF and extract text with pdfplumber
        pdf_bytes = await file.read()
        text = extract_text(pdf_bytes)

        if not text.strip():
            return JSONResponse({"error": "Could not extract text from PDF"}, status_code=400)
        
        # build RAG index (chunks + embeddings)
        chunk_count = build_index(text)

        return JSONResponse({
            "success": True,
            "chunk_count": chunk_count,
            "filename": file.filename,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()

    if not data or "question" not in data:
        return JSONResponse({"error": "No question provided"}, status_code=400)

    question = data["question"].strip()

    if not question:
        return JSONResponse({"error": "Question is empty"}, status_code=400)

    try:
        # find top-4 relevant chunks
        relevant_chunks = find_relevant_chunks(question, top_k=4)

        if not relevant_chunks:
            return JSONResponse({"error": "No document indexed. Please upload a PDF first."}, status_code=400)

        # get answer from Claude
        answer = ask_claude(question, relevant_chunks)

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


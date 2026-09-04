from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import PARSER
from app.pipeline import RAGPipeline

app = FastAPI(title="RAG Tender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

UPLOAD_DIR = Path("/app/test_tenders")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

pipeline_state = {"rag": None, "pdf": None}


class Question(BaseModel):
    question: str


@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "loaded_pdf": pipeline_state["pdf"]}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if file.filename.lower().endswith(".pdf") is False:
        raise HTTPException(400, "File must be a PDF.")

    pdf_path = UPLOAD_DIR / file.filename
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    pipeline_state["rag"] = RAGPipeline(pdf_path)
    pipeline_state["pdf"] = str(pdf_path)

    return {"message": "PDF loaded", "pdf": str(pdf_path), "parser": PARSER}


@app.post("/ask")
def ask_question(payload: Question):
    if pipeline_state["rag"] is None:
        raise HTTPException(400, "No PDF loaded. Call /upload first.")

    result = pipeline_state["rag"].ask(payload.question)

    def clean_results(items):
        cleaned = []
        for item in items:
            cleaned.append({
                "page": item.get("page"),
                "text": str(item.get("text", ""))[:500],
                "score": float(item.get("rerank_score", item.get("score", 0))),
            })
        return cleaned

    return {
        "answer": str(result["answer"]),
        "candidate_count": int(result["candidate_count"]),
        "reranked_results": clean_results(result.get("reranked_results", [])),
    }
import gc
import io
import re
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pipeline import RAGPipeline
from multi_document_test import (
    MultiDocumentPipeline,
    get_test_pdfs,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Tender RAG API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = (
    Path("data")
    / "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ENGINE STATE
# ============================================================

single_rag = None
single_pdf_name = None

multi_rag = None
multi_pdf_names = []


# ============================================================
# REQUEST MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str


# ============================================================
# ENGINE RESET
# ============================================================

def clear_single_engine():
    """
    Release the single-document engine.
    """

    global single_rag
    global single_pdf_name

    single_rag = None
    single_pdf_name = None

    gc.collect()


def clear_multi_engine():
    """
    Release the multi-document engine.
    """

    global multi_rag
    global multi_pdf_names

    multi_rag = None
    multi_pdf_names = []

    gc.collect()


# ============================================================
# SINGLE ENGINE HELPERS
# ============================================================

def initialize_single_rag(
    pdf_path: Path
):
    """
    Initialize the existing single-document RAGPipeline
    while capturing internal stdout/stderr.
    """

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    with redirect_stdout(captured_stdout):
        with redirect_stderr(captured_stderr):

            rag = RAGPipeline(
                pdf_path
            )

    return rag


def ask_single_rag(
    rag,
    question: str
):
    """
    Run the single-document engine while keeping the API
    response clean.
    """

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    with redirect_stdout(captured_stdout):
        with redirect_stderr(captured_stderr):

            return rag.ask(
                question
            )


# ============================================================
# MULTI ENGINE HELPERS
# ============================================================

def initialize_multi_rag(
    pdf_paths
):
    """
    Initialize the existing MultiDocumentPipeline.
    """

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    with redirect_stdout(captured_stdout):
        with redirect_stderr(captured_stderr):

            rag = MultiDocumentPipeline(
                pdf_paths
            )

    return rag


def ask_multi_rag(
    rag,
    question: str
):
    """
    Run the multi-document engine while keeping the API
    response clean.
    """

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    with redirect_stdout(captured_stdout):
        with redirect_stderr(captured_stderr):

            return rag.ask(
                question
            )


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_single_sources(
    answer: str
):
    """
    Extract only pages explicitly cited by the LLM.

    Example:
        [Page 11]
    """

    sources = []
    seen = set()

    matches = re.findall(
        r"\[Page\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE,
    )

    for page in matches:

        source = (
            f"Page {page}"
        )

        if source not in seen:

            seen.add(
                source
            )

            sources.append(
                source
            )

    return sources


def extract_multi_sources(
    answer: str
):
    """
    Extract document + page citations from multi-document
    answers.

    Supports both:

        [SOURCE: document.pdf | Page 7]

    and:

        [document.pdf | Page 7]
    """

    sources = []
    seen = set()

    # --------------------------------------------------------
    # [SOURCE: document.pdf | Page 7]
    # --------------------------------------------------------

    source_matches = re.findall(
        r"\[SOURCE:\s*"
        r"([^\|\]]+?)\s*\|\s*"
        r"Page\s+(\d+)"
        r"(?:\s*\|[^\]]*)?"
        r"\]",
        answer,
        flags=re.IGNORECASE,
    )

    for document, page in source_matches:

        source = (
            f"{document.strip()} | "
            f"Page {page}"
        )

        if source not in seen:

            seen.add(
                source
            )

            sources.append(
                source
            )

    # --------------------------------------------------------
    # [document.pdf | Page 7]
    # --------------------------------------------------------

    normal_matches = re.findall(
        r"\["
        r"([^\[\]\|]+?\.pdf)"
        r"\s*\|\s*"
        r"Page\s+(\d+)"
        r"(?:\s*\|[^\]]*)?"
        r"\]",
        answer,
        flags=re.IGNORECASE,
    )

    for document, page in normal_matches:

        source = (
            f"{document.strip()} | "
            f"Page {page}"
        )

        if source not in seen:

            seen.add(
                source
            )

            sources.append(
                source
            )

    return sources


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "single_pdf_loaded": (
            single_rag is not None
        ),
        "single_pdf": single_pdf_name,
        "multi_document_loaded": (
            multi_rag is not None
        ),
        "multi_document_count": len(
            multi_pdf_names
        ),
    }


# ============================================================
# SINGLE PDF STATUS
# ============================================================

@app.get("/api/single/status")
def single_status():

    return {
        "ready": (
            single_rag is not None
        ),
        "document": single_pdf_name,
    }


# ============================================================
# SINGLE PDF UPLOAD
# ============================================================

@app.post("/api/single/upload")
async def upload_single_pdf(
    file: UploadFile = File(...)
):

    global single_rag
    global single_pdf_name

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    safe_name = Path(
        file.filename
    ).name

    pdf_path = (
        UPLOAD_DIR
        / safe_name
    )

    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    try:

        contents = (
            await file.read()
        )

        pdf_path.write_bytes(
            contents
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to save PDF: {exc}"
            ),
        )

    # --------------------------------------------------------
    # CLEAR OTHER ENGINE
    # --------------------------------------------------------

    clear_multi_engine()

    # --------------------------------------------------------
    # INITIALIZE SINGLE ENGINE
    # --------------------------------------------------------

    init_start = (
        time.perf_counter()
    )

    try:

        rag = initialize_single_rag(
            pdf_path
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"RAG initialization failed: {exc}"
            ),
        )

    init_time = (
        time.perf_counter()
        - init_start
    )

    single_rag = rag
    single_pdf_name = safe_name

    return {
        "status": "ready",
        "document": safe_name,
        "initialization_time_sec": round(
            init_time,
            3,
        ),
        "message": (
            "Single-document RAG engine is ready."
        ),
    }


# ============================================================
# SINGLE PDF ASK
# ============================================================

@app.post("/api/single/ask")
def ask_single(
    request: AskRequest
):

    if single_rag is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "No single PDF is loaded. "
                "Upload a PDF first."
            ),
        )

    question = (
        request.question.strip()
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    start = (
        time.perf_counter()
    )

    try:

        result = ask_single_rag(
            single_rag,
            question,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Query failed: {exc}"
            ),
        )

    query_time = (
        time.perf_counter()
        - start
    )

    if not isinstance(
        result,
        dict
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected RAG response."
            ),
        )

    answer = result.get(
        "answer",
        "",
    )

    reranked_results = result.get(
        "reranked_results",
        [],
    )

    sources = extract_single_sources(
        answer
    )

    return {
        "status": "success",
        "document": single_pdf_name,
        "question": question,
        "answer": answer,
        "sources": sources,
        "query_time_sec": round(
            query_time,
            3,
        ),
        "evidence_count": len(
            reranked_results
        ),
    }


# ============================================================
# MULTI DOCUMENT STATUS
# ============================================================

@app.get("/api/multi/status")
def multi_status():

    return {
        "ready": (
            multi_rag is not None
        ),
        "document_count": len(
            multi_pdf_names
        ),
        "documents": multi_pdf_names,
    }


# ============================================================
# MULTI DOCUMENT START
# ============================================================

@app.post("/api/multi/start")
def start_multi_document():

    global multi_rag
    global multi_pdf_names

    # --------------------------------------------------------
    # ALREADY READY
    # --------------------------------------------------------

    if multi_rag is not None:

        return {
            "status": "ready",
            "document_count": len(
                multi_pdf_names
            ),
            "documents": multi_pdf_names,
            "message": (
                "Multi-document RAG engine is already ready."
            ),
        }

    # --------------------------------------------------------
    # FIND EXISTING BACKEND PDFs
    # --------------------------------------------------------

    pdfs = get_test_pdfs()

    if not pdfs:

        raise HTTPException(
            status_code=500,
            detail=(
                "No multi-document tender PDFs "
                "were found in test_tenders."
            ),
        )

    # --------------------------------------------------------
    # CLEAR SINGLE ENGINE
    # --------------------------------------------------------

    clear_single_engine()

    # --------------------------------------------------------
    # INITIALIZE ENGINE
    # --------------------------------------------------------

    init_start = (
        time.perf_counter()
    )

    try:

        rag = initialize_multi_rag(
            pdfs
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Multi-document initialization "
                f"failed: {exc}"
            ),
        )

    init_time = (
        time.perf_counter()
        - init_start
    )

    multi_rag = rag

    multi_pdf_names = [
        pdf.name
        for pdf in pdfs
    ]

    return {
        "status": "ready",
        "document_count": len(
            multi_pdf_names
        ),
        "documents": multi_pdf_names,
        "initialization_time_sec": round(
            init_time,
            3,
        ),
        "message": (
            "Multi-document RAG engine is ready."
        ),
    }


# ============================================================
# MULTI DOCUMENT ASK
# ============================================================

@app.post("/api/multi/ask")
def ask_multi(
    request: AskRequest
):

    if multi_rag is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Multi-document engine is not ready. "
                "Start the engine first."
            ),
        )

    question = (
        request.question.strip()
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    start = (
        time.perf_counter()
    )

    try:

        result = ask_multi_rag(
            multi_rag,
            question,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Multi-document query failed: "
                f"{exc}"
            ),
        )

    query_time = (
        time.perf_counter()
        - start
    )

    if not isinstance(
        result,
        dict
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected multi-document "
                "RAG response."
            ),
        )

    answer = result.get(
        "answer",
        "",
    )

    reranked_results = result.get(
        "reranked_results",
        [],
    )

    sources = extract_multi_sources(
        answer
    )

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "sources": sources,
        "query_time_sec": round(
            query_time,
            3,
        ),
        "evidence_count": len(
            reranked_results
        ),
        "document_count": len(
            multi_pdf_names
        ),
    }
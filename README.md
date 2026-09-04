3. Hybrid Multi-Stage Retrieval

The system combines three retrieval approaches.

Dense Retrieval

Dense semantic retrieval is implemented using FAISS.

Model:

sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:

384

Dense retrieval is useful when the question and document use different but semantically related wording.

BM25 Retrieval

BM25 provides lexical keyword-based retrieval.

It is useful for exact matches involving technical terminology, clause names, numbers, item descriptions, and procurement-specific wording.

SPLADE Retrieval

SPLADE provides sparse neural retrieval.

Model:

ibm-granite/granite-embedding-30m-sparse

Configuration:

Document max active dimensions: 192
Query max active dimensions: 50

The outputs from FAISS, BM25 and SPLADE are combined into a unique candidate pool.

4. CrossEncoder Reranking

The retrieved candidates are reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

Initial retrieval is designed to provide candidate recall, while CrossEncoder reranking improves the final relevance ordering.

Current configuration:

retrieval_top_k: 10
rerank_top_k: 5

The reranker operates on CPU when CUDA is unavailable.

5. Table-Aware Retrieval

Tender questions frequently refer to structured information such as:

Material schedules
Material lists
Bills of quantities
Schedules of quantities
Item quantities
Procurement tables

A normal top-k retrieval strategy may return only a subset of rows from the same table.

To address this, the system implements table-aware retrieval.

Workflow
Table/List Query
       ↓
Detect table-oriented intent
       ↓
Retrieve relevant table rows
       ↓
Identify parent table
       ↓
Find sibling rows
       ↓
Reconstruct complete table
       ↓
Rerank reconstructed context
       ↓
Generate answer

The mechanism uses metadata such as:

page
table_index
row_index
type

to associate individual rows with their parent table.

The reconstructed rows are ordered using their original row positions.

6. Page-Level Evidence

Retrieved passages retain their document page number.

The API can return evidence such as:

Page 8
Page 14
Page 27

together with retrieved text and relevance scores.

This helps users verify generated answers against the source tender document.

7. Grounded LLM Generation

The final answer is generated using retrieved tender evidence as context.

LLM provider:

Groq

Model:

openai/gpt-oss-20b

The application uses an OpenAI-compatible client to communicate with the Groq API.

The current design uses one generation call per user question.

8. FastAPI REST API

The RAG pipeline is exposed through FastAPI.

The backend provides endpoints for:

Health checking
Question answering
Document interaction
Frontend serving
9. Tender Desk Web UI

The project includes a browser-based interface called Tender Desk.

The UI provides:

Tender interaction
Question input
Generated answers
Evidence display
Page references
Retrieval scores
Backend/API connectivity

The frontend is served directly by FastAPI.

10. Dockerized Deployment

The complete application can run inside Docker.

Docker provides:

Consistent Python runtime
Reproducible dependency installation
Encapsulated application environment
Persistent volume support
Simple startup using Docker Compose

After pasting, press:

```text
Ctrl + D
Step 2 — append the second chunk

Run:

cat >> README.md

Paste:

---

# System Architecture

```text
                         Tender PDF
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Document Processing  │
                  │ Docling + PyMuPDF    │
                  └──────────┬───────────┘
                             │
                             ▼
                  Page-aware document data
                  Text chunks + table rows
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Multi-Stage Search   │
                  ├──────────────────────┤
                  │ FAISS                │
                  │ BM25                 │
                  │ SPLADE               │
                  └──────────┬───────────┘
                             │
                             ▼
                   Unique Candidate Pool
                             │
                             ▼
                  ┌──────────────────────┐
                  │ CrossEncoder         │
                  │ Reranker             │
                  └──────────┬───────────┘
                             │
                             ▼
                 Table-Aware Reconstruction
                     when required
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Groq LLM             │
                  │ GPT-OSS-20B          │
                  └──────────┬───────────┘
                             │
                             ▼
                     Grounded Answer
                             │
                             ▼
                  ┌──────────────────────┐
                  │ FastAPI API          │
                  │ Tender Desk UI       │
                  └──────────────────────┘
End-to-End Workflow
1. Tender PDF is provided
          ↓
2. Document is parsed
          ↓
3. Text, tables and metadata are extracted
          ↓
4. Text is divided into chunks
          ↓
5. Dense embeddings are generated
          ↓
6. Sparse representations are generated
          ↓
7. FAISS, BM25 and SPLADE indexes are prepared
          ↓
8. Processing artifacts are cached
          ↓
9. User submits a question
          ↓
10. Multiple retrieval methods find candidates
          ↓
11. Candidate results are deduplicated
          ↓
12. CrossEncoder reranks candidates
          ↓
13. Table-aware reconstruction is applied when needed
          ↓
14. Relevant context is prepared
          ↓
15. Groq LLM generates a grounded answer
          ↓
16. Answer and evidence are returned to the UI
Retrieval Architecture
Candidate Retrieval

The first retrieval stage combines:

FAISS
+
BM25
+
SPLADE

Conceptually:

FAISS candidates
        +
BM25 candidates
        +
SPLADE candidates
        ↓
Unique candidate pool

The goal of this stage is to maximize retrieval recall across different types of questions.

Candidate Reranking

The candidate pool is passed to the CrossEncoder.

Question
   +
Candidate passage
   ↓
CrossEncoder
   ↓
Relevance score

The candidates are reordered according to their question-to-passage relevance.

Why Hybrid Retrieval?

Different retrieval methods have different strengths:

FAISS
→ semantic similarity

BM25
→ exact lexical matching

SPLADE
→ sparse neural matching

CrossEncoder
→ deeper question/passage relevance

Combining these methods provides a more robust retrieval pipeline than relying on only one retrieval mechanism.

Table-Aware Retrieval

Table information is stored using row-level metadata.

Example structure:

page
type
table_index
row_index
path
text

For a table query, the system:

Detects table/list intent.
Retrieves relevant table rows.
Groups rows using the table index.
Selects the strongest parent table.
Retrieves sibling rows belonging to that table.
Orders rows by row index.
Reranks the reconstructed context.
Sends the complete structured context to the LLM.

Example query:

What items are listed in the material schedule and what are their quantities?

Instead of returning only one matching row, the system can reconstruct the complete material schedule.
# Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| PDF Processing | Docling |
| PDF Utilities | PyMuPDF |
| Text Splitting | LangChain Text Splitters |
| Dense Retrieval | FAISS |
| Keyword Retrieval | BM25 |
| Sparse Retrieval | SPLADE |
| Dense Embeddings | Sentence Transformers |
| Reranking | CrossEncoder |
| LLM Provider | Groq |
| LLM | GPT-OSS-20B |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Frontend | HTML / CSS / JavaScript |
| Configuration | YAML |
| Environment Variables | python-dotenv |
| Containerization | Docker |
| Orchestration | Docker Compose |

---

# Models

## Dense Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`

Embedding dimension:

`384`

## Sparse Retrieval Model

`ibm-granite/granite-embedding-30m-sparse`

Configuration:

- Document max active dimensions: 192
- Query max active dimensions: 50

## CrossEncoder Reranker

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The reranker uses CPU in environments where compatible CUDA acceleration is not available.

## Generation Model

`openai/gpt-oss-20b`

The model is accessed through Groq's OpenAI-compatible API.

---

# Project Structure

```text
Rag/
│
├── app/
│   ├── api.py
│   ├── cache.py
│   ├── chunker.py
│   ├── config.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── keyword_search.py
│   ├── logger.py
│   ├── parser.py
│   ├── pipeline.py
│   ├── reranker.py
│   ├── sparse_search.py
│   ├── vector_store.py
│   │
│   └── static/
│       └── index.html
│
├── evaluation/
│   ├── evaluate.py
│   └── questions.json
│
├── test_tenders/
│   └── Tender PDF documents
│
├── data/
│   └── Cached processing and retrieval artifacts
│
├── config.yaml
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── main.py
└── README.md
Configuration

The main configuration file is:

config.yaml

Current settings:

parser: "docling"

chunk_size: 700
chunk_overlap: 200

embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_dimension: 384

sparse_model: "ibm-granite/granite-embedding-30m-sparse"
sparse_document_max_active_dims: 192
sparse_query_max_active_dims: 50

retrieval_top_k: 10
rerank_top_k: 5

reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

llm_provider: "groq"
llm_model: "openai/gpt-oss-20b"
Environment Variables

The application requires a Groq API key.

Create a local .env file:

GROQ_API_KEY=your_groq_api_key_here

Never commit the real API key to Git.

A repository-safe .env.example should contain a placeholder rather than a real secret.

Installation
Clone the Repository
git clone https://github.com/walmikingle/A-Multi-Modal-Framework-for-Tender-Compliance-and-Risk-Assessment..git
cd A-Multi-Modal-Framework-for-Tender-Compliance-and-Risk-Assessment.
Create a Virtual Environment
python3 -m venv .venv

Activate it:

source .venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Running Locally

The command-line application can be started with:

python main.py --pdf /path/to/tender.pdf

The application initializes the RAG pipeline and provides an interactive question interface.

Running the API Locally

The FastAPI application can be started with:

uvicorn app.api:app --host 0.0.0.0 --port 8000

Once started:

http://localhost:8000
Running with Docker

Docker Compose is the recommended way to run the packaged application.

Build the Image
docker compose build
Start the Application
docker compose up

After startup, the application should report:

RAG pipeline ready

The web interface is available at:

http://localhost:8000/
Docker Architecture

The Docker image contains the application runtime and Python dependencies.

The current Docker setup uses Python 3.11.

The image installs system dependencies required by the PDF/document-processing and machine-learning libraries.

The application code is copied into:

/app

Docker Compose manages the application container and its persistent volumes.

Docker Persistence

The Docker Compose configuration keeps large or reusable data outside the image.

RAG Cache
Host:
./data

Container:
/app/data

This preserves:

Processed document artifacts
Retrieval indexes
Cached tender data
Tender PDFs
Host:
./test_tenders

Container:
/app/test_tenders
Hugging Face Cache
Host:
~/.cache/huggingface

Container:
/root/.cache/huggingface

This reduces unnecessary model downloads when containers are recreated.

Web Interface

The frontend is served directly by FastAPI.

Frontend location:

app/static/index.html

The root endpoint serves the frontend:

GET /

Static resources are exposed through:

/static

The UI communicates with the backend API and presents generated answers and retrieved evidence.

REST API
Health Check
GET /health

Used to verify that the service is available.

Ask a Question
POST /ask

Example request:

{
  "question": "What is the penalty for delay in completion of the contract?"
}

The response contains:

Generated answer
Candidate count
Reranked evidence
Source page information
Retrieval scores
Swagger Documentation

FastAPI automatically provides interactive API documentation.

Open:

http://localhost:8000/docs

Swagger can be used to:

Test endpoints
Submit questions
Inspect API requests
Inspect API responses
Verify backend functionality
Caching

Document processing and retrieval-index creation can be expensive, especially on CPU-only systems.

The application therefore stores generated artifacts in the cache.

The cache uses document identity and relevant configuration information to determine whether previously generated artifacts can be reused.

Cache Workflow

Without caching:

PDF
 ↓
Parse
 ↓
Chunk
 ↓
Embed
 ↓
Build indexes

With caching:

PDF
 ↓
Check cache
 ↓
Cache found
 ↓
Load existing artifacts

This substantially reduces repeated processing time.

Evaluation

The project contains an evaluation workflow under:

evaluation/

The evaluation dataset contains:

55 questions
11 tender documents

This corresponds to:

605 question-document combinations

The completed manual evaluation achieved approximately:

96% overall performance

The evaluation includes questions covering:

Contract conditions
Penalties
Tender requirements
Material schedules
Quantities
Procurement information
Compliance-related information

Some weaker cases remain in documents where OCR quality or document structure makes retrieval more difficult.

Evaluation Usage

The evaluation script is located at:

evaluation/evaluate.py

Example:

python evaluation/evaluate.py

The required API key, test documents and environment should be configured before running evaluation.

Performance Considerations

The system is designed to operate without requiring an NVIDIA GPU.

CPU inference is supported for:

Dense embeddings
SPLADE
CrossEncoder reranking

Conservative processing parameters are used to support lower-memory systems.

Caching is especially important because document processing and model loading can otherwise be expensive.

The pipeline also limits the amount of information passed to the final generation stage to control latency and token usage.

Limitations
Document Quality

Poor-quality PDFs, scans and OCR-heavy documents can reduce extraction and retrieval quality.

OCR

The current stable configuration does not depend on aggressive OCR processing.

Complex Tables

Highly irregular or multi-page tables may require additional table-reconstruction logic.

CPU Performance

The system works on CPU hardware but inference and document processing can be significantly faster on suitable accelerated hardware.

LLM Dependency

Final answer generation depends on access to the configured Groq API.

Local Deployment

The current Docker setup is primarily intended for local or controlled environments rather than unrestricted public exposure.

Future Improvements
Advanced OCR

Improve support for scanned and image-heavy tender documents.

Multimodal Document Understanding

Incorporate:

Page images
Diagrams
Charts
Visual layouts
Tables
Embedded figures

into retrieval and reasoning.

Advanced Table Reasoning

Future table capabilities could include:

Multi-page tables
Cross-table reasoning
Table comparison
Table arithmetic
Automatic table summarization
Complex row/column relationships
Automated Compliance Scoring

The system could compare tender requirements against retrieved evidence.

Example:

Requirement
     ↓
Evidence Retrieval
     ↓
Compliance Analysis
     ↓
Compliant / Non-Compliant / Unclear
Risk Assessment

Add automated risk identification and prioritization.

Possible categories:

Commercial risk
Contractual risk
Technical risk
Delivery risk
Compliance risk
Tender Comparison

Support comparison between multiple tender documents, suppliers or procurement opportunities.

Production Deployment

Potential improvements include:

Authentication
Role-based access control
HTTPS
Cloud deployment
Persistent databases
Monitoring
Logging
Rate limiting
Horizontal scaling
Security Considerations

The current application is primarily designed for local or controlled deployment.

Production deployments should implement:

Authentication
Authorization
HTTPS
Secure secret management
Restricted CORS
File-upload validation
File-size limits
API rate limiting
Container hardening
Logging and monitoring

API keys must never be hard-coded into source files or committed to GitHub.

Development Workflow

A recommended workflow is:

Modify code
    ↓
Run local tests
    ↓
Test RAG query
    ↓
Test API
    ↓
Test UI
    ↓
Build Docker image
    ↓
Run Docker container
    ↓
Test Dockerized application
    ↓
Run evaluation
    ↓
Commit changes
    ↓
Push to GitHub
Reproducibility

The repository provides the main components needed to reproduce the application:

requirements.txt
config.yaml
Dockerfile
compose.yaml
.dockerignore
Application source code
Evaluation scripts
Docker volume configuration
Environment variable configuration

Docker provides a consistent Python runtime and dependency environment across systems.

Project Status

The current project includes:

✅ PDF document processing
✅ Page-aware chunking
✅ Dense retrieval
✅ BM25 retrieval
✅ SPLADE retrieval
✅ Candidate deduplication
✅ CrossEncoder reranking
✅ Table-aware retrieval
✅ Complete table reconstruction
✅ Cached document processing
✅ Page-level evidence
✅ Groq LLM generation
✅ FastAPI REST API
✅ Swagger API documentation
✅ Tender Desk web interface
✅ Docker deployment
✅ Persistent Docker volumes
✅ Evaluation workflow
Author

Walmik

Engineering Student / Developer

Project:

A Multi-Modal Framework for Tender Compliance and Risk Assessment

License

This project currently does not specify a final open-source license.

Before public redistribution, select an appropriate license such as MIT, Apache-2.0, or another license compatible with the project and its dependencies.

Acknowledgements

This project uses open-source technologies and models including:

Docling
PyMuPDF
FAISS
BM25
Sentence Transformers
SPLADE
Transformers
FastAPI
Docker
Groq
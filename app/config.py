import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

EMBEDDING_DIMENSION = 384

CHUNK_SIZE = 700
CHUNK_OVERLAP = 200

TOP_K = 5

LLM_MODEL = "openai/gpt-oss-20b"

DATA_DIR = "data"
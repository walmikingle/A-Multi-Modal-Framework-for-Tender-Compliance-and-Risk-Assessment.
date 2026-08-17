from .config import DATA_DIR, TOP_K

from .chunker import get_text_splitter
from .parser import create_directories, process_pdf
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .generator import Generator


class RAGPipeline:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

        create_directories(
            DATA_DIR
        )

        splitter = get_text_splitter()

        print("Loading PDF...")

        self.items = process_pdf(
            pdf_path,
            DATA_DIR,
            splitter
        )

        print(
            f"Extracted {len(self.items)} items."
        )

        print("Loading embedding model...")

        self.embedding_service = (
            EmbeddingService()
        )

        print("Generating embeddings...")

        self.items = (
            self.embedding_service
            .embed_items(self.items)
        )

        print("Building FAISS index...")

        self.vector_store = VectorStore()

        self.vector_store.build(
            self.items
        )

        print(
            f"Indexed "
            f"{self.vector_store.index.ntotal} vectors."
        )

        self.generator = Generator()

    def ask(self, question):

        query_embedding = (
            self.embedding_service
            .embed(question)
        )

        matched_items = (
            self.vector_store
            .search(
                query_embedding,
                TOP_K
            )
        )

        print(
            f"Retrieved "
            f"{len(matched_items)} chunks."
        )

        return self.generator.generate(
            question,
            matched_items
        )
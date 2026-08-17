from .config import DATA_DIR, TOP_K
from .keyword_search import KeywordSearch
from .chunker import get_text_splitter
from .parser import create_directories, process_pdf
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .generator import Generator


class RAGPipeline:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

        # -------------------------
        # Create required directories
        # -------------------------

        create_directories(DATA_DIR)

        # -------------------------
        # Initialize text splitter
        # -------------------------

        splitter = get_text_splitter()

        # -------------------------
        # Process PDF
        # -------------------------

        print("Loading PDF...")

        self.items = process_pdf(
            pdf_path,
            DATA_DIR,
            splitter
        )

        print(
            f"Extracted {len(self.items)} items."
        )

        # -------------------------
        # Initialize embeddings
        # -------------------------

        print("Loading embedding model...")

        self.embedding_service = EmbeddingService()

        # -------------------------
        # Generate embeddings
        # -------------------------

        print("Generating embeddings...")

        self.items = self.embedding_service.embed_items(
            self.items
        )

        # -------------------------
        # Build FAISS index
        # -------------------------

        print("Building FAISS index...")

        self.vector_store = VectorStore()

        self.vector_store.build(
            self.items
        )

        print(
            f"Indexed "
            f"{self.vector_store.index.ntotal} vectors."
        )

        # -------------------------
        # Initialize BM25
        # -------------------------

        print("Building keyword search index...")

        self.keyword_search = KeywordSearch(
            self.items
        )

        print("Keyword search index ready.")

        # -------------------------
        # Initialize LLM generator
        # -------------------------

        print("Initializing generator...")

        self.generator = Generator()

        print("RAG pipeline ready.")

    def ask(self, question):

        # -------------------------
        # Generate query embedding
        # -------------------------

        query_embedding = self.embedding_service.embed(
            question
        )

        # -------------------------
        # Semantic Search - FAISS
        # -------------------------

        semantic_results = self.vector_store.search(
            query_embedding,
            TOP_K
        )

        # -------------------------
        # Keyword Search - BM25
        # -------------------------

        keyword_results = self.keyword_search.search(
            question,
            TOP_K
        )

        # -------------------------
        # Display Semantic Results
        # -------------------------

        print("\n--- Semantic Results (FAISS) ---")

        for i, item in enumerate(
            semantic_results,
            start=1
        ):

            print(
                f"\n{i}. Page {item['page']}"
            )

            print(
                item["text"][:150]
            )

        # -------------------------
        # Display Keyword Results
        # -------------------------

        print("\n--- Keyword Results (BM25) ---")

        for i, item in enumerate(
            keyword_results,
            start=1
        ):

            print(
                f"\n{i}. Page {item['page']} "
                f"| Score: {item['keyword_score']:.2f}"
            )

            print(
                item["text"][:150]
            )

        # -------------------------
        # Current RAG generation
        # -------------------------
        #
        # For Day 2, we are still
        # using FAISS results for
        # the final answer.
        #
        # We will combine FAISS +
        # BM25 results after verifying
        # both retrieval methods.
        # -------------------------

        print(
            f"\nUsing {len(semantic_results)} "
            f"semantic results for generation."
        )

        answer = self.generator.generate(
            question,
            semantic_results
        )

        return answer
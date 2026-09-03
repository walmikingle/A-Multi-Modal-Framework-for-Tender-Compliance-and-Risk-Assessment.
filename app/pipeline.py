from pathlib import Path
from .config import (
    DATA_DIR,
    PARSER,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    SPARSE_MODEL,
    SPARSE_DOCUMENT_MAX_ACTIVE_DIMS,
    SPARSE_QUERY_MAX_ACTIVE_DIMS,
    RERANKER_MODEL
)

from .cache import (
    DocumentCache,
    create_config_fingerprint
)

from .reranker import Reranker
from .keyword_search import KeywordSearch
from .chunker import get_text_splitter
from .parser import (
    process_pdf,
    get_document_output_dir
)
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .generator import Generator
from .sparse_search import SparseSearch
from .logger import logger


class RAGPipeline:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

        logger.info(
            "Initializing RAG pipeline | "
            f"PDF={pdf_path} | "
            f"Parser={PARSER}"
        )

        # ==================================================
        # CREATE REQUIRED DIRECTORIES
        # ==================================================



        cache_config = {
    "parser": PARSER,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "embedding_dimension": EMBEDDING_DIMENSION,
    "embedding_model": EMBEDDING_MODEL,
    "sparse_model": SPARSE_MODEL,
    "sparse_document_max_active_dims": (
        SPARSE_DOCUMENT_MAX_ACTIVE_DIMS
    ),
    "sparse_query_max_active_dims": (
        SPARSE_QUERY_MAX_ACTIVE_DIMS
    ),
    "reranker_model": RERANKER_MODEL,
}

        config_fingerprint = (
    create_config_fingerprint(
        cache_config
    )
)

        # ==================================================
        # INITIALIZE CACHE
        # ==================================================

        cache_dir = (
            Path(DATA_DIR)
            / "cache"
        )

        self.cache = DocumentCache(
            cache_dir
        )

        cache_path = (
            self.cache.get_cache_path(
                pdf_path
            )
        )

        # ==================================================
        # CHECK CACHE
        # ==================================================

        if self.cache.exists(
    pdf_path,
    parser=PARSER,
    config_fingerprint=config_fingerprint
):

            logger.info(
                "Cache hit | "
                f"PDF={pdf_path} | "
                f"Parser={PARSER}"
            )

            print(
                "\n" + "=" * 60
            )

            print(
                "CACHE FOUND"
            )

            print(
                "=" * 60
            )

            print(
                "Loading processed document "
                "from cache..."
            )

            # -------------------------
            # Load processed items
            # -------------------------

            self.items = (
                self.cache.load_items(
                    pdf_path
                )
            )

            print(
                f"Loaded {len(self.items)} items "
                "from cache."
            )

            logger.info(
                "Loaded cached items | "
                f"Items={len(self.items)}"
            )

            # -------------------------
            # Load FAISS index
            # -------------------------

            print(
                "Loading FAISS index from cache..."
            )

            self.vector_store = (
                VectorStore()
            )

            self.vector_store.load(
                cache_path / "faiss.index",
                self.items
            )

            print(
                f"Loaded "
                f"{self.vector_store.index.ntotal} "
                "vectors."
            )

            logger.info(
                "Loaded FAISS index | "
                f"Vectors="
                f"{self.vector_store.index.ntotal}"
            )

            # -------------------------
            # Load SPLADE embeddings
            # -------------------------

            sparse_embeddings = (
                self.cache.load_sparse_embeddings(
                    pdf_path
                )
            )

            # -------------------------
            # Initialize sparse search
            # -------------------------

            print(
                "Initializing sparse search "
                "from cache..."
            )

            self.sparse_search = (
                SparseSearch(
                    self.items,
                    cached_embeddings=sparse_embeddings
                )
            )

            print(
                "Sparse search index ready."
            )

            logger.info(
                "Loaded SPLADE index | "
                f"Documents="
                f"{len(self.sparse_search.items)}"
            )

            print(
                "\nCache loading complete."
            )

            logger.info(
                "Cache loading complete"
            )

        # ==================================================
        # BUILD EVERYTHING
        # ==================================================

        else:

            logger.info(
                "Cache miss | "
                f"PDF={pdf_path} | "
                f"Parser={PARSER}"
            )

            print(
                "\n" + "=" * 60
            )

            print(
                "CACHE NOT FOUND"
            )

            print(
                "=" * 60
            )

            # -------------------------
            # Initialize text splitter
            # -------------------------

            splitter = (
                get_text_splitter()
            )

            # -------------------------
            # Process PDF
            # -------------------------

            print(
                "Loading PDF..."
            )

            print(
                f"Using parser: {PARSER}"
            )

            document_output_dir = (
    get_document_output_dir(
        pdf_path,
        DATA_DIR
    )
)

            logger.info(
                "Document output directory created | "
                f"Directory={document_output_dir}"
            )

            print(
                f"Document output directory: "
                f"{document_output_dir}"
            )

            self.items = process_pdf(
                pdf_path,
                document_output_dir,
                splitter,
                parser=PARSER
            )

            logger.info(
                "Document processing complete | "
                f"Items={len(self.items)} | "
                f"Parser={PARSER}"
            )

            print(
                f"Extracted {len(self.items)} items."
            )

            # -------------------------
            # Initialize embedding service
            # -------------------------

            print(
                "Loading embedding model..."
            )

            self.embedding_service = (
                EmbeddingService()
            )

            # -------------------------
            # Generate dense embeddings
            # -------------------------

            print(
                "Generating embeddings..."
            )

            self.items = (
                self.embedding_service.embed_items(
                    self.items
                )
            )

            logger.info(
                "Dense embeddings generated"
            )

            # -------------------------
            # Build FAISS index
            # -------------------------

            print(
                "Building FAISS index..."
            )

            self.vector_store = (
                VectorStore()
            )

            self.vector_store.build(
                self.items
            )

            logger.info(
                "FAISS index built | "
                f"Vectors="
                f"{self.vector_store.index.ntotal}"
            )

            print(
                f"Indexed "
                f"{self.vector_store.index.ntotal} "
                "vectors."
            )

            # -------------------------
            # Build SPLADE sparse index
            # -------------------------

            print(
                "Building sparse search index..."
            )

            self.sparse_search = (
                SparseSearch(
                    self.items
                )
            )

            logger.info(
                "SPLADE index built | "
                f"Documents="
                f"{len(self.sparse_search.items)}"
            )

            print(
                "Sparse search index ready."
            )

            # ==================================================
            # SAVE CACHE
            # ==================================================

            print(
                "\nSaving cache..."
            )

            # -------------------------
            # Save processed items
            # -------------------------

            self.cache.save_items(
                pdf_path,
                self.items
            )

            # -------------------------
            # Save FAISS index
            # -------------------------

            self.vector_store.save(
                cache_path / "faiss.index"
            )

            # -------------------------
            # Save SPLADE embeddings
            # -------------------------

            self.sparse_search.save(
                cache_path
                / "sparse_embeddings.pkl"
            )

            # -------------------------
            # Save metadata
            # -------------------------

            self.cache.save_metadata(
    pdf_path,
    len(self.items),
    parser=PARSER,
    config_fingerprint=config_fingerprint
)

            # Log only after all cache files
            # have successfully been written.

            logger.info(
                "Cache saved successfully | "
                f"Items={len(self.items)} | "
                f"Vectors="
                f"{self.vector_store.index.ntotal} | "
                f"Parser={PARSER}"
            )

            print(
                "Cache saved successfully."
            )

        # ==================================================
        # INITIALIZE SERVICES USED FOR QUERYING
        # ==================================================

        # -------------------------
        # Dense embedding service
        # -------------------------
        #
        # Even on a cache hit we need the
        # embedding model because every
        # user query needs a dense embedding.
        # -------------------------

        print(
            "\nLoading embedding model "
            "for query processing..."
        )

        self.embedding_service = (
            EmbeddingService()
        )

        logger.info(
            "Query embedding service initialized"
        )

        # -------------------------
        # Build BM25 keyword index
        # -------------------------

        print(
            "Building keyword search index..."
        )

        self.keyword_search = (
            KeywordSearch(
                self.items
            )
        )

        print(
            "Keyword search index ready."
        )

        logger.info(
            "BM25 keyword index initialized"
        )

        # -------------------------
        # Initialize re-ranker
        # -------------------------

        print(
            "Initializing re-ranker..."
        )

        self.reranker = (
            Reranker()
        )

        print(
            "Re-ranker ready."
        )

        logger.info(
            "Re-ranker initialized"
        )

        # -------------------------
        # Initialize LLM generator
        # -------------------------

        print(
            "Initializing generator..."
        )

        self.generator = (
            Generator()
        )

        logger.info(
            "Generator initialized"
        )

        print(
            "\nRAG pipeline ready."
        )

        logger.info(
            "RAG pipeline ready"
        )

    # ==================================================
    # ASK
    # ==================================================

    def ask(
        self,
        question
    ):

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question must be a non-empty string."
            )

        question = question.strip()

        logger.info(
            "Query received | "
            f"Question={question}"
        )

        try:

            # -------------------------
            # Generate query embedding
            # -------------------------

            query_embedding = (
                self.embedding_service.embed(
                    question
                )
            )

            logger.info(
                "Query embedding generated"
            )

            # -------------------------
            # Semantic Search - FAISS
            # -------------------------

            semantic_results = (
                self.vector_store.search(
                    query_embedding,
                    RETRIEVAL_TOP_K
                )
            )

            logger.info(
                "FAISS retrieval complete | "
                f"Results="
                f"{len(semantic_results)}"
            )

            # -------------------------
            # Keyword Search - BM25
            # -------------------------

            keyword_results = (
                self.keyword_search.search(
                    question,
                    RETRIEVAL_TOP_K
                )
            )

            logger.info(
                "BM25 retrieval complete | "
                f"Results="
                f"{len(keyword_results)}"
            )

            # -------------------------
            # Sparse Search - SPLADE
            # -------------------------

            sparse_results = (
                self.sparse_search.search(
                    question,
                    RETRIEVAL_TOP_K
                )
            )

            logger.info(
                "SPLADE retrieval complete | "
                f"Results="
                f"{len(sparse_results)}"
            )

            # ==================================================
            # CREATE UNIQUE RERANKING CANDIDATE POOL
            # ==================================================

            candidate_pool = []

            seen = set()

            for item in (
                semantic_results
                + keyword_results
                + sparse_results
            ):

                key = (
                    item.get("page"),
                    item.get("text", "")
                )

                if key not in seen:

                    seen.add(
                        key
                    )

                    candidate_pool.append(
                        item
                    )

            logger.info(
                "Candidate pool created | "
                f"Unique candidates="
                f"{len(candidate_pool)}"
            )

            print(
                f"\nReranking "
                f"{len(candidate_pool)} "
                "unique candidates..."
            )

            # ==================================================
            # RE-RANK CANDIDATES
            # ==================================================

            reranked_results = (
                self.reranker.rerank(
                    question,
                    candidate_pool,
                    RERANK_TOP_K
                )
            )

            logger.info(
                "Reranking complete | "
                f"Candidates="
                f"{len(candidate_pool)} | "
                f"TopK="
                f"{len(reranked_results)}"
            )

            # ==================================================
            # DISPLAY RE-RANKED RESULTS
            # ==================================================

            print(
                "\n--- Re-ranked Results ---"
            )

            for i, item in enumerate(
                reranked_results,
                start=1
            ):

                print(
                    f"\n{i}. Page {item['page']} "
                    f"| Score: "
                    f"{item['rerank_score']:.4f}"
                )

                print(
                    item["text"][:150]
                )

            # ==================================================
            # GENERATE FINAL ANSWER
            # ==================================================

            print(
                f"\nGenerating final answer using "
                f"{len(reranked_results)} "
                "re-ranked results..."
            )

            logger.info(
                "Generating final answer | "
                f"Context chunks="
                f"{len(reranked_results)}"
            )

            try:

                final_answer = (
                    self.generator.generate(
                        question,
                        reranked_results
                    )
                )

            except Exception:

                logger.exception(
                    "Final answer generation failed | "
                    f"Question={question}"
                )

                raise

            logger.info(
                "Final answer generated successfully"
            )

            # ==================================================
            # DISPLAY FINAL ANSWER
            # ==================================================

            print(
                "\n"
            )

            print(
                "=" * 60
            )

            print(
                "FINAL RAG ANSWER"
            )

            print(
                "=" * 60
            )

            print(
                final_answer
            )

            logger.info(
                "Query completed successfully"
            )

            return {
    "answer": final_answer,
    "candidate_count": len(
        candidate_pool
    ),
    "reranked_results": reranked_results,
    "faiss_results": semantic_results,
    "bm25_results": keyword_results,
    "splade_results": sparse_results
}

        except Exception:

            logger.exception(
                "Query processing failed | "
                f"Question={question}"
            )

            raise

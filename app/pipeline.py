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
        # CREATE REQUIRED DIRECTORIES
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

        config_fingerprint = create_config_fingerprint(
            cache_config
        )
        # INITIALIZE CACHE
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
        # CHECK CACHE
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
            # Load processed items
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
            # Load FAISS index
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
            # Load SPLADE embeddings
            sparse_embeddings = (
                self.cache.load_sparse_embeddings(
                    pdf_path
                )
            )
            # Initialize sparse search
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
        # BUILD EVERYTHING
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
            # Initialize text splitter
            splitter = (
                get_text_splitter()
            )
            # Process PDF
            print(
                "Loading PDF..."
            )

            print(
                f"Using parser: {PARSER}"
            )

            document_output_dir = get_document_output_dir(
                pdf_path,
                DATA_DIR
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
            # Initialize embedding service
            print(
                "Loading embedding model..."
            )

            self.embedding_service = (
                EmbeddingService()
            )
            # Generate dense embeddings
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
            # Build FAISS index
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
            # Build SPLADE sparse index
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
            # SAVE CACHE
            print(
                "\nSaving cache..."
            )
            # Save processed items
            self.cache.save_items(
                pdf_path,
                self.items
            )
            # Save FAISS index
            self.vector_store.save(
                cache_path / "faiss.index"
            )
            # Save SPLADE embeddings
            self.sparse_search.save(
                cache_path
                / "sparse_embeddings.pkl"
            )
            # Save metadata
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
        # INITIALIZE SERVICES USED FOR QUERYING
        # Dense embedding service
        #
        # Even on a cache hit we need the
        # embedding model because every
        # user query needs a dense embedding.
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
        # Build BM25 keyword index
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
        # Initialize re-ranker
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
        # Initialize LLM generator
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
    def _is_table_query(self, question):
        table_terms = {
            "material schedule",
            "material list",
            "schedule of quantities",
            "schedule of quantities & prices",
            "bill of quantities",
            "boq",
        }

        table_intent_terms = {
            "list",
            "items",
            "materials",
            "quantities",
            "quantity",
            "show",
            "display",
            "enumerate",
            "what are",
            "which items",
            "give me",
            "provide",
        }

        question_normalized = question.lower()

        has_table_term = any(
            term in question_normalized
            for term in table_terms
        )

        has_table_intent = any(
            term in question_normalized
            for term in table_intent_terms
        )

        return (
            has_table_term
            and has_table_intent
        )

    def _reconstruct_tables(self, results):
        table_candidates = {}

        for rank, item in enumerate(results):
            if item.get("type") != "table":
                continue

            table_index = item.get("table_index")

            if table_index is None:
                continue

            info = table_candidates.setdefault(
                table_index,
                {
                    "row_indices": set(),
                    "first_rank": rank,
                },
            )

            row_index = item.get("row_index")

            if row_index is not None:
                info["row_indices"].add(row_index)

        if not table_candidates:
            return []

        best_table_index = max(
            table_candidates,
            key=lambda index: (
                len(table_candidates[index]["row_indices"]),
                -table_candidates[index]["first_rank"],
            ),
        )

        table_rows = [
            candidate
            for candidate in self.items
            if (
                candidate.get("type") == "table"
                and candidate.get("table_index") == best_table_index
            )
        ]

        table_rows.sort(
            key=lambda candidate: candidate.get("row_index", 0)
        )

        return table_rows

        # --------------------------------------------------------
    # QUERY ASPECT DECOMPOSITION
    # --------------------------------------------------------

    def _build_aspect_queries(
        self,
        question
    ):
        """
        Detect independent information requirements in a question.

        Returns:
            list[str]

        If multiple aspects are detected, each aspect gets its
        own focused retrieval query.

        If no known aspect is detected, the original question
        is returned as a single query.
        """

        q = question.strip()
        lower_q = q.lower()

        aspect_queries = []

        # ----------------------------------------------------
        # EMD / BID SECURITY
        # ----------------------------------------------------

        has_emd = (
            "emd" in lower_q
            or "earnest money" in lower_q
            or "bid security" in lower_q
        )

        if has_emd:

            # EMD VALIDITY
            if (
                "validity" in lower_q
                or "valid for" in lower_q
                or "valid period" in lower_q
                or "validity period" in lower_q
                or "from which date" in lower_q
                or "how long" in lower_q
            ):

                aspect_queries.append(
                    "What is the validity period of the "
                    "e-Bank Guarantee submitted as Earnest Money, "
                    "and from which date is that validity calculated?"
                )

            # EMD AMOUNT
            if (
                "amount" in lower_q
                or "cost" in lower_q
                or "value" in lower_q
                or "how much" in lower_q
            ):

                aspect_queries.append(
                    "What is the exact EMD or bid security "
                    "amount required in the tender?"
                )

            # EMD ACCEPTED FORMS
            if (
                "form" in lower_q
                or "forms" in lower_q
                or "accepted" in lower_q
                or "acceptable" in lower_q
                or "instrument" in lower_q
                or "instruments" in lower_q
            ):

                aspect_queries.append(
                    "What forms or instruments of EMD or bid security "
                    "are accepted in the tender, including Demand Draft, "
                    "Fixed Deposit Receipt, Bankers Cheque, electronic transfer, "
                    "Bank Guarantee/e-BG, and Insurance Surety Bond?"
                )

        # ----------------------------------------------------
        # BID VALIDITY
        # ----------------------------------------------------

        if (
            "bid validity" in lower_q
            or "validity of bid" in lower_q
        ):

            aspect_queries.append(
                "What is the bid validity period, "
                "or for how many days does the bid remain valid?"
            )

        # ----------------------------------------------------
        # TURNOVER
        # ----------------------------------------------------

        if (
            "turnover" in lower_q
            or "annual turnover" in lower_q
        ):

            aspect_queries.append(
                "What is the minimum annual turnover requirement "
                "and which financial years are considered?"
            )

        # ----------------------------------------------------
        # ELIGIBILITY / QUALIFICATION
        # ----------------------------------------------------

        if (
            "eligibility" in lower_q
            or "eligible" in lower_q
            or "qualification" in lower_q
            or "qualifying" in lower_q
        ):

            aspect_queries.append(
                "What are the bidder eligibility and "
                "qualification requirements?"
            )

        # ----------------------------------------------------
        # COMPLETION PERIOD
        # ----------------------------------------------------

        if (
            "completion period" in lower_q
            or "completion time" in lower_q
            or "period of completion" in lower_q
        ):

            aspect_queries.append(
                "What is the required completion period "
                "or completion time?"
            )

        # ----------------------------------------------------
        # DEDUPLICATE
        # ----------------------------------------------------

        deduplicated = []

        seen = set()

        for aspect in aspect_queries:

            normalized = (
                aspect.strip().lower()
            )

            if normalized not in seen:

                seen.add(
                    normalized
                )

                deduplicated.append(
                    aspect
                )

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if not deduplicated:

            return [q]

        return deduplicated


        # --------------------------------------------------------
    # ASK
    # --------------------------------------------------------

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

            # ------------------------------------------------
            # QUERY ASPECTS
            # ------------------------------------------------

            aspect_queries = (
                self._build_aspect_queries(
                    question
                )
            )

            multi_aspect = (
                len(aspect_queries) > 1
            )

            logger.info(
                "Query aspect analysis complete | "
                f"Aspects={len(aspect_queries)} | "
                f"MultiAspect={multi_aspect}"
            )

            # =================================================
            # PATH 1
            # SIMPLE / SINGLE-ASPECT QUERY
            # =================================================
            #
            # Keep the existing fast retrieval path.
            # This avoids additional embedding, retrieval and
            # reranking work for normal questions.
            #
            # =================================================

            if not multi_aspect:

                logger.info(
                    "Using standard single-query retrieval path"
                )

                # ---------------------------------------------
                # QUERY EMBEDDING
                # ---------------------------------------------

                query_embedding = (
                    self.embedding_service.embed(
                        question
                    )
                )

                logger.info(
                    "Query embedding generated"
                )

                # ---------------------------------------------
                # FAISS
                # ---------------------------------------------

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

                # ---------------------------------------------
                # BM25
                # ---------------------------------------------

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

                # ---------------------------------------------
                # SPLADE
                # ---------------------------------------------

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

                # ---------------------------------------------
                # UNIQUE CANDIDATE POOL
                # ---------------------------------------------

                candidate_pool = []

                seen = set()

                for item in (
                    semantic_results
                    + keyword_results
                    + sparse_results
                ):

                    key = (
                        item.get("page"),
                        item.get(
                            "text",
                            ""
                        )
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

                # ---------------------------------------------
                # RERANK
                # ---------------------------------------------

                reranked_results = (
                    self.reranker.rerank(
                        question,
                        candidate_pool,
                        RERANK_TOP_K
                    )
                )

                # ---------------------------------------------
                # TABLE QUERY
                # ---------------------------------------------

                if self._is_table_query(
                    question
                ):

                    table_results = (
                        self._reconstruct_tables(
                            candidate_pool
                        )
                    )

                    if table_results:

                        reranked_results = (
                            self.reranker.rerank(
                                question,
                                table_results,
                                len(table_results)
                            )
                        )

                        logger.info(
                            "Table query detected | "
                            f"ReconstructedRows="
                            f"{len(table_results)}"
                        )

                logger.info(
                    "Reranking complete | "
                    f"Candidates="
                    f"{len(candidate_pool)} | "
                    f"TopK="
                    f"{len(reranked_results)}"
                )

            # =================================================
            # PATH 2
            # MULTI-ASPECT QUERY
            # =================================================
            #
            # Each aspect gets its own:
            #   embedding
            #   FAISS retrieval
            #   BM25 retrieval
            #   SPLADE retrieval
            #   candidate fusion
            #   reranking
            #
            # This prevents one aspect from crowding another
            # out of the final evidence set.
            #
            # =================================================

            else:

                logger.info(
                    "Using aspect-aware retrieval path | "
                    f"Aspects={len(aspect_queries)}"
                )

                combined_results = []

                seen_result_keys = set()

                # Keep raw retrieval results for diagnostics
                # and return compatibility.
                all_faiss_results = []
                all_bm25_results = []
                all_splade_results = []

                for aspect_query in aspect_queries:

                    logger.info(
                        "Processing query aspect | "
                        f"Aspect={aspect_query}"
                    )

                    # -----------------------------------------
                    # ASPECT QUERY EMBEDDING
                    # -----------------------------------------

                    aspect_embedding = (
                        self.embedding_service.embed(
                            aspect_query
                        )
                    )

                    # -----------------------------------------
                    # ASPECT FAISS
                    # -----------------------------------------

                    aspect_semantic_results = (
                        self.vector_store.search(
                            aspect_embedding,
                            RETRIEVAL_TOP_K
                        )
                    )

                    all_faiss_results.extend(
                        aspect_semantic_results
                    )

                    # -----------------------------------------
                    # ASPECT BM25
                    # -----------------------------------------

                    aspect_keyword_results = (
                        self.keyword_search.search(
                            aspect_query,
                            RETRIEVAL_TOP_K
                        )
                    )

                    all_bm25_results.extend(
                        aspect_keyword_results
                    )

                    # -----------------------------------------
                    # ASPECT SPLADE
                    # -----------------------------------------

                    aspect_sparse_results = (
                        self.sparse_search.search(
                            aspect_query,
                            RETRIEVAL_TOP_K
                        )
                    )

                    all_splade_results.extend(
                        aspect_sparse_results
                    )

                    # -----------------------------------------
                    # ASPECT CANDIDATE POOL
                    # -----------------------------------------

                    aspect_candidate_pool = []

                    aspect_seen = set()

                    for item in (
                        aspect_semantic_results
                        + aspect_keyword_results
                        + aspect_sparse_results
                    ):

                        key = (
                            item.get("page"),
                            item.get(
                                "text",
                                ""
                            )
                        )

                        if key not in aspect_seen:

                            aspect_seen.add(
                                key
                            )

                            aspect_candidate_pool.append(
                                item
                            )

                    logger.info(
                        "Aspect candidate pool created | "
                        f"Aspect={aspect_query} | "
                        f"Candidates="
                        f"{len(aspect_candidate_pool)}"
                    )

                    if not aspect_candidate_pool:
                        continue

                    # -----------------------------------------
                    # ASPECT RERANK
                    # -----------------------------------------

                    aspect_reranked = (
                        self.reranker.rerank(
                            aspect_query,
                            aspect_candidate_pool,
                            RERANK_TOP_K
                        )
                    )

                    logger.info(
                        "Aspect reranking complete | "
                        f"Aspect={aspect_query} | "
                        f"TopK="
                        f"{len(aspect_reranked)}"
                    )

                    # -----------------------------------------
                    # TABLE HANDLING
                    # -----------------------------------------

                    if self._is_table_query(
                        aspect_query
                    ):

                        table_results = (
                            self._reconstruct_tables(
                                aspect_candidate_pool
                            )
                        )

                        if table_results:

                            aspect_reranked = (
                                self.reranker.rerank(
                                    aspect_query,
                                    table_results,
                                    len(table_results)
                                )
                            )

                            logger.info(
                                "Aspect table query detected | "
                                f"Aspect={aspect_query} | "
                                f"Rows="
                                f"{len(table_results)}"
                            )

                    # -----------------------------------------
                    # KEEP BEST EVIDENCE FOR THIS ASPECT
                    # -----------------------------------------
                    #
                    # Keep up to 2 chunks per aspect.
                    # This provides enough evidence for the
                    # generator without flooding the context.
                    #
                    # -----------------------------------------

                    # -----------------------------------------
                    # KEEP BEST EVIDENCE FOR THIS ASPECT
                    # -----------------------------------------
                    #
                    # Keep up to 2 chunks per aspect.
                    #
                    # IMPORTANT:
                    # The same chunk may legitimately support
                    # different aspects. Therefore the aspect
                    # is part of the deduplication key.
                    # -----------------------------------------

                    aspect_selected = 0

                    for item in aspect_reranked:

                        if aspect_selected >= 2:
                            break

                        aspect_key = (
                            item.get("page"),
                            item.get(
                                "text",
                                ""
                            ),
                            aspect_query,
                        )

                        if aspect_key in seen_result_keys:
                            continue

                        seen_result_keys.add(
                            aspect_key
                        )

                        item_copy = dict(
                            item
                        )

                        item_copy[
                            "aspect_query"
                        ] = aspect_query

                        combined_results.append(
                            item_copy
                        )

                        aspect_selected += 1

                        # ---------------------------------------------
                        # FINAL MULTI-ASPECT EVIDENCE
                        # ---------------------------------------------

                        reranked_results = (
                            combined_results
                        )

                # ---------------------------------------------
                # FALLBACK
                # ---------------------------------------------
                #
                # If aspect retrieval produced nothing,
                # fall back to the original question.
                #
                # ---------------------------------------------

                if not reranked_results:

                    logger.warning(
                        "Aspect-aware retrieval returned "
                        "no evidence; using fallback retrieval"
                    )

                    query_embedding = (
                        self.embedding_service.embed(
                            question
                        )
                    )

                    semantic_results = (
                        self.vector_store.search(
                            query_embedding,
                            RETRIEVAL_TOP_K
                        )
                    )

                    keyword_results = (
                        self.keyword_search.search(
                            question,
                            RETRIEVAL_TOP_K
                        )
                    )

                    sparse_results = (
                        self.sparse_search.search(
                            question,
                            RETRIEVAL_TOP_K
                        )
                    )

                    fallback_pool = []

                    fallback_seen = set()

                    for item in (
                        semantic_results
                        + keyword_results
                        + sparse_results
                    ):

                        key = (
                            item.get("page"),
                            item.get(
                                "text",
                                ""
                            )
                        )

                        if key not in fallback_seen:

                            fallback_seen.add(
                                key
                            )

                            fallback_pool.append(
                                item
                            )

                    reranked_results = (
                        self.reranker.rerank(
                            question,
                            fallback_pool,
                            RERANK_TOP_K
                        )
                    )

                    logger.info(
                        "Fallback retrieval complete | "
                        f"Candidates="
                        f"{len(fallback_pool)} | "
                        f"TopK="
                        f"{len(reranked_results)}"
                    )

                # ---------------------------------------------
                # RETURN RAW RESULTS FOR COMPATIBILITY
                # ---------------------------------------------

                semantic_results = (
                    all_faiss_results
                )

                keyword_results = (
                    all_bm25_results
                )

                sparse_results = (
                    all_splade_results
                )

                candidate_pool = (
                    list(
                        {
                            (
                                item.get("page"),
                                item.get(
                                    "text",
                                    ""
                                )
                            ): item
                            for item in (
                                semantic_results
                                + keyword_results
                                + sparse_results
                            )
                        }.values()
                    )
                )

                logger.info(
                    "Multi-aspect evidence assembled | "
                    f"Aspects="
                    f"{len(aspect_queries)} | "
                    f"Evidence="
                    f"{len(reranked_results)}"
                )

            # =================================================
            # GENERATE FINAL ANSWER
            # =================================================

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
                "splade_results": sparse_results,
            }

        except Exception:

            logger.exception(
                "Query processing failed | "
                f"Question={question}"
            )

            raise
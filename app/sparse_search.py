from sentence_transformers import SparseEncoder

from .config import (
    SPARSE_MODEL,
    SPARSE_DOCUMENT_MAX_ACTIVE_DIMS,
    SPARSE_QUERY_MAX_ACTIVE_DIMS
)


class SparseSearch:

    def __init__(
        self,
        items,
        cached_embeddings=None
    ):

        print(
            "Loading sparse retrieval model..."
        )

        # -----------------------------------------
        # Load sparse embedding model
        # -----------------------------------------

        self.model = SparseEncoder(
            SPARSE_MODEL
        )

        # -----------------------------------------
        # Keep only searchable text/table items
        # -----------------------------------------

        self.items = [
            item
            for item in items
            if item["type"] in [
                "text",
                "table"
            ]
            and item.get("text")
        ]

        if not self.items:

            raise ValueError(
                "No text or table items available "
                "for sparse retrieval."
            )

        print(
            f"Sparse search using "
            f"{len(self.items)} documents."
        )

        # -----------------------------------------
        # Load cached document embeddings
        # -----------------------------------------

        if cached_embeddings is not None:

            print(
                "Loading cached sparse "
                "document embeddings..."
            )

            self.document_embeddings = (
                cached_embeddings
            )

            print(
                "Cached sparse document "
                "embeddings loaded."
            )

        # -----------------------------------------
        # Generate document embeddings
        # -----------------------------------------

        else:

            print(
                "Generating sparse document "
                "embeddings..."
            )

            documents = [
                item["text"]
                for item in self.items
            ]

            # -------------------------------------
            # Encode in batches to reduce RAM usage
            # -------------------------------------

            batch_size = 8

            batches = []

            total_documents = len(
                documents
            )

            for start in range(
                0,
                total_documents,
                batch_size
            ):

                end = min(
                    start + batch_size,
                    total_documents
                )

                batch = documents[
                    start:end
                ]

                print(
                    f"Generating sparse embeddings "
                    f"for documents "
                    f"{start + 1}-{end} "
                    f"of {total_documents}..."
                )

                batch_embeddings = (
                    self.model.encode_document(
                        batch,
                        max_active_dims=(
                            SPARSE_DOCUMENT_MAX_ACTIVE_DIMS
                        )
                    )
                )

                batches.append(
                    batch_embeddings
                )

            # -------------------------------------
            # Combine sparse embedding batches
            # -------------------------------------

            self.document_embeddings = (
                self._combine_embeddings(
                    batches
                )
            )

            print(
                "Sparse search document "
                "embeddings ready."
            )

    # -----------------------------------------
    # Combine sparse embedding batches
    # -----------------------------------------

    @staticmethod
    def _combine_embeddings(
    batches
):

        if not batches:

            raise ValueError(
                "No sparse embedding batches "
                "were generated."
            )

        if len(batches) == 1:

            return batches[0]

        import torch

        # -----------------------------------------
        # Combine PyTorch sparse COO tensors
        # without converting them to dense.
        # -----------------------------------------

        if all(
            isinstance(batch, torch.Tensor)
            and batch.layout == torch.sparse_coo
            for batch in batches
        ):

            combined = torch.cat(
                batches,
                dim=0
            )

            return combined.coalesce()

        # -----------------------------------------
        # Fallback for dense tensors / other types
        # -----------------------------------------

        try:

            return torch.cat(
                batches,
                dim=0
            )

        except Exception as exc:

            raise TypeError(
                "Unsupported sparse embedding "
                "batch type."
            ) from exc

    # -----------------------------------------
    # Save document embeddings
    # -----------------------------------------

    def save(
        self,
        path
    ):

        import pickle

        print(
            "Saving sparse document embeddings..."
        )

        with open(
            path,
            "wb"
        ) as f:

            pickle.dump(
                self.document_embeddings,
                f
            )

        print(
            "Sparse embeddings saved."
        )

    # -----------------------------------------
    # Load document embeddings
    # -----------------------------------------

    @staticmethod
    def load_embeddings(
        path
    ):

        import pickle

        print(
            "Loading sparse embeddings "
            "from cache..."
        )

        with open(
            path,
            "rb"
        ) as f:

            embeddings = pickle.load(
                f
            )

        print(
            "Sparse embeddings loaded."
        )

        return embeddings

    # -----------------------------------------
    # Search
    # -----------------------------------------

    def search(
        self,
        query,
        k=5
    ):

        query_embedding = (
            self.model.encode_query(
                [query],
                max_active_dims=(
                    SPARSE_QUERY_MAX_ACTIVE_DIMS
                )
            )
        )

        scores = self.model.similarity(
            query_embedding,
            self.document_embeddings
        )[0]

        k = min(
            k,
            len(self.items)
        )

        if k == 0:
            return []

        # SPLADE similarity returns a tensor.
        top_indices = (
            scores.argsort(
                descending=True
            )[:k]
        )

        results = []

        for index in top_indices:

            index = index.item()

            item = dict(
                self.items[index]
            )

            item["sparse_score"] = (
                scores[index].item()
            )

            results.append(
                item
            )

        return results
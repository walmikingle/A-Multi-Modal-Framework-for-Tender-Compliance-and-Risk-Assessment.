import pickle

import torch
from sentence_transformers import SparseEncoder

from .config import (
    SPARSE_DOCUMENT_MAX_ACTIVE_DIMS,
    SPARSE_MODEL,
    SPARSE_QUERY_MAX_ACTIVE_DIMS,
)


class SparseSearch:

    def __init__(
        self,
        items,
        cached_embeddings=None,
    ):
        print("Loading sparse retrieval model...")

        self.model = SparseEncoder(
            SPARSE_MODEL
        )

        self.items = [
            item
            for item in items
            if item["type"] in {"text", "table"}
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

        else:
            print(
                "Generating sparse document "
                "embeddings..."
            )

            documents = [
                item["text"]
                for item in self.items
            ]

            batch_size = 8
            batches = []
            total_documents = len(documents)

            for start in range(
                0,
                total_documents,
                batch_size,
            ):
                end = min(
                    start + batch_size,
                    total_documents,
                )

                batch = documents[start:end]

                print(
                    f"Generating sparse embeddings "
                    f"for documents {start + 1}-{end} "
                    f"of {total_documents}..."
                )

                batch_embeddings = (
                    self.model.encode_document(
                        batch,
                        max_active_dims=(
                            SPARSE_DOCUMENT_MAX_ACTIVE_DIMS
                        ),
                    )
                )

                batches.append(
                    batch_embeddings
                )

            self.document_embeddings = (
                self._combine_embeddings(
                    batches
                )
            )

            print(
                "Sparse search document "
                "embeddings ready."
            )

    @staticmethod
    def _combine_embeddings(batches):

        if not batches:
            raise ValueError(
                "No sparse embedding batches "
                "were generated."
            )

        if len(batches) == 1:
            return batches[0]

        if all(
            isinstance(batch, torch.Tensor)
            and batch.layout == torch.sparse_coo
            for batch in batches
        ):
            combined = torch.cat(
                batches,
                dim=0,
            )

            return combined.coalesce()

        try:
            return torch.cat(
                batches,
                dim=0,
            )

        except Exception as exc:
            raise TypeError(
                "Unsupported sparse embedding "
                "batch type."
            ) from exc

    def save(self, path):

        print(
            "Saving sparse document embeddings..."
        )

        with open(
            path,
            "wb",
        ) as file:
            pickle.dump(
                self.document_embeddings,
                file,
            )

        print(
            "Sparse embeddings saved."
        )

    @staticmethod
    def load_embeddings(path):

        print(
            "Loading sparse embeddings "
            "from cache..."
        )

        with open(
            path,
            "rb",
        ) as file:
            embeddings = pickle.load(
                file
            )

        print(
            "Sparse embeddings loaded."
        )

        return embeddings

    def search(
        self,
        query,
        k=5,
    ):
        query_embedding = (
            self.model.encode_query(
                [query],
                max_active_dims=(
                    SPARSE_QUERY_MAX_ACTIVE_DIMS
                ),
            )
        )

        scores = self.model.similarity(
            query_embedding,
            self.document_embeddings,
        )[0]

        k = min(
            k,
            len(self.items),
        )

        if k == 0:
            return []

        top_indices = scores.argsort(
            descending=True
        )[:k]

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

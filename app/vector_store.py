import faiss
import numpy as np

from .config import EMBEDDING_DIMENSION


class VectorStore:

    def __init__(self):

        self.index = None

        self.items = []

    def build(self, items):

        self.items = [
            item
            for item in items
            if item.get("embedding") is not None
        ]

        embeddings = np.array(
            [
                item["embedding"]
                for item in self.items
            ],
            dtype=np.float32
        )

        if len(embeddings) == 0:
            self.index = faiss.IndexFlatL2(
                EMBEDDING_DIMENSION
            )
            return

        embedding_dimension = embeddings.shape[1]

        if embedding_dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding dimension does not match "
                "embedding_dimension in config.yaml: "
                f"expected {EMBEDDING_DIMENSION}, "
                f"got {embedding_dimension}."
            )

        self.index = faiss.IndexFlatL2(
            embedding_dimension
        )

        self.index.add(embeddings)

    def save(self, path):

        if self.index is None:
            raise RuntimeError(
                "Cannot save a vector store before building it."
            )

        faiss.write_index(
            self.index,
            str(path)
        )

    def load(self, path, items):

        self.index = faiss.read_index(
            str(path)
        )

        self.items = [
            item
            for item in items
            if item.get("embedding") is not None
        ]

    def search(self, query_embedding, k=5):

        if self.index is None or k <= 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim != 1:
            raise ValueError(
                "Query embedding must be a one-dimensional vector."
            )

        if query_embedding.shape[0] != self.index.d:
            raise ValueError(
                "Query embedding dimension does not match "
                "the FAISS index."
            )

        k = min(
            k,
            self.index.ntotal
        )

        if k == 0:
            return []

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )

        results = []

        for idx in indices[0]:

            if idx != -1:

                item = dict(
                    self.items[idx]
                )

                item.pop(
                    "embedding",
                    None
                )

                results.append(item)

        return results

import faiss
import numpy as np

from .config import EMBEDDING_DIMENSION


class VectorStore:

    def __init__(self):

        self.index = faiss.IndexFlatL2(
            EMBEDDING_DIMENSION
        )

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

        if len(embeddings) > 0:

            self.index.add(embeddings)

    def search(self, query_embedding, k=5):

        k = min(
            k,
            self.index.ntotal
        )

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
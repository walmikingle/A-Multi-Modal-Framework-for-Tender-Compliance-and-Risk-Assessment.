from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


class EmbeddingService:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def embed(self, text):

        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

    def embed_items(self, items):

        for item in items:

            if item["type"] in {"text", "table"}:

                item["embedding"] = self.embed(
                    item["text"]
                )

            else:

                item["embedding"] = None

        return items
from rank_bm25 import BM25Okapi


class KeywordSearch:

    def __init__(self, items):

        self.items = [
            item
            for item in items
            if item["type"] in ["text", "table"]
            and item.get("text")
        ]

        self.tokenized_documents = [
            self.tokenize(item["text"])
            for item in self.items
        ]

        self.bm25 = (
            BM25Okapi(self.tokenized_documents)
            if self.tokenized_documents
            else None
        )

    @staticmethod
    def tokenize(text):

        return text.lower().split()

    def search(self, query, k=5):

        if self.bm25 is None or k <= 0:
            return []

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = []

        for index in ranked_indices:

            item = dict(
                self.items[index]
            )

            item["keyword_score"] = float(
                scores[index]
            )

            results.append(item)

        return results


import torch
from sentence_transformers import CrossEncoder
import re
from .config import RERANKER_MODEL


class Reranker:

    def __init__(self):
        print("Loading re-ranker model...")

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = CrossEncoder(
            RERANKER_MODEL,
            device=device,
            max_length=512,
        )

        print(
            f"Re-ranker ready. Device: {device}"
        )

    @staticmethod
    def _normalize(text):
        return re.sub(
            r"\s+",
            " ",
            text.lower(),
        ).strip()

    @staticmethod
    def _query_terms(query):
        words = re.findall(
            r"\b[a-zA-Z0-9€$%]+\b",
            query.lower(),
        )

        stopwords = {
            "what",
            "which",
            "who",
            "where",
            "when",
            "why",
            "how",
            "was",
            "were",
            "did",
            "does",
            "do",
            "the",
            "a",
            "an",
            "to",
            "of",
            "and",
            "or",
            "is",
            "are",
            "for",
            "from",
            "in",
            "on",
            "according",
            "this",
            "that",
        }

        return {
            word
            for word in words
            if word not in stopwords
        }

    def _evidence_score(
        self,
        query,
        text,
    ):
        query_normalized = self._normalize(
            query
        )

        text_normalized = self._normalize(
            text
        )

        query_terms = self._query_terms(
            query
        )

        if not query_terms:
            return 0.0

        matched_terms = sum(
            1
            for term in query_terms
            if term in text_normalized
        )

        term_coverage = (
            matched_terms
            / len(query_terms)
        )

        score = term_coverage * 2.0

        specific_terms = [
            term
            for term in query_terms
            if len(term) >= 6
            or term[0].isdigit()
        ]

        for term in specific_terms:
            if term in text_normalized:
                score += 0.5

        query_numbers = re.findall(
            r"\d+(?:[.,]\d+)*",
            query_normalized,
        )

        for number in query_numbers:
            if number in text_normalized:
                score += 1.0

        return score

    def rerank(
        self,
        query,
        items,
        top_k=5,
    ):
        if not items:
            return []

        valid_items = [
            item
            for item in items
            if item.get("text")
        ]

        if not valid_items:
            return []

        pairs = [
            [
                query,
                item["text"],
            ]
            for item in valid_items
        ]

        model_scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        scored_items = []

        for item, model_score in zip(
            valid_items,
            model_scores,
        ):
            evidence_score = (
                self._evidence_score(
                    query,
                    item["text"],
                )
            )

            final_score = (
                float(model_score)
                + evidence_score
            )

            result = dict(item)

            result["rerank_model_score"] = (
                float(model_score)
            )

            result["evidence_score"] = (
                float(evidence_score)
            )

            result["rerank_score"] = final_score

            scored_items.append(result)

        scored_items.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return scored_items[:top_k]


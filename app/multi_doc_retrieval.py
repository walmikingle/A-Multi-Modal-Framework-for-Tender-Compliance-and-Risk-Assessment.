"""Robust multi-document retrieval for comparative and exhaustive questions.

This module is responsible for fixing the retrieval weaknesses
observed when the same question must be answered across many
tender PDFs:

* a global top-k rerank that crowds most documents out,
* no detection of comparison / aggregation / exhaustive / max-min
  intents,
* no synonym expansion for procurement terminology
  (EMD, Quotation, Validity, Turnover),
* no deterministic numeric normalization or comparison.

It does NOT replace the existing single-document pipeline. The
multi-document pipeline routes questions here only when a
question is multi-document in scope AND has a comparative,
aggregating, exhaustive, or max/min structure.

Public entry point:
    MultiDocRetrieval.retrieve(...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import (
    MULTI_DOC_ENABLE_EXHAUSTIVE,
    MULTI_DOC_FINAL_EVIDENCE_PER_DOCUMENT,
    MULTI_DOC_RERANK_K_PER_DOCUMENT,
    MULTI_DOC_RETRIEVAL_K_PER_DOCUMENT,
)
from .logger import logger


# ============================================================
# TARGET FIELD SYNONYM LAYER
# ============================================================
#
# Each target field has multiple surface forms used across
# Indian government and PSU tender documents. The retrieval
# formulations below are derived from these synonyms.

FIELD_SYNONYMS: dict[str, list[str]] = {
    "EMD": [
        "EMD",
        "Earnest Money Deposit",
        "Earnest Money",
        "Bid Security",
        "Bid Security Amount",
        "Earnest Money Deposit amount",
        "bid security amount",
        "earnest money",
    ],
    "QUOTED_AMOUNT": [
        "quoted amount",
        "quotation",
        "quoted price",
        "quoted rate",
        "financial bid",
        "financial offer",
        "price quoted",
        "price schedule",
        "schedule of prices",
        "total bid value",
        "total quoted amount",
        "bid value",
        "bid price",
        "offer price",
        "price bid",
        "lump sum price",
        "contract price",
    ],
    "BID_VALIDITY": [
        "bid validity",
        "validity of bid",
        "offer validity",
        "bid offer validity",
        "validity period",
        "bid validity period",
        "tender validity",
    ],
    "TURNOVER": [
        "annual turnover",
        "minimum turnover",
        "average annual turnover",
        "turnover requirement",
        "financial turnover",
        "turnover",
    ],
    "COMPLETION_PERIOD": [
        "completion period",
        "period of completion",
        "completion time",
        "delivery period",
        "execution period",
    ],
}

# ============================================================
# INTENT DETECTION
# ============================================================

INTENT_FACT = "fact"
INTENT_COMPARISON_MAX = "comparison_max"
INTENT_COMPARISON_MIN = "comparison_min"
INTENT_AGGREGATION = "aggregation"
INTENT_EXHAUSTIVE = "exhaustive"

DOC_SCOPE_ALL = "all"
DOC_SCOPE_NAMED = "named"


@dataclass
class QueryPlan:
    """A structured description of what the question is asking."""

    raw_question: str
    intent: str = INTENT_FACT
    target_field: str | None = None
    scope: str = DOC_SCOPE_ALL
    formulations: list[str] = field(default_factory=list)
    is_multi_document: bool = False
    uses_exhaustive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_question": self.raw_question,
            "intent": self.intent,
            "target_field": self.target_field,
            "scope": self.scope,
            "formulations": list(self.formulations),
            "is_multi_document": self.is_multi_document,
            "uses_exhaustive": self.uses_exhaustive,
        }


# ============================================================
# TARGET FIELD DETECTION
# ============================================================

_FIELD_DETECTION_ORDER: list[tuple[str, list[str]]] = [
    ("EMD", ["emd", "earnest money", "bid security"]),
    (
        "BID_VALIDITY",
        ["bid validity", "validity of bid", "validity period", "offer validity"],
    ),
    ("TURNOVER", ["turnover", "annual turnover"]),
    (
        "QUOTED_AMOUNT",
        [
            "quotation",
            "quoted",
            "quoted amount",
            "quoted price",
            "financial bid",
            "price quoted",
            "price schedule",
            "total bid value",
        ],
    ),
    (
        "COMPLETION_PERIOD",
        ["completion period", "completion time", "period of completion"],
    ),
]


def _detect_target_field(question_lower: str) -> str | None:
    """Return the most specific procurement field the question targets."""

    for field_name, terms in _FIELD_DETECTION_ORDER:
        for term in terms:
            if term in question_lower:
                return field_name
    return None


# ============================================================
# INTENT DETECTION
# ============================================================

_MAX_TOKENS = {
    "highest",
    "maximum",
    "max",
    "largest",
    "most",
    "top",
    "biggest",
    "greatest",
}

_MIN_TOKENS = {
    "lowest",
    "minimum",
    "min",
    "smallest",
    "least",
    "cheapest",
    "shortest",
}

_AGGREGATION_TOKENS = {
    "total",
    "sum",
    "average",
    "mean",
    "combined",
    "overall",
}

_EXHAUSTIVE_TOKENS = {
    "all",
    "every",
    "each",
    "list",
    "list of",
    "for each",
    "for every",
    "across all",
    "in each",
}


def _has_any(question_lower: str, tokens: Iterable[str]) -> bool:
    for token in tokens:
        # use word-boundary-style matching for single-word tokens,
        # substring matching for multi-word phrases
        if " " in token or "-" in token:
            if token in question_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(token)}\b", question_lower):
                return True
    return False


def _detect_intent(question_lower: str) -> str:
    if _has_any(question_lower, _MAX_TOKENS):
        return INTENT_COMPARISON_MAX
    if _has_any(question_lower, _MIN_TOKENS):
        return INTENT_COMPARISON_MIN
    if _has_any(question_lower, _AGGREGATION_TOKENS):
        return INTENT_AGGREGATION
    if _has_any(question_lower, _EXHAUSTIVE_TOKENS):
        return INTENT_EXHAUSTIVE
    return INTENT_FACT


# ============================================================
# FORMULATION GENERATION
# ============================================================

_FORMULATION_TEMPLATES: dict[str, list[str]] = {
    "EMD": [
        "What is the EMD or Bid Security amount required in this tender?",
        "Earnest Money Deposit amount",
        "Bid Security amount in rupees",
        "EMD amount and accepted instruments",
        "What is the bid security amount for this tender?",
    ],
    "QUOTED_AMOUNT": [
        "What is the total quoted amount or bid value in this tender?",
        "Quoted amount / price schedule",
        "Financial bid / total bid price",
        "Lump sum price quoted by the bidder",
        "Total bid value / contract price",
        "Price schedule of the tender",
    ],
    "BID_VALIDITY": [
        "What is the bid validity period in days?",
        "Offer validity / bid offer validity",
        "Validity of bid from the last date of bid opening",
        "Bid validity period stated in the tender",
    ],
    "TURNOVER": [
        "What is the minimum annual turnover requirement?",
        "Average annual turnover requirement",
        "Financial turnover requirement",
    ],
    "COMPLETION_PERIOD": [
        "What is the completion period or delivery period?",
        "Period of completion in months",
        "Execution period for the work",
    ],
}


def _build_formulations(target_field: str | None, raw_question: str) -> list[str]:
    formulations: list[str] = [raw_question.strip()]
    if target_field and target_field in _FORMULATION_TEMPLATES:
        formulations.extend(_FORMULATION_TEMPLATES[target_field])
    # Always include a few high-signal synonyms verbatim so BM25
    # and SPLADE get lexical hits even if the LLM-formulated
    # query is too soft.
    if target_field:
        formulations.extend(FIELD_SYNONYMS.get(target_field, []))
    # Deduplicate while preserving order.
    seen = set()
    unique: list[str] = []
    for text in formulations:
        key = text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(text.strip())
    return unique


# ============================================================
# PLAN
# ============================================================

def build_query_plan(
    raw_question: str,
    is_multi_document: bool,
) -> QueryPlan:
    """Inspect the question and return a structured QueryPlan."""

    question_lower = raw_question.lower()

    target_field = _detect_target_field(question_lower)
    intent = _detect_intent(question_lower)

    uses_exhaustive = (
        MULTI_DOC_ENABLE_EXHAUSTIVE
        and is_multi_document
        and intent in {
            INTENT_COMPARISON_MAX,
            INTENT_COMPARISON_MIN,
            INTENT_AGGREGATION,
            INTENT_EXHAUSTIVE,
        }
    )

    formulations = _build_formulations(target_field, raw_question)

    plan = QueryPlan(
        raw_question=raw_question,
        intent=intent,
        target_field=target_field,
        scope=DOC_SCOPE_ALL if is_multi_document else DOC_SCOPE_NAMED,
        formulations=formulations,
        is_multi_document=is_multi_document,
        uses_exhaustive=uses_exhaustive,
    )

    logger.info(
        "QueryPlan built | "
        f"intent={intent} | "
        f"target={target_field} | "
        f"scope={plan.scope} | "
        f"formulations={len(formulations)} | "
        f"exhaustive={uses_exhaustive}"
    )

    return plan


# ============================================================
# NUMERIC NORMALIZATION
# ============================================================

_NUM_WITH_UNIT = re.compile(
    r"(?:Rs\.?|INR|₹)\s*"
    r"([\d]+(?:[.,][\d]+)*)"
    r"\s*(crore|cr|crores| crore)?\s*(lakh|lac|lakhs| lacs)?",
    re.IGNORECASE,
)

_PLAIN_NUM = re.compile(r"\d+(?:[.,]\d+)*")

_UNIT_RUPEE = {
    "crore": 10_000_000,
    "cr": 10_000_000,
    "crores": 10_000_000,
    "crore ": 10_000_000,
    "lakh": 100_000,
    "lac": 100_000,
    "lakhs": 100_000,
    "lacs": 100_000,
}


def _to_float(num_text: str) -> float:
    # Indian number formatting uses comma separators like
    # 2,00,00,000. Remove commas and treat the result as a float.
    cleaned = num_text.replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def normalize_rupee_value(text: str) -> float | None:
    """Return the largest rupee value mentioned in the text, in rupees.

    Handles Rs. 2.0 Crores, ₹2,00,00,000, Rs. 20000000, Rs. 50 Lakhs.
    Returns None if nothing usable is found.
    """

    if not text:
        return None

    best: float | None = None
    for match in _NUM_WITH_UNIT.finditer(text):
        number_text = match.group(1)
        unit_text = (match.group(2) or match.group(3) or "").strip().lower()
        unit_text = unit_text.replace(" ", "")
        value = _to_float(number_text)
        if unit_text in _UNIT_RUPEE:
            value *= _UNIT_RUPEE[unit_text]
        # Sometimes the same number is repeated in words. Keep
        # the largest sane positive value.
        if best is None or value > best:
            best = value
    if best is not None:
        return best
    # Fallback: plain number with no unit (assume rupees).
    plain_matches = _PLAIN_NUM.findall(text)
    candidates: list[float] = []
    for token in plain_matches:
        try:
            v = _to_float(token)
            if v > 0:
                candidates.append(v)
        except ValueError:
            continue
    if not candidates:
        return None
    return max(candidates)


def normalize_day_value(text: str) -> float | None:
    """Return the largest day count mentioned in the text."""

    if not text:
        return None

    # Look for "<n> days" / "<n> day" patterns first; they are
    # unambiguous.
    day_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:days?|day\b)",
        text,
        re.IGNORECASE,
    )
    days: list[float] = []
    for token in day_matches:
        try:
            v = _to_float(token)
            if 1 <= v <= 3650:
                days.append(v)
        except ValueError:
            continue
    if days:
        return max(days)

    # Fallback: "X months" -> convert to days (~30).
    month_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:months?|month\b)",
        text,
        re.IGNORECASE,
    )
    months = [_to_float(t) for t in month_matches if _to_float(t) > 0]
    if months:
        return max(months) * 30.0

    return None


# ============================================================
# FIELD EXTRACTION
# ============================================================

@dataclass
class FieldRecord:
    document: str
    field: str
    value_numeric: float | None
    value_raw: str | None
    page: int | None
    source_text: str


def _extract_for_field(
    field_name: str,
    text: str,
) -> float | None:
    if field_name in {"EMD", "QUOTED_AMOUNT"}:
        return normalize_rupee_value(text)
    if field_name == "BID_VALIDITY":
        return normalize_day_value(text)
    if field_name == "TURNOVER":
        return normalize_rupee_value(text)
    if field_name == "COMPLETION_PERIOD":
        return normalize_day_value(text)
    return None


def extract_field_records(
    field_name: str,
    per_document_evidence: dict[str, list[dict[str, Any]]],
) -> list[FieldRecord]:
    """Extract a structured numeric record per document."""

    records: list[FieldRecord] = []
    for document, evidence in per_document_evidence.items():
        best_record: FieldRecord | None = None
        for item in evidence:
            text = item.get("text", "") or ""
            value = _extract_for_field(field_name, text)
            if value is None:
                continue
            record = FieldRecord(
                document=document,
                field=field_name,
                value_numeric=value,
                value_raw=_raw_phrase(text, value, field_name),
                page=item.get("page"),
                source_text=text,
            )
            if (
                best_record is None
                or (record.value_numeric or 0)
                > (best_record.value_numeric or 0)
            ):
                best_record = record
        if best_record is not None:
            records.append(best_record)
    return records


def _raw_phrase(text: str, value: float, field_name: str) -> str:
    """Return the substring around the detected value, for citation."""

    if not text:
        return ""

    if field_name in {"EMD", "QUOTED_AMOUNT", "TURNOVER"}:
        pattern = _NUM_WITH_UNIT
    else:
        pattern = re.compile(
            r"\d+(?:\.\d+)?\s*(?:days?|months?)",
            re.IGNORECASE,
        )

    match = pattern.search(text)
    if match:
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        return text[start:end].strip()

    return str(value)


# ============================================================
# COMPARISON
# ============================================================

@dataclass
class ComparisonResult:
    intent: str
    field: str
    winner: FieldRecord | None
    all_records: list[FieldRecord]
    missing_documents: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "field": self.field,
            "winner": (
                {
                    "document": self.winner.document,
                    "field": self.winner.field,
                    "value_numeric": self.winner.value_numeric,
                    "value_raw": self.winner.value_raw,
                    "page": self.winner.page,
                }
                if self.winner
                else None
            ),
            "all_records": [
                {
                    "document": r.document,
                    "value_numeric": r.value_numeric,
                    "value_raw": r.value_raw,
                    "page": r.page,
                }
                for r in self.all_records
            ],
            "missing_documents": list(self.missing_documents),
        }


def perform_comparison(
    intent: str,
    field_name: str,
    records: list[FieldRecord],
    all_documents: list[str],
) -> ComparisonResult:
    """Pick the max or min deterministically from the extracted records."""

    documents_with_records = {r.document for r in records}
    missing = [d for d in all_documents if d not in documents_with_records]

    if not records:
        return ComparisonResult(
            intent=intent,
            field=field_name,
            winner=None,
            all_records=[],
            missing_documents=missing,
        )

    if intent == INTENT_COMPARISON_MAX:
        winner = max(records, key=lambda r: r.value_numeric or -1.0)
    elif intent == INTENT_COMPARISON_MIN:
        winner = min(
            [r for r in records if r.value_numeric is not None],
            key=lambda r: r.value_numeric,
            default=None,
        )
    else:
        winner = None

    return ComparisonResult(
        intent=intent,
        field=field_name,
        winner=winner,
        all_records=records,
        missing_documents=missing,
    )


# ============================================================
# ORCHESTRATOR
# ============================================================


class MultiDocRetrieval:
    """Pluggable per-document retrieval orchestrator.

    The MultiDocumentPipeline constructs one of these with
    references to its already-built per-document indexes. The
    `retrieve()` method is the only entry point used by the
    pipeline.
    """

    def __init__(
        self,
        *,
        embedding_service,
        reranker,
        get_document_vector_store,
        get_document_keyword_search,
        search_sparse_scoped,
        document_names: list[str],
    ) -> None:
        self.embedding_service = embedding_service
        self.reranker = reranker
        self._get_document_vector_store = get_document_vector_store
        self._get_document_keyword_search = get_document_keyword_search
        self._search_sparse_scoped = search_sparse_scoped
        self.document_names = list(document_names)

    # --------------------------------------------------------
    # PER-DOCUMENT RETRIEVAL
    # --------------------------------------------------------

    def _retrieve_for_document(
        self,
        document: str,
        formulations: list[str],
    ) -> list[dict[str, Any]]:
        per_formulation_evidence: list[dict[str, Any]] = []

        for formulation in formulations:
            try:
                embedding = self.embedding_service.embed(formulation)
            except Exception:
                logger.exception(
                    "Embedding failed for formulation; "
                    f"document={document}"
                )
                continue

            try:
                store = self._get_document_vector_store(document)
                semantic = store.search(
                    embedding,
                    MULTI_DOC_RETRIEVAL_K_PER_DOCUMENT,
                )
            except Exception:
                logger.exception(
                    "Per-document FAISS failed "
                    f"document={document}"
                )
                semantic = []

            try:
                kw = self._get_document_keyword_search(document)
                keyword = kw.search(
                    formulation,
                    MULTI_DOC_RETRIEVAL_K_PER_DOCUMENT,
                )
            except Exception:
                logger.exception(
                    "Per-document BM25 failed "
                    f"document={document}"
                )
                keyword = []

            try:
                sparse = self._search_sparse_scoped(
                    formulation,
                    [document],
                    MULTI_DOC_RETRIEVAL_K_PER_DOCUMENT,
                )
            except Exception:
                logger.exception(
                    "Per-document SPLADE failed "
                    f"document={document}"
                )
                sparse = []

            candidates = semantic + keyword + sparse
            seen = set()
            fused: list[dict[str, Any]] = []
            for item in candidates:
                key = (
                    item.get("page"),
                    item.get("type"),
                    item.get("text", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                fused.append(item)

            if not fused:
                continue

            try:
                reranked = self.reranker.rerank(
                    formulation,
                    fused,
                    min(
                        MULTI_DOC_RERANK_K_PER_DOCUMENT,
                        len(fused),
                    ),
                )
            except Exception:
                logger.exception(
                    "Per-document rerank failed "
                    f"document={document}"
                )
                reranked = []

            per_formulation_evidence.extend(reranked)

        if not per_formulation_evidence:
            return []

        # Final per-document keep: take the top
        # final_evidence_per_document after one more rerank
        # across the union of formulations.
        try:
            top = self.reranker.rerank(
                formulations[0],
                per_formulation_evidence,
                MULTI_DOC_FINAL_EVIDENCE_PER_DOCUMENT,
            )
        except Exception:
            logger.exception(
                "Final per-document rerank failed "
                f"document={document}"
            )
            top = per_formulation_evidence[
                :MULTI_DOC_FINAL_EVIDENCE_PER_DOCUMENT
            ]

        return list(top)

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    def retrieve(
        self,
        plan: QueryPlan,
    ) -> dict[str, Any]:
        """Run per-document retrieval and (optionally) comparison.

        Returns a dict with:
            evidence: list of evidence items (per-document kept)
            per_document_evidence: dict[document, list[item]]
            plan: plan.to_dict()
            comparison: comparison.to_dict() or None
        """

        log_payload: dict[str, Any] = {
            "intent": plan.intent,
            "target_field": plan.target_field,
            "documents": len(self.document_names),
            "formulations": len(plan.formulations),
        }

        per_document_evidence: dict[str, list[dict[str, Any]]] = {}
        documents_with_evidence: list[str] = []
        documents_without_evidence: list[str] = []

        for document in self.document_names:
            evidence = self._retrieve_for_document(
                document,
                plan.formulations,
            )
            per_document_evidence[document] = evidence
            if evidence:
                documents_with_evidence.append(document)
            else:
                documents_without_evidence.append(document)

        log_payload["documents_with_evidence"] = len(
            documents_with_evidence
        )
        log_payload["documents_without_evidence"] = len(
            documents_without_evidence
        )

        # Flatten evidence in document order to keep the LLM
        # context well-organized.
        flat_evidence: list[dict[str, Any]] = []
        for document in self.document_names:
            for item in per_document_evidence.get(document, []):
                copy = dict(item)
                copy.setdefault("document", document)
                copy.setdefault(
                    "aspect_query",
                    plan.formulations[0] if plan.formulations else plan.raw_question,
                )
                flat_evidence.append(copy)

        comparison_payload: dict[str, Any] | None = None
        if plan.uses_exhaustive and plan.target_field:
            records = extract_field_records(
                plan.target_field,
                per_document_evidence,
            )
            comparison = perform_comparison(
                plan.intent,
                plan.target_field,
                records,
                self.document_names,
            )
            comparison_payload = comparison.to_dict()
            log_payload["comparison_winner"] = (
                comparison_payload.get("winner", {}).get("document")
                if comparison_payload.get("winner")
                else None
            )
            log_payload["extracted_records"] = len(records)

        logger.info(
            "MultiDocRetrieval complete | "
            f"intent={log_payload['intent']} | "
            f"target={log_payload['target_field']} | "
            f"docs={log_payload['documents']} | "
            f"with_evidence="
            f"{log_payload['documents_with_evidence']} | "
            f"without_evidence="
            f"{log_payload['documents_without_evidence']} | "
            f"formulations={log_payload['formulations']}"
        )

        return {
            "evidence": flat_evidence,
            "per_document_evidence": per_document_evidence,
            "documents_with_evidence": documents_with_evidence,
            "documents_without_evidence": documents_without_evidence,
            "plan": plan.to_dict(),
            "comparison": comparison_payload,
        }
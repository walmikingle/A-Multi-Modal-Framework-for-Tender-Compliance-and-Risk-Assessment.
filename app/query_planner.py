"""
Deterministic query planner for Tender RAG system.
Analyzes user questions and returns structured metadata for retrieval planning.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QueryPlan:
    """Structured query plan for retrieval and answering."""
    intent: str          # normal | comparison
    scope: str           # current_document | all_documents | explicit_documents
    operation: str       # none | max | min | compare
    metric: Optional[str]  # quotation/financial_amount | emd | None
    retrieval_variants: list[str]  # alternative query phrasings for retrieval
    explicit_documents: list[str]  # document names if scope=explicit_documents


# Keyword sets for detection
COMPARISON_KEYWORDS = {
    "highest", "lowest", "maximum", "minimum", "largest", "smallest",
    "most", "least", "compare", "among all", "across all",
    "which tender", "which document", "which pdf", "top", "bottom"
}

EXPLICIT_DOC_PATTERNS = [
    r"(Tendernotice_\d+\.pdf)",
    r"according to\s+(\w+\.pdf)",
    r"in\s+(\w+\.pdf)",
    r"from\s+(\w+\.pdf)",
]

# Ordered list for deterministic detection - longer/more-specific phrases first
FINANCIAL_QUOTATION_TERMS = [
    "total quoted amount",
    "total bid price",
    "quoted amount",
    "quoted price",
    "price quoted",
    "quoted rate",
    "financial bid",
    "financial quote",
    "bid price",
    "price bid",
    "tender amount",
    "tender value",
    "estimated cost",
    "amount put to tender",
    "quotation",
    "price schedule",
]

EMD_TERMS = {
    "emd", "earnest money deposit", "earnest money", "bid security"
}

ALL_DOC_KEYWORDS = {
    "all", "every", "each", "among all", "across all", "all tenders",
    "all pdfs", "all documents"
}


def _detect_intent(question: str) -> str:
    """Detect if question is asking for comparison or normal lookup."""
    q_lower = question.lower()
    if any(kw in q_lower for kw in COMPARISON_KEYWORDS):
        return "comparison"
    return "normal"


def _detect_scope(question: str) -> str:
    """Detect document scope: current_document, all_documents, or explicit_documents."""
    q_lower = question.lower()

    # Check for explicit document references
    for pattern in EXPLICIT_DOC_PATTERNS:
        match = re.search(pattern, q_lower, re.IGNORECASE)
        if match:
            return "explicit_documents"

    # Check for all-documents keywords
    if any(kw in q_lower for kw in ALL_DOC_KEYWORDS):
        return "all_documents"

    # Comparison questions with "which tender/document" imply all_documents
    if any(kw in q_lower for kw in ["which tender", "which document", "which pdf"]):
        return "all_documents"

    # Default to current document
    return "current_document"


def _detect_operation(question: str) -> str:
    """Detect comparison operation: max, min, compare, or none."""
    q_lower = question.lower()

    max_keywords = {"highest", "maximum", "largest", "most", "top"}
    min_keywords = {"lowest", "minimum", "smallest", "least", "bottom"}
    compare_keywords = {"compare"}

    if any(kw in q_lower for kw in max_keywords):
        return "max"
    if any(kw in q_lower for kw in min_keywords):
        return "min"
    if any(kw in q_lower for kw in compare_keywords):
        return "compare"
    return "none"


def _detect_metric(question: str) -> Optional[str]:
    """Detect the metric/concept being queried."""
    q_lower = question.lower()

    # Check for EMD terms
    if any(term in q_lower for term in EMD_TERMS):
        return "emd"

    # Check for financial quotation terms (deterministic order)
    for term in FINANCIAL_QUOTATION_TERMS:
        if term in q_lower:
            return "quotation/financial_amount"

    return None


def _extract_explicit_documents(question: str) -> list[str]:
    """Extract explicit document names from question, preserving original casing."""
    documents = []

    for pattern in EXPLICIT_DOC_PATTERNS:
        # Search in original case to preserve filename casing
        matches = re.findall(pattern, question, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            if match not in documents:
                documents.append(match)

    return documents


def _build_retrieval_variants(question: str, metric: Optional[str], operation: str) -> list[str]:
    """Build alternative query phrasings for better retrieval coverage."""
    variants = [question]
    q_lower = question.lower()

    if metric == "emd":
        # Add EMD synonyms - case-insensitive replacement
        # Find all occurrences of EMD terms and replace
        emd_synonyms = ["earnest money deposit", "earnest money", "bid security"]
        for syn in emd_synonyms:
            # Replace "emd" case-insensitively
            variant = re.sub(r'\bemd\b', syn, question, flags=re.IGNORECASE)
            if variant != question:
                variants.append(variant)
            # Also replace "earnest money deposit" etc. with other synonyms
            variant2 = re.sub(r'\bearnest money deposit\b', syn, question, flags=re.IGNORECASE)
            if variant2 != question:
                variants.append(variant2)
            variant3 = re.sub(r'\bearnest money\b', syn, question, flags=re.IGNORECASE)
            if variant3 != question:
                variants.append(variant3)
            variant4 = re.sub(r'\bbid security\b', syn, question, flags=re.IGNORECASE)
            if variant4 != question:
                variants.append(variant4)

    elif metric == "quotation/financial_amount":
        # Generate meaningful rewritten variants, not just appended synonyms
        # Replace the financial term with alternatives
        financial_synonyms = [
            "quoted amount", "quoted price", "bid price", "financial bid",
            "tender amount", "price schedule", "tender value", "estimated cost",
            "amount put to tender", "total quoted amount", "price bid"
        ]

        # First, identify which financial term was used in the question
        # Use ordered list for deterministic detection
        detected_term = None
        for term in FINANCIAL_QUOTATION_TERMS:
            if term in q_lower:
                detected_term = term
                break

        if detected_term:
            # Replace the detected term with alternatives
            for syn in financial_synonyms:
                if syn != detected_term:
                    variant = re.sub(
                        re.escape(detected_term), syn, question, flags=re.IGNORECASE
                    )
                    if variant != question:
                        variants.append(variant)
        else:
            # No specific term detected, add key financial variants
            base_variants = [
                "highest quoted amount across tenders",
                "highest quoted price across tenders",
                "highest bid price across tenders",
                "highest financial bid amount",
                "highest tender amount across tenders",
                "highest amount put to tender",
            ]
            if operation == "min":
                base_variants = [v.replace("highest", "lowest") for v in base_variants]
            elif operation == "compare":
                base_variants = [v.replace("highest", "compare") for v in base_variants]
            variants.extend(base_variants)

    # Add operation-specific variants
    if operation == "max":
        variants.append(re.sub(r'\bhighest\b', "maximum", question, flags=re.IGNORECASE))
        variants.append(re.sub(r'\bhighest\b', "largest", question, flags=re.IGNORECASE))
        variants.append(re.sub(r'\bhighest\b', "most", question, flags=re.IGNORECASE))
    elif operation == "min":
        variants.append(re.sub(r'\blowest\b', "minimum", question, flags=re.IGNORECASE))
        variants.append(re.sub(r'\blowest\b', "smallest", question, flags=re.IGNORECASE))
        variants.append(re.sub(r'\blowest\b', "least", question, flags=re.IGNORECASE))
    elif operation == "compare":
        variants.append(re.sub(r'\bcompare\b', "compare all", question, flags=re.IGNORECASE))

    # Deduplicate preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)

    return unique_variants


def plan_query(question: str) -> QueryPlan:
    """
    Analyze a user question and return a structured QueryPlan.

    Args:
        question: The user's natural language question.

    Returns:
        QueryPlan with intent, scope, operation, metric, and retrieval variants.
    """
    intent = _detect_intent(question)
    scope = _detect_scope(question)
    operation = _detect_operation(question)
    metric = _detect_metric(question)
    explicit_documents = _extract_explicit_documents(question)
    retrieval_variants = _build_retrieval_variants(question, metric, operation)

    return QueryPlan(
        intent=intent,
        scope=scope,
        operation=operation,
        metric=metric,
        retrieval_variants=retrieval_variants,
        explicit_documents=explicit_documents
    )


if __name__ == "__main__":
    # Test with 10 distinct questions
    test_questions = [
        # Original 5
        "Which tender has the highest quotation among all these?",
        "What is the highest EMD among all the PDFs?",
        "Which tender has the lowest quoted price?",
        "Compare the quoted amounts of all tenders.",
        "According to Tendernotice_10.pdf, what is the bid validity period?",
        # New distinct tests
        "What is the highest quoted amount across all tenders?",
        "Compare the financial bids of Tendernotice_3.pdf and Tendernotice_12.pdf.",
        "What is the bid security amount in Tendernotice_8.pdf?",
        "What is the EMD in Tendernotice_12.pdf?",
        "What is the estimated cost of Tendernotice_5.pdf?",
    ]

    print("=" * 80)
    print("QUERY PLANNER TEST OUTPUT")
    print("=" * 80)

    for i, question in enumerate(test_questions, 1):
        plan = plan_query(question)
        print(f"\n--- Example {i} ---")
        print(f"Question: {question}")
        print(f"  intent              : {plan.intent}")
        print(f"  scope               : {plan.scope}")
        print(f"  operation           : {plan.operation}")
        print(f"  metric              : {plan.metric}")
        print(f"  explicit_documents  : {plan.explicit_documents}")
        print(f"  retrieval_variants  :")
        for v in plan.retrieval_variants[:8]:
            print(f"    - {v}")
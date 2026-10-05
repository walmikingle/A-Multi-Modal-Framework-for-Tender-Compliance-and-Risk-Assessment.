"""Real multi-document retrieval test against cached indexes.

Does NOT call the LLM for the structural coverage test (just
prints coverage + extracted comparisons). The full ask() path
with the LLM is exercised separately for one comparison query.
"""

import io
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multi_document_test import (
    MultiDocumentPipeline,
    get_test_pdfs,
)
from app.multi_doc_retrieval import (
    build_query_plan,
    normalize_rupee_value,
    normalize_day_value,
)


TEST_QUESTIONS = [
    "What is the EMD or Bid Security amount in NOC26062026.pdf?",
    "Which tender has the highest quotation among all these?",
    "What is the highest EMD among all the PDFs?",
    "List the bid validity period for each of the 12 tender documents.",
]


def run_coverage_test(pipeline, question):
    plan = build_query_plan(question, is_multi_document=True)
    print(f"\n=== Question: {question}")
    print(f"   intent={plan.intent} target={plan.target_field}")
    print(f"   exhaustive={plan.uses_exhaustive}")
    print(f"   formulations ({len(plan.formulations)}):")
    for f in plan.formulations[:5]:
        print(f"     - {f[:90]}")
    if len(plan.formulations) > 5:
        print(f"     ... +{len(plan.formulations)-5} more")

    if not plan.uses_exhaustive:
        print("   -> not exhaustive; routing to existing path")
        return None

    t0 = time.perf_counter()
    result = pipeline.multi_doc_retrieval.retrieve(plan)
    dt = time.perf_counter() - t0

    print(f"   retrieval time: {dt:.2f}s")
    print(
        f"   docs with evidence: "
        f"{len(result['documents_with_evidence'])} "
        f"/ {len(pipeline.pdf_paths)}"
    )
    for d in result['documents_with_evidence']:
        ev = result['per_document_evidence'][d]
        print(
            f"     [+] {d}: "
            f"{len(ev)} block(s), "
            f"page {ev[0].get('page')}"
        )
    for d in result['documents_without_evidence']:
        print(f"     [-] {d}")

    if result.get('comparison'):
        c = result['comparison']
        print(f"   comparison intent: {c['intent']} field: {c['field']}")
        if c.get('winner'):
            w = c['winner']
            print(
                f"   WINNER: {w['document']} = {w['value_raw']!r} "
                f"({w['value_numeric']}) "
                f"page {w['page']}"
            )
        print(f"   extracted records: {len(c['all_records'])}")
        print(f"   missing documents: {len(c['missing_documents'])}")

    return result


def main():
    pdfs = get_test_pdfs()
    print(f"Loading {len(pdfs)} PDFs...")
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        pipeline = MultiDocumentPipeline(pdfs)
    print(f"Pipeline ready.")

    for q in TEST_QUESTIONS:
        run_coverage_test(pipeline, q)


if __name__ == "__main__":
    main()
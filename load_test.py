import csv
import json
import time
from pathlib import Path

from multi_document_test import (
    MultiDocumentPipeline,
    get_test_pdfs,
    get_ram_mb,
)


# ============================================================
# LOAD TEST QUERIES
# ============================================================

TEST_QUERIES = [

    {
        "id": 1,
        "name": "Single document - turnover",
        "question": (
            "According to NOC26062026.pdf, what is the "
            "minimum average annual turnover required for the bidder, "
            "and which financial years are considered?"
        ),
    },

    {
        "id": 2,
        "name": "Single document - EMD",
        "question": (
            "According to 25f2785.pdf, what is the EMD amount "
            "and what forms of EMD are accepted?"
        ),
    },

    {
        "id": 3,
        "name": "Two document comparison",
        "question": (
            "Compare the bid validity periods of "
            "NOC26062026.pdf and 25f2785.pdf."
        ),
    },

    {
        "id": 4,
        "name": "Multi-part single document",
        "question": (
            "According to Tendernotice_10.pdf, what is the "
            "EMD validity period, and is it the same as the "
            "bid validity period?"
        ),
    },

    {
        "id": 5,
        "name": "Single document - bid validity",
        "question": (
            "According to NOC26062026.pdf, what is the "
            "bid validity period?"
        ),
    },

    {
        "id": 6,
        "name": "Three document comparison",
        "question": (
            "Compare the bid validity periods of "
            "NOC26062026.pdf, 25f2785.pdf, and Tendernotice_10.pdf."
        ),
    },

    {
        "id": 7,
        "name": "Multi-document turnover",
        "question": (
            "What are the minimum annual turnover requirements "
            "mentioned in NOC26062026.pdf and Tendernotice_8.pdf?"
        ),
    },

    {
        "id": 8,
        "name": "Absent information / hallucination test",
        "question": (
            "According to the provided tender documents, "
            "what is the colour of the tender office building?"
        ),
    },

    {
        "id": 9,
        "name": "Cross-document evidence isolation",
        "question": (
            "According to Tendernotice_10.pdf, what is the "
            "bid validity period?"
        ),
    },

    {
        "id": 10,
        "name": "Repeated query",
        "question": (
            "Compare the bid validity periods of "
            "NOC26062026.pdf and 25f2785.pdf."
        ),
    },

    {
        "id": 11,
        "name": "Multi-part multi-document",
        "question": (
            "Compare the bid validity periods of "
            "NOC26062026.pdf and 25f2785.pdf, and also state "
            "the bid validity period in Tendernotice_10.pdf."
        ),
    },

    {
        "id": 12,
        "name": "Three document mixed query",
        "question": (
            "For NOC26062026.pdf and 25f2785.pdf, compare the "
            "bid validity periods, and for Tendernotice_10.pdf "
            "state the EMD validity period."
        ),
    },
]


# ============================================================
# RESULT CLASSIFICATION
# ============================================================

def classify_answer(answer):

    if not answer:
        return "EMPTY"

    text = answer.lower()

    abstention_terms = [
        "not specified",
        "not explicitly stated",
        "not found",
        "cannot be determined",
        "cannot confirm",
        "insufficient information",
        "not available in the provided",
        "unable to determine",
    ]

    for term in abstention_terms:
        if term in text:
            return "ABSTENTION/UNCERTAIN"

    return "ANSWER_RETURNED"


# ============================================================
# MAIN LOAD TEST
# ============================================================

def main():

    print("=" * 80)
    print("12-DOCUMENT RAG LOAD / STABILITY TEST")
    print("=" * 80)

    pdfs = get_test_pdfs()

    if not pdfs:
        raise RuntimeError("No test tender PDFs found.")

    print("\nDocuments selected:")

    for index, pdf in enumerate(pdfs, start=1):
        print(f"{index:2d}. {pdf.name}")

    print(
        f"\nTotal documents: {len(pdfs)}"
    )

    print("\nInitializing multi-document RAG...")

    init_start = time.perf_counter()

    rag = MultiDocumentPipeline(pdfs)

    init_time = time.perf_counter() - init_start

    init_ram = get_ram_mb()

    print(
        f"\nInitialization time: "
        f"{init_time:.2f} sec"
    )

    print(
        f"RAM after initialization: "
        f"{init_ram:.2f} MB"
    )

    results = []

    peak_ram = init_ram
    total_query_time = 0.0

    print("\n" + "=" * 80)
    print("STARTING QUERY LOAD")
    print("=" * 80)

    for test in TEST_QUERIES:

        query_id = test["id"]
        query_name = test["name"]
        question = test["question"]

        print("\n" + "-" * 80)
        print(
            f"TEST {query_id}/"
            f"{len(TEST_QUERIES)}: "
            f"{query_name}"
        )

        print(
            f"Question: {question}"
        )

        ram_before = get_ram_mb()

        query_start = time.perf_counter()

        error = ""

        answer = ""
        candidate_count = None

        try:

            response = rag.ask(question)

            if response:

                answer = response.get(
                    "answer",
                    "",
                )

                candidate_count = response.get(
                    "candidate_count",
                    None,
                )

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: "
                f"{str(exc)}"
            )

        query_time = (
            time.perf_counter()
            - query_start
        )

        ram_after = get_ram_mb()

        ram_change = (
            ram_after
            - ram_before
        )

        peak_ram = max(
            peak_ram,
            ram_after,
        )

        total_query_time += query_time

        status = (
            "ERROR"
            if error
            else classify_answer(answer)
        )

        result = {
            "id": query_id,
            "name": query_name,
            "question": question,
            "status": status,
            "query_time_sec": round(
                query_time,
                3,
            ),
            "ram_before_mb": round(
                ram_before,
                2,
            ),
            "ram_after_mb": round(
                ram_after,
                2,
            ),
            "ram_change_mb": round(
                ram_change,
                2,
            ),
            "candidate_count": candidate_count,
            "answer": answer,
            "error": error,
        }

        results.append(result)

        print(
            f"\nSTATUS: {status}"
        )

        print(
            f"Query time: "
            f"{query_time:.2f} sec"
        )

        print(
            f"RAM change: "
            f"{ram_change:+.2f} MB"
        )

        if error:
            print(
                f"ERROR: {error}"
            )

        else:
            print(
                "\nAnswer preview:"
            )

            print(
                answer[:600]
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    completed = [
        r
        for r in results
        if r["status"] != "ERROR"
    ]

    errors = [
        r
        for r in results
        if r["status"] == "ERROR"
    ]

    abstentions = [
        r
        for r in results
        if r["status"] == "ABSTENTION/UNCERTAIN"
    ]

    answers = [
        r
        for r in results
        if r["status"] == "ANSWER_RETURNED"
    ]

    average_query_time = (
        total_query_time / len(completed)
        if completed
        else 0
    )

    print("\n" + "=" * 80)
    print("LOAD TEST SUMMARY")
    print("=" * 80)

    print(
        f"Documents tested       : {len(pdfs)}"
    )

    print(
        f"Queries executed       : {len(results)}"
    )

    print(
        f"Successful executions  : {len(completed)}"
    )

    print(
        f"Errors                 : {len(errors)}"
    )

    print(
        f"Answers returned       : {len(answers)}"
    )

    print(
        f"Abstention/uncertain   : {len(abstentions)}"
    )

    print(
        f"Average query time     : "
        f"{average_query_time:.2f} sec"
    )

    print(
        f"Total query time       : "
        f"{total_query_time:.2f} sec"
    )

    print(
        f"Peak observed RAM      : "
        f"{peak_ram:.2f} MB"
    )

    print(
        f"RAM at initialization  : "
        f"{init_ram:.2f} MB"
    )

    print(
        f"Net RAM change         : "
        f"{peak_ram - init_ram:+.2f} MB"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = Path(
        f"load_test_results_{timestamp}.csv"
    )

    json_path = Path(
        f"load_test_results_{timestamp}.json"
    )

    # CSV
    fieldnames = [
        "id",
        "name",
        "question",
        "status",
        "query_time_sec",
        "ram_before_mb",
        "ram_after_mb",
        "ram_change_mb",
        "candidate_count",
        "answer",
        "error",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # JSON
    output = {
        "documents": [
            pdf.name
            for pdf in pdfs
        ],
        "document_count": len(pdfs),
        "query_count": len(results),
        "initialization_time_sec": round(
            init_time,
            3,
        ),
        "initialization_ram_mb": round(
            init_ram,
            2,
        ),
        "peak_ram_mb": round(
            peak_ram,
            2,
        ),
        "average_query_time_sec": round(
            average_query_time,
            3,
        ),
        "total_query_time_sec": round(
            total_query_time,
            3,
        ),
        "errors": len(errors),
        "answers_returned": len(answers),
        "abstentions_or_uncertain": len(abstentions),
        "results": results,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nResults saved:")

    print(
        f"CSV  : {csv_path}"
    )

    print(
        f"JSON : {json_path}"
    )

    print("\nLoad test completed.")


if __name__ == "__main__":
    main()

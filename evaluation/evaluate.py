import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow imports from the project root.
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from app.pipeline import RAGPipeline


TEST_PDF_DIR = (
    PROJECT_ROOT
    / "test_tenders"
)

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)


def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def evaluate_pdf(
    pdf_path,
    questions
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"EVALUATING: {pdf_path.name}"
    )

    print(
        "=" * 70
    )

    pipeline = RAGPipeline(
        pdf_path
    )

    results = []

    for index, test in enumerate(
        questions,
        start=1
    ):

        question = test["question"]

        expected_answer = (
            test.get(
                "expected_answer"
            )
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"Question {index}: "
            f"{question}"
        )

        start_time = time.perf_counter()

        response = pipeline.ask(
            question
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result = {
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": response["answer"],
            "processing_time_seconds": round(
                elapsed,
                3
            ),
            "candidate_count": (
                response["candidate_count"]
            ),
            "reranked_results": [
                {
                    "rank": rank,
                    "page": item.get(
                        "page"
                    ),
                    "score": item.get(
                        "rerank_score"
                    ),
                    "text": item.get(
                        "text",
                        ""
                    )
                }
                for rank, item in enumerate(
                    response[
                        "reranked_results"
                    ],
                    start=1
                )
            ]
        }

        results.append(
            result
        )

        print(
            f"\nEvaluation time: "
            f"{elapsed:.3f}s"
        )

    return results


def main():

    QUESTIONS = load_questions()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = sorted(
        TEST_PDF_DIR.glob(
            "*.pdf"
        )
    )

    if not pdf_files:

        raise FileNotFoundError(
            "No PDF files found in "
            f"{TEST_PDF_DIR}"
        )

    all_results = {}

    for pdf_path in pdf_files:

        filename = (
            pdf_path.name
        )

        if filename not in QUESTIONS:

            print(
                f"\nSkipping {filename}: "
                "no questions defined."
            )

            continue

        results = evaluate_pdf(
            pdf_path,
            QUESTIONS[filename]
        )

        all_results[
            filename
        ] = results

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_file = (
        RESULTS_DIR
        / f"evaluation_{timestamp}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        f"Results saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":

    main()
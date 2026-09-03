import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

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

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)


def evaluate_document(pdf_path):

    print(
        "\n" + "=" * 80
    )

    print(
        f"TESTING DOCUMENT: {pdf_path.name}"
    )

    print(
        "=" * 80
    )

    start_time = time.perf_counter()

    result = {
        "pdf": pdf_path.name,
        "path": str(pdf_path),
        "status": "FAILED",
        "processing_time_seconds": None,
        "error": None
    }

    pipeline = None

    try:

        pipeline = RAGPipeline(
            pdf_path
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result[
            "status"
        ] = "SUCCESS"

        result[
            "processing_time_seconds"
        ] = round(
            elapsed,
            3
        )

        print(
            f"\nSUCCESS: {pdf_path.name}"
        )

        print(
            f"Total time: {elapsed:.3f}s"
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result[
            "processing_time_seconds"
        ] = round(
            elapsed,
            3
        )

        result[
            "error"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"\nFAILED: {pdf_path.name}"
        )

        print(
            f"Error: {type(exc).__name__}: {exc}"
        )

    finally:

        # Release large models before
        # processing the next tender.
        del pipeline

        gc.collect()

    return result


def main():

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
            f"No PDF files found in "
            f"{TEST_PDF_DIR}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "MULTI-TENDER PIPELINE EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    print(
        f"Directory: {TEST_PDF_DIR}"
    )

    all_results = []

    overall_start = (
        time.perf_counter()
    )

    for index, pdf_path in enumerate(
        pdf_files,
        start=1
    ):

        print(
            f"\n[{index}/{len(pdf_files)}]"
        )

        result = evaluate_document(
            pdf_path
        )

        all_results.append(
            result
        )

    overall_elapsed = (
        time.perf_counter()
        - overall_start
    )

    successful = sum(
        1
        for result in all_results
        if result["status"]
        == "SUCCESS"
    )

    failed = (
        len(all_results)
        - successful
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_file = (
        RESULTS_DIR
        / f"multi_tender_evaluation_{timestamp}.json"
    )

    report = {
        "evaluation_timestamp": (
            datetime.now().isoformat()
        ),
        "total_documents": len(
            all_results
        ),
        "successful_documents": successful,
        "failed_documents": failed,
        "total_processing_time_seconds": round(
            overall_elapsed,
            3
        ),
        "results": all_results
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "MULTI-TENDER EVALUATION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Total PDFs: {len(all_results)}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total time: {overall_elapsed:.3f}s"
    )

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":

    main()
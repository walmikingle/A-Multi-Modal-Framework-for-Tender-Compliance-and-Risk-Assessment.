import argparse
from pathlib import Path

from app.config import PARSER
from app.pipeline import RAGPipeline


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="RAG system for question answering over tender PDF documents."
    )

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the tender PDF document.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "The input file must be a PDF."
        )

    print("\n" + "=" * 60)
    print("RAG TENDER DOCUMENT")
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print(f"Parser: {PARSER}")

    rag = RAGPipeline(pdf_path)

    while True:
        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if question.lower().strip() == "exit":
            break

        if not question.strip():
            print("Please enter a non-empty question.")
            continue

        rag.ask(question)


if __name__ == "__main__":
    main()
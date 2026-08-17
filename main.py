from app.config import PDF_PATH
from app.pipeline import RAGPipeline


def main():

    if not PDF_PATH:

        raise ValueError(
            "PDF_PATH is not configured."
        )

    rag = RAGPipeline(
        PDF_PATH
    )

    while True:

        question = input(
            "\nAsk a question "
            "(type 'exit' to quit): "
        )

        if question.lower() == "exit":

            break

        answer = rag.ask(
            question
        )

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()
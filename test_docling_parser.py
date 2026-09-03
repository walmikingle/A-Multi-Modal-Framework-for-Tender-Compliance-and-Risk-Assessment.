from app.chunker import get_text_splitter
from app.parser import process_pdf


PDF_PATH = (
    r"D:\Rag\1-s2.0-S1877050926007295-main.pdf"
)

DATA_DIR = r"D:\Rag\data\docling_test"


def main():

    print("=" * 60)
    print("DOCLING PARSER TEST")
    print("=" * 60)

    splitter = get_text_splitter()

    items = process_pdf(
        PDF_PATH,
        DATA_DIR,
        splitter,
        parser="docling"
    )

    print(
        f"\nTotal items: {len(items)}"
    )

    text_items = [
        item
        for item in items
        if item["type"] == "text"
    ]

    table_items = [
        item
        for item in items
        if item["type"] == "table"
    ]

    print(
        f"Text items: {len(text_items)}"
    )

    print(
        f"Table items: {len(table_items)}"
    )

    print("\n" + "=" * 60)
    print("FIRST 10 TEXT ITEMS")
    print("=" * 60)

    for i, item in enumerate(
        text_items[:10],
        start=1
    ):

        print(
            f"\n{i}. Page: {item['page']}"
        )

        print(
            item["text"][:300]
        )

    print("\n" + "=" * 60)
    print("TABLE ITEMS")
    print("=" * 60)

    for i, item in enumerate(
        table_items,
        start=1
    ):

        print(
            f"\nTable {i}"
        )

        print(
            f"Page: {item['page']}"
        )

        print(
            item["text"]
        )


if __name__ == "__main__":
    main()
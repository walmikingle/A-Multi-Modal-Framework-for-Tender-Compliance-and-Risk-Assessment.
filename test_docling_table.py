from pathlib import Path

from docling.document_converter import DocumentConverter


PDF_PATH = Path(
    r"D:\Rag\1-s2.0-S1877050926007295-main.pdf"
)


def main():

    print("=" * 60)
    print("DOCLING TABLE TEST")
    print("=" * 60)

    converter = DocumentConverter()

    result = converter.convert(
        PDF_PATH
    )

    document = result.document

    print(
        f"\nPages: {len(document.pages)}"
    )

    print(
        f"Tables: {len(document.tables)}"
    )

    for i, table in enumerate(
        document.tables,
        start=1
    ):

        print("\n" + "=" * 60)
        print(f"TABLE {i}")
        print("=" * 60)

        dataframe = table.export_to_dataframe(
            doc=document
        )

        print(
            dataframe.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
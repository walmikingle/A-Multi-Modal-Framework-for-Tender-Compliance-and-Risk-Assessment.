import os
import base64
import tabula
import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_directories(base_dir):
    directories = [
        "images",
        "text",
        "tables",
        "page_images"
    ]

    for directory in directories:
        os.makedirs(
            os.path.join(base_dir, directory),
            exist_ok=True
        )


def process_pdf(pdf_path, base_dir, text_splitter):

    doc = pymupdf.open(pdf_path)

    items = []

    for page_num in range(len(doc)):

        page = doc[page_num]

        # -------------------------
        # TEXT
        # -------------------------

        text = page.get_text()

        chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):

            text_file = os.path.join(
                base_dir,
                "text",
                f"text_{page_num}_{i}.txt"
            )

            with open(text_file, "w", encoding="utf-8") as f:
                f.write(chunk)

            items.append({
                "page": page_num,
                "type": "text",
                "text": chunk,
                "path": text_file
            })

        # -------------------------
        # TABLES
        # -------------------------

        try:

            tables = tabula.read_pdf(
                pdf_path,
                pages=page_num + 1,
                multiple_tables=True
            )

            if tables:

                for table_idx, table in enumerate(tables):

                    table_text = "\n".join(
                        [
                            " | ".join(map(str, row))
                            for row in table.values
                        ]
                    )

                    table_file = os.path.join(
                        base_dir,
                        "tables",
                        f"table_{page_num}_{table_idx}.txt"
                    )

                    with open(
                        table_file,
                        "w",
                        encoding="utf-8"
                    ) as f:
                        f.write(table_text)

                    items.append({
                        "page": page_num,
                        "type": "table",
                        "text": table_text,
                        "path": table_file
                    })

        except Exception as e:

            print(
                f"Table extraction failed "
                f"on page {page_num}: {e}"
            )

        # -------------------------
        # EMBEDDED IMAGES
        # -------------------------

        for idx, image in enumerate(page.get_images()):

            xref = image[0]

            pix = pymupdf.Pixmap(
                doc,
                xref
            )

            image_path = os.path.join(
                base_dir,
                "images",
                f"image_{page_num}_{idx}_{xref}.png"
            )

            pix.save(image_path)

            with open(image_path, "rb") as f:

                encoded_image = base64.b64encode(
                    f.read()
                ).decode("utf-8")

            items.append({
                "page": page_num,
                "type": "image",
                "path": image_path,
                "image": encoded_image
            })

        # -------------------------
        # PAGE IMAGE
        # -------------------------

        page_pixmap = page.get_pixmap()

        page_image_path = os.path.join(
            base_dir,
            "page_images",
            f"page_{page_num:03d}.png"
        )

        page_pixmap.save(page_image_path)

        with open(
            page_image_path,
            "rb"
        ) as f:

            page_image = base64.b64encode(
                f.read()
            ).decode("utf-8")

        items.append({
            "page": page_num,
            "type": "page",
            "path": page_image_path,
            "image": page_image
        })

    doc.close()

    return items
import os
import base64
import re
import hashlib

import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_directories(base_dir):
    """
    Create document-specific extraction directories.

    base_dir should point to:

        data/tenders/<document_name>_<hash>/
    """

    directories = [
        "images",
        "text",
        "tables",
        "page_images"
    ]

    for directory in directories:

        os.makedirs(
            os.path.join(
                base_dir,
                directory
            ),
            exist_ok=True
        )


def get_document_output_dir(
    pdf_path,
    data_dir
):
    """
    Create a meaningful and unique extraction directory
    for a PDF.

    Example:

        Tendernotice_1.pdf

    becomes:

        data/tenders/Tendernotice_1_a1b2c3d4/
    """

    pdf_path = os.path.abspath(
        str(pdf_path)
    )

    filename = os.path.basename(
        pdf_path
    )

    stem = os.path.splitext(
        filename
    )[0]

    # Make filename safe for Windows paths.
    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        stem
    )

    safe_stem = safe_stem.strip(
        "._-"
    )

    if not safe_stem:
        safe_stem = "document"

    # Generate a short content hash so two PDFs
    # with the same filename remain separate.
    sha256 = hashlib.sha256()

    with open(
        pdf_path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    hash_prefix = (
        sha256.hexdigest()[:8]
    )

    document_dir = (
        os.path.join(
            data_dir,
            "tenders",
            f"{safe_stem}_{hash_prefix}"
        )
    )

    create_directories(
        document_dir
    )

    return document_dir


def _save_text_item(
    items,
    base_dir,
    page_num,
    chunk_idx,
    text
):

    text = text.strip()

    if not text:
        return

    text_file = os.path.join(
        base_dir,
        "text",
        f"text_{page_num}_{chunk_idx}.txt"
    )

    with open(
        text_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            text
        )

    items.append({
        "page": page_num,
        "type": "text",
        "text": text,
        "path": text_file
    })


def _save_table_item(
    items,
    base_dir,
    table_idx,
    row_idx,
    page_num,
    table_text
):

    table_text = table_text.strip()

    if not table_text:
        return

    table_file = os.path.join(
        base_dir,
        "tables",
        f"table_{table_idx}_row_{row_idx}.txt"
    )

    with open(
        table_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            table_text
        )

    items.append({
        "page": page_num,
        "type": "table",
        "text": table_text,
        "path": table_file,
        "table_index": table_idx,
        "row_index": row_idx
    })


def _extract_page_assets(
    pdf_path,
    base_dir,
    items
):

    doc = pymupdf.open(
        pdf_path
    )

    for page_num in range(
        len(doc)
    ):

        page = doc[page_num]

        # =========================================
        # EMBEDDED IMAGES
        # =========================================

        for idx, image in enumerate(
            page.get_images()
        ):

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

            # PNG output supports RGB/RGBA.  Convert CMYK and other
            # non-RGB pixmaps before saving them.
            if pix.n - pix.alpha > 3:

                pix = pymupdf.Pixmap(
                    pymupdf.csRGB,
                    pix
                )

            pix.save(
                image_path
            )

            with open(
                image_path,
                "rb"
            ) as file:

                encoded_image = (
                    base64.b64encode(
                        file.read()
                    ).decode(
                        "utf-8"
                    )
                )

            items.append({
                "page": page_num,
                "type": "image",
                "path": image_path,
                "image": encoded_image
            })

            pix = None

        # =========================================
        # PAGE IMAGE
        # =========================================

        page_pixmap = page.get_pixmap()

        page_image_path = os.path.join(
            base_dir,
            "page_images",
            f"page_{page_num:03d}.png"
        )

        page_pixmap.save(
            page_image_path
        )

        with open(
            page_image_path,
            "rb"
        ) as file:

            page_image = (
                base64.b64encode(
                    file.read()
                ).decode(
                    "utf-8"
                )
            )

        items.append({
            "page": page_num,
            "type": "page",
            "path": page_image_path,
            "image": page_image
        })

    doc.close()


def process_pdf_pymupdf(
    pdf_path,
    base_dir,
    text_splitter
):
    """
    Existing PyMuPDF parser.

    Kept as the fallback parser.
    """

    create_directories(
        base_dir
    )

    doc = pymupdf.open(
        pdf_path
    )

    items = []

    for page_num in range(
        len(doc)
    ):

        page = doc[page_num]

        text = page.get_text()

        chunks = (
            text_splitter.split_text(
                text
            )
        )

        for i, chunk in enumerate(
            chunks
        ):

            _save_text_item(
                items,
                base_dir,
                page_num,
                i,
                chunk
            )

    doc.close()

    _extract_page_assets(
        pdf_path,
        base_dir,
        items
    )

    return items


def process_pdf_docling(
    pdf_path,
    base_dir,
    text_splitter
):
    """
    Docling-based PDF parser.

    Text is grouped by page before chunking.
    Tables are stored row-by-row.
    Images are handled by PyMuPDF.
    """

    from docling_core.types.doc import (
    TextItem,
    TableItem
)
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption
    )
    from docling.backend.docling_parse_v4_backend import (
        DoclingParseV4DocumentBackend
    )

    create_directories(
        base_dir
    )

    print(
        "\nLoading Docling..."
    )

    pipeline_options = PdfPipelineOptions()

    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseV4DocumentBackend
            )
        }
    )

    print(
        "Converting PDF with Docling..."
    )

    result = converter.convert(
        pdf_path
    )

    document = result.document

    print(
        f"Docling conversion complete. "
        f"Pages: {len(document.pages)}"
    )

    print(
        f"Docling tables detected: "
        f"{len(document.tables)}"
    )

    print(
        f"Docling pictures detected: "
        f"{len(document.pictures)}"
    )

    items = []

    # =========================================
    # GROUP TEXT BY PAGE
    # =========================================

    page_text = {}

    for item, level in (
        document.iterate_items()
    ):

        if not isinstance(
            item,
            TextItem
        ):
            continue

        text = item.text.strip()

        if not text:
            continue

        page_num = None

        if item.prov:

            page_num = (
                item.prov[0].page_no
            )

            if page_num is not None:

                page_num -= 1

        if page_num is None:
            continue

        if page_num not in page_text:

            page_text[
                page_num
            ] = []

        page_text[
            page_num
        ].append(
            text
        )

    # =========================================
    # CHUNK TEXT PAGE BY PAGE
    # =========================================

    text_chunk_counter = 0

    for page_num in sorted(
        page_text.keys()
    ):

        page_content = "\n\n".join(
            page_text[page_num]
        )

        chunks = (
            text_splitter.split_text(
                page_content
            )
        )

        for chunk in chunks:

            _save_text_item(
                items,
                base_dir,
                page_num,
                text_chunk_counter,
                chunk
            )

            text_chunk_counter += 1

    # =========================================
    # TABLES
    # =========================================

    table_counter = 0

    for table in document.tables:

        dataframe = (
            table.export_to_dataframe(
                doc=document
            )
        )

        page_num = None

        if table.prov:

            page_num = (
                table.prov[0].page_no
            )

            if page_num is not None:

                page_num -= 1

        columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        for row_idx, row in enumerate(
            dataframe.itertuples(
                index=False,
                name=None
            )
        ):

            row_values = [
                str(value).strip()
                for value in row
            ]

            row_text_parts = []

            for column, value in zip(
                columns,
                row_values
            ):

                if (
                    not value
                    or value.lower() == "nan"
                ):

                    continue

                row_text_parts.append(
                    f"{column}: {value}"
                )

            if not row_text_parts:
                continue

            table_row_text = "\n".join(
                row_text_parts
            )

            _save_table_item(
                items,
                base_dir,
                table_counter,
                row_idx,
                page_num,
                table_row_text
            )

        table_counter += 1

    # =========================================
    # PAGE / EMBEDDED IMAGES
    # =========================================

    _extract_page_assets(
        pdf_path,
        base_dir,
        items
    )

    return items


def process_pdf(
    pdf_path,
    base_dir,
    text_splitter,
    parser="pymupdf"
):
    """
    Main parser entry point.

    parser:
        "docling"
        "pymupdf"

    base_dir is the document-specific
    extraction directory.
    """

    parser = parser.lower().strip()

    if parser == "docling":

        return process_pdf_docling(
            pdf_path,
            base_dir,
            text_splitter
        )

    if parser == "pymupdf":

        return process_pdf_pymupdf(
            pdf_path,
            base_dir,
            text_splitter
        )

    raise ValueError(
        f"Unsupported parser: {parser}. "
        f"Use 'docling' or 'pymupdf'."
    )

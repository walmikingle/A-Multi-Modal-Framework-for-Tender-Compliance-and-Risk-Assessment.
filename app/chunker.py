from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def get_text_splitter():

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
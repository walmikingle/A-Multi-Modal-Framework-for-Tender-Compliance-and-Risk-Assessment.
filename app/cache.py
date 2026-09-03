import hashlib
import json
import pickle
from pathlib import Path


CACHE_VERSION = "8.0"


def create_config_fingerprint(
    config
):
    """
    Create a deterministic SHA-256 fingerprint
    from cache-relevant configuration.
    """

    config_json = json.dumps(
        config,
        sort_keys=True,
        separators=(
            ",",
            ":"
        )
    )

    return hashlib.sha256(
        config_json.encode(
            "utf-8"
        )
    ).hexdigest()


class DocumentCache:

    def __init__(
        self,
        cache_dir
    ):

        self.cache_dir = Path(
            cache_dir
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # -----------------------------------------
    # Generate SHA-256 hash of PDF
    # -----------------------------------------

    def get_file_hash(
        self,
        file_path
    ):

        sha256 = hashlib.sha256()

        with open(
            file_path,
            "rb"
        ) as f:

            while True:

                chunk = f.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                sha256.update(
                    chunk
                )

        return sha256.hexdigest()

    # -----------------------------------------
    # Get cache directory for PDF
    # -----------------------------------------

    def get_cache_path(
        self,
        pdf_path
    ):

        file_hash = (
            self.get_file_hash(
                pdf_path
            )
        )

        return (
            self.cache_dir
            / file_hash
        )

    # -----------------------------------------
    # Check whether valid cache exists
    # -----------------------------------------

    def exists(
        self,
        pdf_path,
        parser=None,
        config_fingerprint=None
    ):

        cache_path = (
            self.get_cache_path(
                pdf_path
            )
        )

        metadata_path = (
            cache_path
            / "metadata.json"
        )

        required_files = [
            "metadata.json",
            "items.pkl",
            "faiss.index",
            "sparse_embeddings.pkl"
        ]

        # Check required files.
        if not all(
            (
                cache_path
                / filename
            ).exists()
            for filename in required_files
        ):

            return False

        # Load metadata.
        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as f:

                metadata = json.load(
                    f
                )

        except (
            OSError,
            json.JSONDecodeError
        ):

            return False

        # -----------------------------------------
        # Validate cache version
        # -----------------------------------------

        if metadata.get(
            "cache_version"
        ) != CACHE_VERSION:

            return False

        # -----------------------------------------
        # Validate PDF hash
        # -----------------------------------------

        current_pdf_hash = (
            self.get_file_hash(
                pdf_path
            )
        )

        if metadata.get(
            "pdf_hash"
        ) != current_pdf_hash:

            return False

        # -----------------------------------------
        # Validate parser
        # -----------------------------------------

        if (
            parser is not None
            and metadata.get(
                "parser"
            ) != parser
        ):

            return False

        # -----------------------------------------
        # Validate configuration fingerprint
        # -----------------------------------------

        if (
            config_fingerprint is not None
            and metadata.get(
                "config_fingerprint"
            ) != config_fingerprint
        ):

            return False

        return True

    # -----------------------------------------
    # Save cache metadata
    # -----------------------------------------

    def save_metadata(
        self,
        pdf_path,
        item_count,
        parser=None,
        config_fingerprint=None
    ):

        cache_path = (
            self.get_cache_path(
                pdf_path
            )
        )

        cache_path.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata = {
            "cache_version": CACHE_VERSION,

            "pdf_hash": (
                self.get_file_hash(
                    pdf_path
                )
            ),

            "pdf_name": (
                Path(
                    pdf_path
                ).name
            ),

            "item_count": item_count,

            "parser": parser,

            "config_fingerprint": (
                config_fingerprint
            )
        }

        metadata_path = (
            cache_path
            / "metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

    # -----------------------------------------
    # Save processed items
    # -----------------------------------------

    def save_items(
        self,
        pdf_path,
        items
    ):

        cache_path = (
            self.get_cache_path(
                pdf_path
            )
        )

        cache_path.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            cache_path
            / "items.pkl",
            "wb"
        ) as f:

            pickle.dump(
                items,
                f
            )

    # -----------------------------------------
    # Load processed items
    # -----------------------------------------

    def load_items(
        self,
        pdf_path
    ):

        cache_path = (
            self.get_cache_path(
                pdf_path
            )
        )

        with open(
            cache_path
            / "items.pkl",
            "rb"
        ) as f:

            return pickle.load(
                f
            )

    # -----------------------------------------
    # Save SPLADE embeddings
    # -----------------------------------------

    def save_sparse_embeddings(
        self,
        pdf_path,
        embeddings
    ):

        cache_path = (
            self.get_cache_path(
                pdf_path
            )
        )

        cache_path.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            cache_path
            / "sparse_embeddings.pkl",
            "wb"
        ) as f:

            pickle.dump(
                embeddings,
                f
            )

    # -----------------------------------------
    # Load SPLADE embeddings
    # -----------------------------------------

    def load_sparse_embeddings(
        self,
        pdf_path
    ):

        cache_path = (
            self.get_cache_path(
                pdf_path
            )
        )

        with open(
            cache_path
            / "sparse_embeddings.pkl",
            "rb"
        ) as f:

            return pickle.load(
                f
            )
import hashlib
import json
import pickle
from pathlib import Path


CACHE_VERSION = "8.0"


def create_config_fingerprint(config):
    """Create a deterministic SHA-256 fingerprint for cache configuration."""
    config_json = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        config_json.encode("utf-8")
    ).hexdigest()


class DocumentCache:

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def get_file_hash(self, file_path):
        """Return the SHA-256 hash of a file."""
        sha256 = hashlib.sha256()

        with open(
            file_path,
            "rb",
        ) as file:
            while True:
                chunk = file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    def get_cache_path(self, pdf_path):
        """Return the cache directory for a PDF."""
        file_hash = self.get_file_hash(
            pdf_path
        )

        return self.cache_dir / file_hash

    def exists(
        self,
        pdf_path,
        parser=None,
        config_fingerprint=None,
    ):
        """Check whether a valid cache exists for a PDF."""
        cache_path = self.get_cache_path(
            pdf_path
        )

        required_files = [
            "metadata.json",
            "items.pkl",
            "faiss.index",
            "sparse_embeddings.pkl",
        ]

        if not all(
            (
                cache_path / filename
            ).exists()
            for filename in required_files
        ):
            return False

        metadata_path = (
            cache_path / "metadata.json"
        )

        try:
            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

        except (
            OSError,
            json.JSONDecodeError,
        ):
            return False

        if metadata.get(
            "cache_version"
        ) != CACHE_VERSION:
            return False

        current_pdf_hash = self.get_file_hash(
            pdf_path
        )

        if metadata.get(
            "pdf_hash"
        ) != current_pdf_hash:
            return False

        if (
            parser is not None
            and metadata.get("parser") != parser
        ):
            return False

        if (
            config_fingerprint is not None
            and metadata.get(
                "config_fingerprint"
            ) != config_fingerprint
        ):
            return False

        return True

    def save_metadata(
        self,
        pdf_path,
        item_count,
        parser=None,
        config_fingerprint=None,
    ):
        """Save metadata for a processed PDF."""
        cache_path = self.get_cache_path(
            pdf_path
        )

        cache_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = {
            "cache_version": CACHE_VERSION,
            "pdf_hash": self.get_file_hash(
                pdf_path
            ),
            "pdf_name": Path(pdf_path).name,
            "item_count": item_count,
            "parser": parser,
            "config_fingerprint": config_fingerprint,
        }

        metadata_path = (
            cache_path / "metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
            )

    def save_items(self, pdf_path, items):
        """Save processed document items."""
        cache_path = self.get_cache_path(
            pdf_path
        )

        cache_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            cache_path / "items.pkl",
            "wb",
        ) as file:
            pickle.dump(
                items,
                file,
            )

    def load_items(self, pdf_path):
        """Load processed document items."""
        cache_path = self.get_cache_path(
            pdf_path
        )

        with open(
            cache_path / "items.pkl",
            "rb",
        ) as file:
            return pickle.load(file)

    def load_sparse_embeddings(self, pdf_path):
        """Load cached SPLADE document embeddings."""
        cache_path = self.get_cache_path(
            pdf_path
        )

        with open(
            cache_path / "sparse_embeddings.pkl",
            "rb",
        ) as file:
            return pickle.load(file)

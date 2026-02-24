"""Hashing utilities for input deduplication and integrity."""
import hashlib
from pathlib import Path


def hash_bytes(data: bytes, algo: str = "sha256") -> str:
    """Return hex digest of raw bytes using the given hash algorithm."""
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def hash_string(text: str) -> str:
    """Return SHA-256 hex digest of a UTF-8 string."""
    return hash_bytes(text.encode("utf-8"))


def hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file, read in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_image(pil_image) -> str:
    """Hash a PIL Image by its raw bytes."""
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return hash_bytes(buf.getvalue())

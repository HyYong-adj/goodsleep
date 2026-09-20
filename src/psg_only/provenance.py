"""Small, streaming provenance helpers for local experiment artifacts."""
import hashlib
from pathlib import Path

from .paths import resolve


def file_hash(path):
    digest = hashlib.sha256()
    with resolve(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

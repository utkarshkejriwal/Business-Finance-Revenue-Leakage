"""File-based vector store: chunks + embeddings saved inside this folder.

    vector_store/chunks.json      chunk text and metadata
    vector_store/embeddings.npy   L2-normalised float32 matrix, one row per chunk
    vector_store/manifest.json    embedding model and hash of the source file

Run `python vector_store.py` to (re)build. `load_store()` rebuilds on its own
when the source markdown changes.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from chunking import build_chunks

BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR.parent / "data" / "contracts_and_rate_card.md"
STORE_DIR = BASE_DIR / "vector_store"
CHUNKS_PATH = STORE_DIR / "chunks.json"
EMBEDDINGS_PATH = STORE_DIR / "embeddings.npy"
MANIFEST_PATH = STORE_DIR / "manifest.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBEDDING_MODEL)
    return _embedder


def normalise(matrix):
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_store():
    chunks = build_chunks(SOURCE_PATH)
    vectors = np.array(list(get_embedder().embed([c["text"] for c in chunks])),
                       dtype=np.float32)
    vectors = normalise(vectors)

    STORE_DIR.mkdir(exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    np.save(EMBEDDINGS_PATH, vectors)
    MANIFEST_PATH.write_text(json.dumps({
        "source": SOURCE_PATH.name,
        "source_sha256": file_sha256(SOURCE_PATH),
        "embedding_model": EMBEDDING_MODEL,
        "dimensions": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    return chunks, vectors


def store_is_current():
    if not (CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists() and MANIFEST_PATH.exists()):
        return False
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return (manifest.get("source_sha256") == file_sha256(SOURCE_PATH)
            and manifest.get("embedding_model") == EMBEDDING_MODEL)


def load_store():
    if not store_is_current():
        return build_store()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    vectors = np.load(EMBEDDINGS_PATH)
    if len(chunks) != len(vectors):
        return build_store()
    return chunks, vectors


def embed_query(text):
    vector = np.array(next(iter(get_embedder().query_embed(text))), dtype=np.float32)
    return normalise(vector)


if __name__ == "__main__":
    built_chunks, built_vectors = build_store()
    print(f"Saved {len(built_chunks)} chunks, embeddings {built_vectors.shape} -> {STORE_DIR}")

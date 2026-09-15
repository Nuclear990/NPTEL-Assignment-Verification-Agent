import json
import shutil
from pathlib import Path

import faiss
import numpy as np


# ================================================================
# SAVE
# ================================================================

def save_store(
    chunks: list[dict],
    vectors: np.ndarray,
    store_dir: Path
) -> None:
    """
    Persist a course+week's chunks + their embeddings as a FAISS
    flat index + a row-aligned chunks.json.

    Pure persistence — chunks and vectors must already be computed;
    no model loading happens here. index.add(vectors) preserves
    input order, so FAISS row i <-> chunks[i] <-> vectors[i].

    Both files are built in a sibling temp directory and swapped in
    together, so index.faiss and chunks.json on disk are never a
    mismatched pair from two different builds.
    """

    if len(chunks) != vectors.shape[0]:

        raise ValueError(
            f"chunks/vectors length mismatch: "
            f"{len(chunks)} chunks vs {vectors.shape[0]} vectors"
        )

    tmp_dir = store_dir.with_name(store_dir.name + ".tmp")
    old_dir = store_dir.with_name(store_dir.name + ".old")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    tmp_dir.mkdir(
        parents=True
    )

    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(vectors)

    faiss.write_index(
        index,
        str(tmp_dir / "index.faiss")
    )

    (tmp_dir / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    shutil.rmtree(old_dir, ignore_errors=True)

    if store_dir.exists():

        store_dir.rename(old_dir)

    tmp_dir.rename(store_dir)

    shutil.rmtree(old_dir, ignore_errors=True)


# ================================================================
# LOAD
# ================================================================

def load_store(store_dir: Path) -> tuple[faiss.Index, list[dict]]:
    """
    Load a previously saved store. Raises FileNotFoundError if
    either file is missing (no store built yet for this
    course+week).
    """

    index_path = store_dir / "index.faiss"
    chunks_path = store_dir / "chunks.json"

    if not index_path.exists():

        raise FileNotFoundError(
            f"No FAISS index at {index_path}"
        )

    if not chunks_path.exists():

        raise FileNotFoundError(
            f"No chunks.json at {chunks_path}"
        )

    index = faiss.read_index(str(index_path))

    chunks = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    return index, chunks

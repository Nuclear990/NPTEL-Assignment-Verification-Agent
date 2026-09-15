import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL_NAME


# ================================================================
# MODEL (lazy singleton)
# ================================================================

_model = None


def _get_model() -> SentenceTransformer:

    global _model

    if _model is None:

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


# ================================================================
# EMBED
# ================================================================

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts into L2-normalized float32 vectors,
    shape (len(texts), dim).

    Normalized so a FAISS IndexFlatIP (inner product) search is
    equivalent to cosine similarity. Used identically here (index
    time, embedding chunks) and in rag/retrieval/retrieval.py
    (query time, embedding "question + option" strings) — both
    must share this exact embedding space.
    """

    if not texts:

        dim = _get_model().get_sentence_embedding_dimension()

        return np.empty((0, dim), dtype=np.float32)

    vectors = _get_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return vectors.astype(np.float32)

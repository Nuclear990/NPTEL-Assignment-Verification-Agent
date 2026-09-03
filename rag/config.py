from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "data" / "rag"


# ============================================================
# MODELS
# ============================================================

# Small, CPU-friendly semantic embedding model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lightweight cross-encoder reranker.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# CHUNKING
# ============================================================

TARGET_CHUNK_WORDS = 180
MAX_CHUNK_WORDS = 260
OVERLAP_WORDS = 45


# ============================================================
# RETRIEVAL
# ============================================================

DENSE_TOP_K = 15
BM25_TOP_K = 15
RERANK_TOP_K = 8
FINAL_EVIDENCE_K = 5


# Reciprocal Rank Fusion constant.
RRF_K = 60

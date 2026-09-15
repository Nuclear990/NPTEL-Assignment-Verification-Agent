from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "data" / "rag"

# Per-lecture chunks + the consolidated per-course-per-week FAISS/BM25
# store live here, separate from RAG_DIR (which holds only the
# LLM-cleaned transcript .txt files) — keeps the two concerns apart
# on disk instead of interleaving chunk/store files with cleaned text.
CHUNKS_DIR = BASE_DIR / "data" / "chunks"


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

DENSE_TOP_K = 5
BM25_TOP_K = 5
RERANK_TOP_K = 8

# Top chunks kept per option (post-RRF), before union-ing across a
# question's options. Reranking (RERANK_TOP_K/RERANKER_MODEL_NAME
# above) stays unused for now — no cross-encoder step yet.
TOP_K_PER_OPTION = 3


# Reciprocal Rank Fusion constant.
RRF_K = 60

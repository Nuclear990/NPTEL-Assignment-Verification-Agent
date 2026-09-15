import re
from dataclasses import dataclass

import faiss
from rank_bm25 import BM25Okapi

from rag.config import BM25_TOP_K, CHUNKS_DIR, DENSE_TOP_K, RRF_K, TOP_K_PER_OPTION
from rag.indexing.embedding import embed_texts
from rag.indexing.storage import load_store


# ================================================================
# HELPERS
# ================================================================

def _slugify(text: str) -> str:
    """
    Convert course name into filesystem-safe format.

    Kept identical to the _slugify/slugify already duplicated in
    rag/indexing/index.py, ingestion/transcripts.py and
    jobs.py — a fourth local copy, consistent with this
    codebase's existing convention.
    """

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


def _tokenize(text: str) -> list[str]:
    """
    Lightweight regex tokenizer for BM25 — no stemming/stopwords,
    matches this project's existing lightweight-regex style.
    """

    return re.findall(
        r"[a-z0-9]+",
        text.lower()
    )


def _build_bm25(chunks: list[dict]) -> BM25Okapi:

    return BM25Okapi(
        [_tokenize(chunk["text"]) for chunk in chunks]
    )


# ================================================================
# SESSION
# ================================================================

@dataclass
class RetrievalSession:
    """
    Holds everything needed to search one course+week, loaded once
    and reused across every option of every question — not reloaded
    per call.
    """

    course: str
    week: str
    faiss_index: faiss.Index
    chunks: list[dict]  # chunks[i] <-> faiss_index row i
    bm25: BM25Okapi


def load_retrieval_session(course: str, week) -> RetrievalSession:
    """
    Load the consolidated store for course+week and build a
    RetrievalSession from it. Raises FileNotFoundError if no store
    has been built yet for this course+week (see
    rag/indexing/index.py's receive_transcript_job).
    """

    store_dir = (
        CHUNKS_DIR
        / _slugify(course)
        / f"week_{week}"
        / "store"
    )

    faiss_index, chunks = load_store(store_dir)

    bm25 = _build_bm25(chunks)

    return RetrievalSession(
        course=course,
        week=week,
        faiss_index=faiss_index,
        chunks=chunks,
        bm25=bm25
    )


# ================================================================
# SEARCH
# ================================================================

def _dense_search(
    query_text: str,
    faiss_index: faiss.Index,
    top_k: int
) -> list[int]:
    """
    Dense (embedding) search. Returns row ids already ranked by
    FAISS, with the -1 padding it returns when fewer than top_k
    rows exist filtered out.
    """

    query_vector = embed_texts([query_text])

    _, ids = faiss_index.search(query_vector, top_k)

    return [
        int(row_id)
        for row_id in ids[0]
        if row_id != -1
    ]


def _bm25_search(
    query_text: str,
    bm25: BM25Okapi,
    top_k: int
) -> list[int]:
    """
    Sparse (BM25) search. Returns the top min(top_k, n_docs) row
    ids, ranked by score descending.
    """

    scores = bm25.get_scores(
        _tokenize(query_text)
    )

    ranked_ids = sorted(
        range(len(scores)),
        key=lambda row_id: scores[row_id],
        reverse=True
    )

    return ranked_ids[:min(top_k, len(scores))]


def _rrf_fuse(
    dense_ids: list[int],
    sparse_ids: list[int]
) -> list[int]:
    """
    Reciprocal Rank Fusion. Per-list ranks are 1-indexed; a row id's
    score is the sum of 1/(RRF_K+rank) over every list it appears
    in (0 if absent from a list). Ties are broken by stable sort —
    dense processed before sparse — deterministic, not otherwise
    specified.
    """

    scores = {}

    for rank, row_id in enumerate(dense_ids, start=1):

        scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (RRF_K + rank)

    for rank, row_id in enumerate(sparse_ids, start=1):

        scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (RRF_K + rank)

    ordered_ids = sorted(
        scores,
        key=lambda row_id: scores[row_id],
        reverse=True
    )

    return ordered_ids


def _top_chunks_for_query(
    query_text: str,
    session: RetrievalSession
) -> list[int]:

    dense_ids = _dense_search(
        query_text,
        session.faiss_index,
        DENSE_TOP_K
    )

    sparse_ids = _bm25_search(
        query_text,
        session.bm25,
        BM25_TOP_K
    )

    return _rrf_fuse(dense_ids, sparse_ids)[:TOP_K_PER_OPTION]


# ================================================================
# ENTRY POINT
# ================================================================

def retrieve_context_for_question(
    session: RetrievalSession,
    question: dict
) -> str:
    """
    Return a single combined string of the most relevant transcript
    evidence for one MCQ question with its options.

        question = {
            "question": "...",
            "options": [{"id": "...", "text": "..."}, ...],
            ...  # extra keys ignored
        }

    For each option, the question + option text is searched as one
    query; each option's top TOP_K_PER_OPTION chunks (post-RRF) are
    unioned and deduplicated by row id, then returned in source
    order. "" if there are no options or the store is empty.
    """

    options = question.get("options", [])

    if not options or not session.chunks:

        return ""

    matched_ids = set()

    for option in options:

        query_text = f"{question['question']} {option['text']}"

        matched_ids.update(
            _top_chunks_for_query(query_text, session)
        )

    ordered_ids = sorted(
        matched_ids,
        key=lambda row_id: (
            session.chunks[row_id]["source_file"],
            session.chunks[row_id]["chunk_index"]
        )
    )

    return "\n\n".join(
        session.chunks[row_id]["text"]
        for row_id in ordered_ids
    )

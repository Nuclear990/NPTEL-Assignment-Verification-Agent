import json
import re
from pathlib import Path

from groq import RateLimitError

from jobs import find_job, load_jobs, mark_transcript_cleaned
from rag.config import CHUNKS_DIR, RAG_DIR
from rag.indexing.chunking import chunk_cleaned_transcript
from rag.indexing.embedding import embed_texts
from rag.indexing.preprocessing import clean_transcript
from rag.indexing.storage import save_store


# ================================================================
# CROSS-JOB CLEANING BLOCKING
# ================================================================
#
# An LLM rate limit here is systemic for this run, the same way a
# YouTube block is for fetching (see ingestion/transcripts.py's
# is_fetch_blocked): once Groq starts rate-limiting, every further
# cleaning call in this run — this job's remaining lectures, and
# every job after it — would hit it again. Stop entirely instead of
# retrying into the same limit.
#

_cleaning_blocked = False


def is_cleaning_blocked() -> bool:

    return _cleaning_blocked


def _block_cleaning(reason: str) -> None:

    global _cleaning_blocked

    if not _cleaning_blocked:

        print(
            f"\n🚫 Blocking further LLM cleaning for the rest of "
            f"this run: {reason}"
        )

    _cleaning_blocked = True


# ================================================================
# HELPERS
# ================================================================

def _slugify(text: str) -> str:
    """
    Convert course name into filesystem-safe format.
    """

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


def _leading_number(stem: str) -> int:
    """
    Extract the leading lecture number from a transcript filename
    stem (e.g. "01_research_in_applied_mechanics" -> 1). Lecture
    files are always saved as f"{index:02d}_{slug}.txt" by
    ingestion/transcripts.py, so this recovers that same index.
    """

    return int(re.match(r"\d+", stem).group())


# ================================================================
# ENTRY POINT
# ================================================================

def receive_transcript_job(payload: dict) -> dict:
    """
    Entry point for the RAG ingestion pipeline.

    Expects a job payload:

        {
            "course": "...",
            "week": "...",
            "folder_path": ".../data/transcripts/<course>/week_<n>"
        }

    Runs preprocessing (LLM transcript cleaning), chunking, and
    embedding+storage on every lecture file found in the folder.
    Cleaned .txt files go under RAG_DIR; chunk .json files and the
    consolidated FAISS+BM25-ready store go under the separate
    CHUNKS_DIR (kept apart so chunk/store files never interleave
    with cleaned transcript text on disk).
    """

    course = payload["course"]
    week = payload["week"]
    folder_path = Path(payload["folder_path"])

    print("\n" + "=" * 80)
    print("RAG INGESTION: JOB RECEIVED")
    print("=" * 80)

    print(f"Course: {course}")
    print(f"Week: {week}")
    print(f"Folder: {folder_path}")

    lecture_files = sorted(
        folder_path.glob("*.txt")
    )

    if is_cleaning_blocked():

        print(
            "\nLLM cleaning is blocked for the rest of this run — "
            "skipping cleaning (chunking/embedding still run "
            "against whatever's already cleaned)."
        )

    if not lecture_files:

        print("No lecture transcript files found — nothing to preprocess.")

        return {
            "course": course,
            "week": week,
            "source_folder": str(folder_path),
            "preprocessed_files": [],
            "store_dir": None
        }

    out_dir = RAG_DIR / _slugify(course) / f"week_{week}"
    chunks_dir = CHUNKS_DIR / _slugify(course) / f"week_{week}"

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    persisted_job = find_job(load_jobs(), course, week)

    already_fetched = (
        set(persisted_job["transcriptsFetched"])
        if persisted_job
        else set()
    )

    already_cleaned = (
        set(persisted_job["transcriptsCleaned"])
        if persisted_job
        else set()
    )

    preprocessed_files = []
    chunk_files = []

    for lecture_file in lecture_files:

        video_number = _leading_number(lecture_file.stem)

        cleaned_text = None

        cleaned_path = out_dir / lecture_file.name

        # Trust "already cleaned" only if the cleaned file is still
        # actually there — jobs.json can say a lecture is cleaned
        # while the file itself was since deleted (or never really
        # written), in which case re-clean rather than crash later
        # when chunking tries to read it.
        if video_number in already_cleaned and cleaned_path.exists():

            print(f"\nAlready cleaned: {lecture_file.name} — skipping cleaning.")

        else:

            if video_number in already_cleaned:

                print(
                    f"\nMarked cleaned but file missing: "
                    f"{cleaned_path} — re-cleaning."
                )

            # A file can exist here without being a real transcript —
            # save_week_transcripts writes a "[TRANSCRIPT UNAVAILABLE]"
            # placeholder for a lecture whose fetch failed. Only clean
            # lectures jobs.json actually records as fetched, or a
            # failed fetch's placeholder gets "cleaned" into garbage
            # and wrongly marked done.
            if video_number not in already_fetched:

                print(
                    f"\nNot marked fetched yet: "
                    f"{lecture_file.name} — skipping."
                )

                continue

            if is_cleaning_blocked():

                print(
                    f"\nLLM cleaning blocked this run: "
                    f"{lecture_file.name} — skipping."
                )

                continue

            print(f"\nPreprocessing: {lecture_file.name}")

            raw_text = lecture_file.read_text(encoding="utf-8")

            try:

                cleaned_text = clean_transcript(raw_text)

            except RateLimitError as e:

                # Systemic for this run — stop cleaning entirely,
                # this job and every job after it (see
                # _block_cleaning above). `continue`, not `break`:
                # later lectures that are already cleaned still
                # need chunking.

                _block_cleaning(
                    f"{type(e).__name__}: {e}"
                )

                continue

            except Exception as e:

                print(
                    f"  Preprocessing failed: "
                    f"{type(e).__name__}: {e}"
                )

                continue

            cleaned_path.write_text(
                cleaned_text,
                encoding="utf-8"
            )

            preprocessed_files.append(str(cleaned_path))

            print(f"  Saved cleaned transcript -> {cleaned_path}")

            mark_transcript_cleaned(
                course,
                week,
                video_number
            )

        # ------------------------------------------------
        # Chunking — independent of whether cleaning ran
        # this call, so a lecture cleaned in an earlier run
        # (or one whose chunking previously failed) still
        # gets chunked here instead of being skipped forever
        # once "cleaned" is set.
        # ------------------------------------------------

        chunk_path = chunks_dir / f"{lecture_file.stem}.chunks.json"

        # Only trust an existing chunk file when cleaning was
        # actually skipped this call (cleaned_text is None) — if we
        # just re-cleaned this lecture above, any existing chunk
        # file is stale (e.g. built from previously-empty cleaned
        # content) and must be regenerated from the fresh text.
        if chunk_path.exists() and cleaned_text is None:

            print(f"  Already chunked: {chunk_path.name} — skipping.")

            continue

        if cleaned_text is None:

            cleaned_text = (out_dir / lecture_file.name).read_text(
                encoding="utf-8"
            )

        try:

            chunks = chunk_cleaned_transcript(cleaned_text)

        except Exception as e:

            print(
                f"  Chunking failed: "
                f"{type(e).__name__}: {e}"
            )

            continue

        chunk_path.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        chunk_files.append(str(chunk_path))

        print(f"  Saved {len(chunks)} chunk(s) -> {chunk_path}")

    print(
        f"\nPreprocessed {len(preprocessed_files)} new + "
        f"{len(already_cleaned)} already-cleaned out of "
        f"{len(lecture_files)} lecture(s)."
    )

    print(
        f"Chunked {len(chunk_files)}/"
        f"{len(lecture_files)} lecture(s)."
    )

    # ------------------------------------------------------------
    # Embedding + storage
    #
    # Rebuilt from every *.chunks.json currently on disk in
    # chunks_dir — not from chunk_files above, which only holds
    # what THIS call freshly chunked. The loop above skips lectures
    # already chunked in an earlier call, so chunk_files is usually
    # a small subset; the store must reflect the complete, current
    # chunk set for this course+week every time.
    # ------------------------------------------------------------

    store_dir = None

    try:

        all_chunks_for_store = []

        for chunks_file in sorted(chunks_dir.glob("*.chunks.json")):

            source_file = chunks_file.name.removesuffix(".chunks.json")

            lecture_chunks = json.loads(
                chunks_file.read_text(encoding="utf-8")
            )

            for chunk_index, chunk in enumerate(lecture_chunks):

                all_chunks_for_store.append(
                    {
                        **chunk,
                        "chunk_id": f"{source_file}::{chunk_index}",
                        "source_file": source_file,
                        "chunk_index": chunk_index
                    }
                )

        if all_chunks_for_store:

            texts = [
                chunk["text"]
                for chunk in all_chunks_for_store
            ]

            vectors = embed_texts(texts)

            candidate_store_dir = chunks_dir / "store"

            save_store(
                all_chunks_for_store,
                vectors,
                candidate_store_dir
            )

            store_dir = candidate_store_dir

            print(
                f"\nStore built: {len(all_chunks_for_store)} chunk(s) "
                f"-> {store_dir}"
            )

        else:

            print(
                "\nNo chunks available yet — nothing to embed/store."
            )

    except Exception as e:

        print(
            f"\nEmbedding/storage failed: "
            f"{type(e).__name__}: {e}"
        )

        store_dir = None

    return {
        "course": course,
        "week": week,
        "source_folder": str(folder_path),
        "preprocessed_folder": str(out_dir),
        "preprocessed_files": preprocessed_files,
        "chunks_folder": str(chunks_dir),
        "chunk_files": chunk_files,
        "store_dir": str(store_dir) if store_dir else None
    }

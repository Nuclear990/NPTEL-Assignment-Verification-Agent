import re
from pathlib import Path

from rag.config import RAG_DIR
from rag.indexing.preprocessing import clean_transcript


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

    Runs preprocessing (LLM transcript cleaning) on every lecture file
    found in the folder. Chunking and embedding are wired in later.
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

    if not lecture_files:

        print("No lecture transcript files found — nothing to preprocess.")

        return {
            "course": course,
            "week": week,
            "source_folder": str(folder_path),
            "preprocessed_files": []
        }

    out_dir = RAG_DIR / _slugify(course) / f"week_{week}"

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    preprocessed_files = []

    for lecture_file in lecture_files:

        print(f"\nPreprocessing: {lecture_file.name}")

        raw_text = lecture_file.read_text(encoding="utf-8")

        try:

            cleaned_text = clean_transcript(raw_text)

        except Exception as e:

            print(
                f"  Preprocessing failed: "
                f"{type(e).__name__}: {e}"
            )

            continue

        out_path = out_dir / lecture_file.name

        out_path.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        preprocessed_files.append(str(out_path))

        print(f"  Saved cleaned transcript -> {out_path}")

    print(
        f"\nPreprocessed {len(preprocessed_files)}/"
        f"{len(lecture_files)} lecture(s)."
    )

    return {
        "course": course,
        "week": week,
        "source_folder": str(folder_path),
        "preprocessed_folder": str(out_dir),
        "preprocessed_files": preprocessed_files
    }

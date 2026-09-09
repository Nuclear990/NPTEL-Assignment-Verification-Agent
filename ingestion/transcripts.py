import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

from config import DATA_DIR
from rag.pipeline import receive_transcript_job


# ============================================================
# CONFIG
# ============================================================

TRANSCRIPTS_DIR = DATA_DIR / "transcripts"


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_transcript(text: str) -> str:
    """
    Clean transcript text.
    """

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CREATE SAFE DIRECTORY NAME
# ============================================================

def slugify(text: str) -> str:
    """
    Convert course name into filesystem-safe format.

    Example:

    Governance of Artificial Intelligence
        ->
    governance_of_artificial_intelligence
    """

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    text = text.strip("_")

    return text


# ============================================================
# FETCH ONE VIDEO TRANSCRIPT
# ============================================================

def get_video_transcript(video_id: str) -> str:
    """
    Fetch transcript for one YouTube video.
    """

    print("\nFetching transcript...")
    print(f"Video ID: {video_id}")

    api = YouTubeTranscriptApi()

    fetched_transcript = api.fetch(
        video_id,
        languages=["en"]
    )

    transcript_text = " ".join(
        snippet.text
        for snippet in fetched_transcript
    )

    transcript_text = clean_transcript(
        transcript_text
    )

    print(
        "Transcript fetched successfully."
    )

    print(
        f"Characters: "
        f"{len(transcript_text)}"
    )

    return transcript_text


# ============================================================
# FETCH ALL LECTURE TRANSCRIPTS
# ============================================================

def fetch_week_transcripts(job: dict) -> list:
    """
    Fetch transcripts for every lecture in a job.

    Returns:

    [
        {
            "title": "...",
            "transcript": "..."
        }
    ]

    Failed lectures are preserved with an error.
    """

    lectures = job.get(
        "lectures",
        []
    )

    results = []

    total = len(lectures)

    print("\n" + "=" * 80)
    print("FETCHING WEEK TRANSCRIPTS")
    print("=" * 80)

    print(
        f"Course: {job.get('course')}"
    )

    print(
        f"Week: {job.get('week')}"
    )

    print(
        f"Lectures: {total}"
    )

    for index, lecture in enumerate(
        lectures,
        start=1
    ):

        title = lecture.get(
            "title",
            f"Lecture {index}"
        )

        print("\n" + "-" * 80)

        print(
            f"LECTURE {index}/{total}"
        )

        print(
            f"Title: {title}"
        )

        # --------------------------------------------
        # Skip lectures with no extracted video
        # --------------------------------------------

        video_id = lecture.get(
            "video_id"
        )

        if not video_id:

            print(
                "Skipping: no video ID."
            )

            results.append(
                {
                    "title": title,
                    "error": (
                        "No video_id available"
                    )
                }
            )

            continue

        # --------------------------------------------
        # Fetch transcript
        # --------------------------------------------

        try:

            transcript = get_video_transcript(
                video_id
            )

            results.append(
                {
                    "title": title,
                    "transcript": transcript
                }
            )

        except Exception as e:

            print("\nTRANSCRIPT FAILED")

            print(
                f"{type(e).__name__}: {e}"
            )

            # Don't kill entire week

            results.append(
                {
                    "title": title,
                    "error": str(e)
                }
            )

    return results


# ============================================================
# SAVE WEEK TRANSCRIPTS
# ============================================================

def save_week_transcripts(
    job: dict,
    lecture_transcripts: list
) -> Path:
    """
    Save each lecture transcript as its own file inside a week folder.

    File structure:

    data/
        transcripts/
            governance_of_artificial_intelligence/
                week_2/
                    01_introduction.txt
                    02_case_studies.txt
    """

    course = job.get(
        "course",
        "unknown_course"
    )

    week = job.get(
        "week",
        "unknown_week"
    )

    course_slug = slugify(course)

    week_dir = (
        TRANSCRIPTS_DIR
        / course_slug
        / f"week_{week}"
    )

    week_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 80)
    print("SAVING WEEK TRANSCRIPTS")
    print("=" * 80)

    print(
        f"Folder:\n{week_dir}"
    )

    for index, item in enumerate(
        lecture_transcripts,
        start=1
    ):

        title = item.get(
            "title",
            f"Lecture {index}"
        )

        lecture_slug = (
            slugify(title)
            or f"lecture_{index}"
        )

        lecture_path = (
            week_dir
            / f"{index:02d}_{lecture_slug}.txt"
        )

        if "transcript" in item:

            content = item["transcript"]

        else:

            content = (
                "[TRANSCRIPT UNAVAILABLE]\n\n"
                f"Error: "
                f"{item.get('error', 'Unknown error')}"
            )

        with open(
            lecture_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(content)

        print(
            f"  Saved: {lecture_path.name} "
            f"({len(content)} chars)"
        )

    print(
        "\nAll lecture transcripts saved."
    )

    return week_dir


# ============================================================
# HAND OFF TO RAG INGESTION
# ============================================================

def send_to_rag_ingestion(
    job: dict,
    transcript_folder: Path
) -> dict:
    """
    Build the RAG ingestion job payload and pass it to
    rag.pipeline.receive_transcript_job().
    """

    payload = {
        "course": job.get("course"),
        "week": job.get("week"),
        "folder_path": str(transcript_folder)
    }

    print("\n" + "=" * 80)
    print("HANDING OFF TO RAG INGESTION")
    print("=" * 80)

    print(
        f"Payload: {payload}"
    )

    return receive_transcript_job(payload)


# ============================================================
# ENRICH ONE JOB
# ============================================================

def enrich_job_with_transcript_file(
    job: dict
) -> dict:
    """
    Full transcript pipeline for one job.

    1. Fetch transcripts
    2. Save one file per lecture inside a week folder
    3. Hand the week off to the RAG ingestion pipeline
    4. Add transcript_folder path to job
    5. Remove temporary video/lecture data
    """

    print("\n" + "#" * 80)
    print("PROCESSING JOB TRANSCRIPTS")
    print("#" * 80)

    print(
        f"Course: {job.get('course')}"
    )

    print(
        f"Week: {job.get('week')}"
    )

    # --------------------------------------------------------
    # Fetch transcripts
    # --------------------------------------------------------

    lecture_transcripts = (
        fetch_week_transcripts(job)
    )

    # --------------------------------------------------------
    # Save one file per lecture
    # --------------------------------------------------------

    transcript_folder = save_week_transcripts(
        job,
        lecture_transcripts
    )

    # --------------------------------------------------------
    # Count success/failure
    # --------------------------------------------------------

    successful = sum(
        1
        for item in lecture_transcripts
        if "transcript" in item
    )

    failed = (
        len(lecture_transcripts)
        - successful
    )

    # --------------------------------------------------------
    # Enrich job with lightweight reference
    # --------------------------------------------------------

    job["transcript_folder"] = str(
        transcript_folder
    )

    job["transcript_summary"] = {
        "total_lectures": len(
            lecture_transcripts
        ),
        "transcripts_found": successful,
        "transcripts_failed": failed
    }

    # --------------------------------------------------------
    # Hand off to RAG ingestion pipeline
    # --------------------------------------------------------

    try:

        job["rag_ingestion"] = send_to_rag_ingestion(
            job,
            transcript_folder
        )

    except Exception as e:

        print("\nRAG INGESTION FAILED")

        print(
            f"{type(e).__name__}: {e}"
        )

        job["rag_ingestion_error"] = str(e)

    # --------------------------------------------------------
    # REMOVE HEAVY / TEMPORARY VIDEO DATA
    # --------------------------------------------------------

    # At this point we no longer need the
    # lecture → video mapping in the job.

    if "lectures" in job:

        del job["lectures"]

    if "video_extraction_summary" in job:

        del job[
            "video_extraction_summary"
        ]

    print("\n" + "=" * 80)
    print("JOB ENRICHED")
    print("=" * 80)

    print(
        f"Transcript folder: "
        f"{job['transcript_folder']}"
    )

    print(
        f"Transcripts found: "
        f"{successful}"
    )

    print(
        f"Transcripts failed: "
        f"{failed}"
    )

    return job


# ============================================================
# ENRICH ALL JOBS
# ============================================================

def enrich_jobs_with_transcript_files(
    jobs: list
) -> list:
    """
    Process every NPTEL job.

    Input jobs contain temporary lecture/video data.

    Output jobs contain only transcript file references.
    """

    print("\n" + "=" * 80)
    print("STARTING TRANSCRIPT FILE PIPELINE")
    print("=" * 80)

    print(
        f"Jobs to process: {len(jobs)}"
    )

    enriched_jobs = []

    for index, job in enumerate(
        jobs,
        start=1
    ):

        print("\n" + "#" * 80)

        print(
            f"JOB {index}/{len(jobs)}"
        )

        print("#" * 80)

        try:

            enriched_job = (
                enrich_job_with_transcript_file(
                    job
                )
            )

            enriched_jobs.append(
                enriched_job
            )

        except Exception as e:

            print("\n" + "!" * 80)
            print("JOB TRANSCRIPT PROCESSING FAILED")
            print("!" * 80)

            print(
                f"Course: "
                f"{job.get('course')}"
            )

            print(
                f"Week: "
                f"{job.get('week')}"
            )

            print(
                f"Error: "
                f"{type(e).__name__}: {e}"
            )

            job["transcript_error"] = str(e)

            enriched_jobs.append(
                job
            )

    return enriched_jobs

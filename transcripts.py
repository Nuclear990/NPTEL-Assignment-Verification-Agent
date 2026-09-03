import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

from config import DATA_DIR


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
# SAVE WEEK TRANSCRIPT
# ============================================================

def save_week_transcript(
    job: dict,
    lecture_transcripts: list
) -> Path:
    """
    Combine all lecture transcripts into one file.

    File structure:

    data/
        transcripts/
            governance_of_artificial_intelligence/
                week_2.txt
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

    course_dir = (
        TRANSCRIPTS_DIR
        / course_slug
    )

    course_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    transcript_path = (
        course_dir
        / f"week_{week}.txt"
    )

    print("\n" + "=" * 80)
    print("SAVING WEEK TRANSCRIPT")
    print("=" * 80)

    print(
        f"File:\n{transcript_path}"
    )

    # ========================================================
    # BUILD FILE CONTENT
    # ========================================================

    content_parts = []

    # --------------------------------------------------------
    # Metadata header
    # --------------------------------------------------------

    content_parts.append(
        "=" * 80
    )

    content_parts.append(
        f"COURSE: {course}"
    )

    content_parts.append(
        f"WEEK: {week}"
    )

    content_parts.append(
        "=" * 80
    )

    content_parts.append("")

    # --------------------------------------------------------
    # Lecture transcripts
    # --------------------------------------------------------

    for index, item in enumerate(
        lecture_transcripts,
        start=1
    ):

        title = item.get(
            "title",
            f"Lecture {index}"
        )

        content_parts.append(
            "=" * 80
        )

        content_parts.append(
            f"LECTURE {index}: {title}"
        )

        content_parts.append(
            "=" * 80
        )

        content_parts.append("")

        if "transcript" in item:

            content_parts.append(
                item["transcript"]
            )

        else:

            content_parts.append(
                "[TRANSCRIPT UNAVAILABLE]"
            )

            content_parts.append("")

            content_parts.append(
                f"Error: "
                f"{item.get('error', 'Unknown error')}"
            )

        content_parts.append("")
        content_parts.append("")

    # ========================================================
    # WRITE FILE
    # ========================================================

    final_content = "\n".join(
        content_parts
    )

    with open(
        transcript_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(final_content)

    print(
        "\nTranscript file saved successfully."
    )

    print(
        f"Characters written: "
        f"{len(final_content)}"
    )

    return transcript_path


# ============================================================
# ENRICH ONE JOB
# ============================================================

def enrich_job_with_transcript_file(
    job: dict
) -> dict:
    """
    Full transcript pipeline for one job.

    1. Fetch transcripts
    2. Combine into one week file
    3. Add transcript_file path to job
    4. Remove temporary video/lecture data
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
    # Save combined transcript
    # --------------------------------------------------------

    transcript_path = save_week_transcript(
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

    job["transcript_file"] = str(
        transcript_path
    )

    job["transcript_summary"] = {
        "total_lectures": len(
            lecture_transcripts
        ),
        "transcripts_found": successful,
        "transcripts_failed": failed
    }

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
        f"Transcript file: "
        f"{job['transcript_file']}"
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

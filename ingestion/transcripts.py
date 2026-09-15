import html
import re
from pathlib import Path

import yt_dlp

from config import DATA_DIR, YT_PROXY
from jobs import (
    find_job,
    load_jobs,
    mark_transcript_fetched,
    set_number_of_lectures
)
from rag.indexing.index import receive_transcript_job


# ============================================================
# CONFIG
# ============================================================

TRANSCRIPTS_DIR = DATA_DIR / "transcripts"


# ============================================================
# CROSS-JOB FETCH BLOCKING
# ============================================================
#
# A single transcript fetch failure (YouTube bot-check, IP block,
# rate limit — see docs/youtube-ip-block.md) is almost always
# systemic, not lecture- or job-specific: once it happens, every
# other fetch attempt in this run — this job's remaining lectures,
# and every job after it — would fail identically. This flag is
# set the first time that happens and checked before every further
# fetch attempt in the run, so we stop hammering YouTube with
# doomed requests instead of retrying per lecture/job.
#

_fetch_blocked = False


def is_fetch_blocked() -> bool:

    return _fetch_blocked


def _block_fetching(reason: str) -> None:

    global _fetch_blocked

    if not _fetch_blocked:

        print(
            f"\n🚫 Blocking further transcript fetching for the "
            f"rest of this run: {reason}"
        )

    _fetch_blocked = True


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
# PARSE VTT CAPTIONS
# ============================================================

def parse_vtt(vtt_text: str) -> str:
    """
    Extract plain spoken text from a WebVTT caption file,
    dropping cue numbers, timestamps and inline tags, and
    collapsing the duplicate lines YouTube's rolling
    auto-captions produce.
    """

    lines = []

    raw_lines = vtt_text.splitlines()

    for index, raw_line in enumerate(raw_lines):

        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue

        if "-->" in line:
            continue

        # A bare number is a cue identifier only when a timing line
        # follows it; otherwise it's spoken content (e.g. a year).
        if (
            line.isdigit()
            and index + 1 < len(raw_lines)
            and "-->" in raw_lines[index + 1]
        ):
            continue

        # Strip inline cue tags, e.g. <00:00:01.000><c> word</c>
        line = re.sub(r"<[^>]+>", "", line).strip()

        line = html.unescape(line)

        if not line:
            continue

        # Rolling auto-captions repeat the previous line
        if lines and lines[-1] == line:
            continue

        lines.append(line)

    return " ".join(lines)


# ============================================================
# FETCH ONE VIDEO TRANSCRIPT
# ============================================================

class CaptionsUnavailableError(RuntimeError):
    """A lecture-specific caption problem, not a YouTube block — other lectures are still worth fetching."""


def get_video_transcript(video_id: str) -> str:
    """
    Fetch transcript for one YouTube video via yt-dlp.
    """

    print("\nFetching transcript...")
    print(f"Video ID: {video_id}")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    if YT_PROXY:
        ydl_opts["proxy"] = YT_PROXY

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(
            f"https://www.youtube.com/watch?v={video_id}",
            download=False
        )

        tracks = (
            (info.get("subtitles") or {}).get("en")
            or (info.get("automatic_captions") or {}).get("en")
        )

        if not tracks:
            raise CaptionsUnavailableError(
                f"No English captions available for video {video_id}"
            )

        track = next(
            (t for t in tracks if t.get("ext") == "vtt"),
            tracks[0]
        )

        # Fetch through yt-dlp's own HTTP client so the proxy set
        # above is attached to this request too, not just the
        # metadata lookup above.
        vtt_bytes = ydl.urlopen(track["url"]).read()

    vtt_text = vtt_bytes.decode("utf-8", errors="ignore")

    transcript_text = parse_vtt(vtt_text)

    transcript_text = clean_transcript(
        transcript_text
    )

    if not transcript_text:
        raise CaptionsUnavailableError(
            f"Fetched captions for video {video_id} were empty"
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

    Failed lectures are preserved with an error. Lectures already
    marked fetched in jobs.json are skipped entirely (no repeat
    YouTube download) and come back tagged "skip": True. If a prior
    fetch in this run already tripped is_fetch_blocked(), every
    lecture here comes back the same way, untried.
    """

    lectures = job.get(
        "lectures",
        []
    )

    total = len(lectures)

    if is_fetch_blocked():

        print(
            "\nFetching is blocked for the rest of this run — "
            "skipping this job's lectures entirely."
        )

        return [
            {
                "title": lecture.get("title", f"Lecture {index}"),
                "skip": True,
                "reason": "fetching blocked this run"
            }
            for index, lecture in enumerate(lectures, start=1)
        ]

    results = []

    persisted_job = find_job(
        load_jobs(),
        job.get("course"),
        job.get("week")
    )

    already_fetched = (
        set(persisted_job["transcriptsFetched"])
        if persisted_job
        else set()
    )

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
        # Skip lectures already fetched in a prior run
        # --------------------------------------------

        if index in already_fetched:

            print(
                "Already fetched — skipping."
            )

            results.append(
                {
                    "title": title,
                    "skip": True
                }
            )

            continue

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

        except CaptionsUnavailableError as e:

            print("\nTRANSCRIPT UNAVAILABLE — continuing with next lecture")

            print(
                f"{type(e).__name__}: {e}"
            )

            results.append(
                {
                    "title": title,
                    "error": str(e)
                }
            )

        except Exception as e:

            print("\nTRANSCRIPT FAILED")

            print(
                f"{type(e).__name__}: {e}"
            )

            results.append(
                {
                    "title": title,
                    "error": str(e)
                }
            )

            # A fetch failure here (YouTube blocking/rate-limiting,
            # bot-check, IP block — see docs/youtube-ip-block.md) is
            # almost always systemic, not lecture-specific: every
            # remaining lecture in this run — this job's, and every
            # job after it — would fail the same way. Stop entirely
            # rather than hammering YouTube with doomed requests;
            # everything unfetched gets retried next run.

            _block_fetching(
                f"{type(e).__name__}: {e}"
            )

            break

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

    # The true total, even if fetching stopped early this run after
    # a failure (see fetch_week_transcripts) and lecture_transcripts
    # is only a prefix — job["lectures"] itself is never trimmed.
    set_number_of_lectures(
        course,
        week,
        len(job.get("lectures", []))
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

        if item.get("skip"):

            print(
                f"  Skipping "
                f"({item.get('reason', 'already fetched')}): {title}"
            )

            continue

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

        if "transcript" in item:

            mark_transcript_fetched(
                course,
                week,
                index
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
    rag.indexing.index.receive_transcript_job().
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
# CLEAN WHATEVER'S ALREADY FETCHED (NO NEW FETCHING)
# ============================================================

def clean_week_transcripts(course: str, week) -> dict:
    """
    Run the RAG cleaning/chunking step against whatever raw
    transcript files already exist on disk for course + week,
    without fetching anything new. receive_transcript_job() skips
    lecture numbers already marked cleaned in jobs.json, so this is
    safe to call whenever a job has nothing left to fetch (or can't
    be fetched this run) but may still have fetched-but-uncleaned
    lectures from an interrupted prior run.
    """

    transcript_folder = (
        TRANSCRIPTS_DIR
        / slugify(course)
        / f"week_{week}"
    )

    return send_to_rag_ingestion(
        {"course": course, "week": week},
        transcript_folder
    )


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
        if "transcript" in item or item.get("skip")
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

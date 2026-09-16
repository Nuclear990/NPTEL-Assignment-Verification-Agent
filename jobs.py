import json
import re

from config import DATA_DIR


# ============================================================
# CONFIG
# ============================================================

JOBS_FILE = DATA_DIR / "jobs.json"

# Small words kept lowercase when reconstructing a human-readable
# course name from a folder slug (see slug_to_title).
_TITLE_STOPWORDS = {
    "of", "the", "a", "an", "in", "on", "for", "and", "to", "with"
}


# ============================================================
# HELPERS
# ============================================================

def _slugify(text: str) -> str:
    """
    Convert course name into filesystem-safe format.

    Kept identical to the _slugify/slugify already duplicated in
    rag/indexing/index.py, ingestion/transcripts.py and
    ingestion/assignment.py — used here only to compare course
    names that may differ in case/punctuation.
    """

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


def slug_to_title(slug: str) -> str:
    """
    Best-effort human-readable reconstruction of a course slug, e.g.
    "governance_of_artificial_intelligence" ->
    "Governance of Artificial Intelligence".

    Only needed when backfilling jobs from on-disk folders, which are
    named with slugs — the original course title as it appeared in the
    NPTEL email is never saved to disk anywhere else.
    """

    words = slug.split("_")

    titled = []

    for index, word in enumerate(words):

        if index > 0 and word in _TITLE_STOPWORDS:

            titled.append(word)

        else:

            titled.append(word.capitalize())

    return " ".join(titled)


# ============================================================
# LOAD / SAVE
# ============================================================

def load_jobs() -> list:
    """
    Load all persisted jobs. Returns an empty list if jobs.json
    doesn't exist yet.
    """

    if not JOBS_FILE.exists():

        return []

    return json.loads(
        JOBS_FILE.read_text(encoding="utf-8")
    )


def save_jobs(jobs: list) -> None:
    """
    Persist the full jobs list, overwriting jobs.json atomically:
    written to a temp file first, then renamed over jobs.json, so a
    crash mid-write never leaves a truncated, unparseable jobs.json.
    """

    tmp_file = JOBS_FILE.with_name(JOBS_FILE.name + ".tmp")

    tmp_file.write_text(
        json.dumps(jobs, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    tmp_file.replace(JOBS_FILE)


# ============================================================
# LOOKUP
# ============================================================

def find_job(jobs: list, course: str, week) -> dict | None:
    """
    Find a job by course + week. Matching is done on slugified
    course name and stringified week, so callers don't need to
    worry about case/formatting differences.
    """

    course_slug = _slugify(course)
    week = str(week)

    for job in jobs:

        if (
            _slugify(job.get("course", "")) == course_slug
            and str(job.get("week")) == week
        ):

            return job

    return None


def get_or_create_job(jobs: list, course: str, week) -> dict:
    """
    Return the existing job for course + week, creating (and
    appending to `jobs`) a fresh one if none exists yet.
    """

    job = find_job(jobs, course, week)

    if job is not None:

        return job

    job = {
        "course": course,
        "week": week,
        "completed": False,
        "numberOfLectures": 0,
        "transcriptsFetched": [],
        "transcriptsCleaned": [],
        "contentUrl": None,
        "assignmentUrl": None,
        "emailId": None
    }

    jobs.append(job)

    return job


# ============================================================
# MUTATORS
# ============================================================
#
# Each mutator loads, updates, and immediately saves the full
# jobs list. At this project's scale (a handful of jobs, run
# sequentially, never in parallel) this keeps jobs.json always
# consistent on disk without needing any locking.
#

def ensure_job(
    course: str,
    week,
    content_url: str = None,
    assignment_url: str = None,
    email_id: str = None
) -> dict:
    """
    Make sure a job record exists for course + week and persist it.
    Safe to call as soon as a job is known (e.g. right after email
    parsing), before any transcripts have been fetched.

    A brand new job is created with the URLs/email id given. For a
    job that already exists, progress fields (completed,
    numberOfLectures, transcriptsFetched, transcriptsCleaned) are
    never touched — only missing contentUrl/assignmentUrl/emailId
    are filled in, so a job stays resumable even in a run whose
    email scan no longer surfaces its original announcement email.
    """

    jobs = load_jobs()

    job = get_or_create_job(jobs, course, week)

    if content_url and not job.get("contentUrl"):
        job["contentUrl"] = content_url

    if assignment_url and not job.get("assignmentUrl"):
        job["assignmentUrl"] = assignment_url

    if email_id and not job.get("emailId"):
        job["emailId"] = email_id

    save_jobs(jobs)

    return job


def set_number_of_lectures(course: str, week, count: int) -> None:
    """
    Record how many lectures exist for course + week.
    """

    jobs = load_jobs()

    job = get_or_create_job(jobs, course, week)

    job["numberOfLectures"] = count

    save_jobs(jobs)


def mark_transcript_fetched(course: str, week, video_number: int) -> None:
    """
    Record that the raw transcript for lecture `video_number` of
    course + week has been fetched.
    """

    jobs = load_jobs()

    job = get_or_create_job(jobs, course, week)

    if video_number not in job["transcriptsFetched"]:

        job["transcriptsFetched"].append(video_number)
        job["transcriptsFetched"].sort()

    save_jobs(jobs)


def mark_transcript_cleaned(course: str, week, video_number: int) -> None:
    """
    Record that the transcript for lecture `video_number` of
    course + week has been LLM-cleaned.
    """

    jobs = load_jobs()

    job = get_or_create_job(jobs, course, week)

    if video_number not in job["transcriptsCleaned"]:

        job["transcriptsCleaned"].append(video_number)
        job["transcriptsCleaned"].sort()

    save_jobs(jobs)


def mark_completed(course: str, week, completed: bool = True) -> None:
    """
    Set the completed flag for course + week. main.py sets it once
    processing/solver.py has answered every question.
    """

    jobs = load_jobs()

    job = get_or_create_job(jobs, course, week)

    job["completed"] = completed

    save_jobs(jobs)

import json

from playwright.sync_api import sync_playwright

from config import BROWSER_PROFILE_DIR, HEADLESS
from ingestion.gmail_watcher import get_recent_nptel_emails
from ingestion.email_parser import parse_nptel_email
from ingestion.videos import get_page, enrich_job_with_videos
from ingestion.transcripts import (
    clean_week_transcripts,
    fetch_week_transcripts,
    is_fetch_blocked,
    save_week_transcripts
)
from ingestion.assignment import ensure_assignment_scraped
from jobs import ensure_job, find_job, load_jobs, mark_completed
from auth.setup_login import NotLoggedInError, setup_login
from processing.classifier import ensure_assignment_classified
from processing.solver import solve_assignment

# ================================================================
# CONFIG
# ================================================================

MAX_EMAIL_RESULTS = 15


# ================================================================
# CROSS-JOB BROWSER BLOCKING
# ================================================================
#
# NPTEL browser steps (video/transcript scrape, assignment scrape)
# only run while the login is positively verified. If setup_login
# can't verify it, or a scrape lands on NPTEL's logged-out view
# mid-run, every remaining browser step this run is skipped — a
# logged-out session only ever sees the public preview page, so
# each attempt would fail (or persist wrong lecture counts).
#

_browser_blocked = False


def is_browser_blocked() -> bool:

    return _browser_blocked


def _block_browser(reason: str) -> None:

    global _browser_blocked

    if not _browser_blocked:

        print(
            f"\n🚫 Skipping NPTEL browser steps for the rest of "
            f"this run: {reason}"
        )

    _browser_blocked = True


# ================================================================
# STEP 1
# FETCH EMAILS
# ================================================================

def fetch_emails():

    print("\n" + "=" * 80)
    print("STEP 1: FETCHING NPTEL EMAILS")
    print("=" * 80)

    emails = get_recent_nptel_emails(
        max_results=MAX_EMAIL_RESULTS
    )

    print(
        f"\nEmails fetched: {len(emails)}"
    )

    return emails


# ================================================================
# STEP 2
# PARSE EMAILS -> REGISTER NEW JOBS ONLY
# ================================================================

def register_new_jobs(emails):
    """
    Parse every fetched email and persist a jobs.json record for
    it. Jobs already known (same course + week) are left exactly
    as they are — this only adds jobs, it never resets progress.
    """

    print("\n" + "=" * 80)
    print("STEP 2: PARSING NPTEL EMAILS")
    print("=" * 80)

    for index, email in enumerate(
        emails,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"EMAIL {index}/{len(emails)}"
        )

        print(
            f"Subject: "
            f"{email['subject']}"
        )

        try:

            job = parse_nptel_email(
                email["body"]
            )

        except Exception as e:

            print(
                f"\n✗ SKIPPED: "
                f"{type(e).__name__}: {e}"
            )

            continue

        ensure_job(
            job["course"],
            job["week"],
            content_url=job.get("content_url"),
            assignment_url=job.get("assignment_url"),
            email_id=email["id"]
        )

        print("\n✓ PARSED")

        print(
            f"Course: {job['course']}"
        )

        print(
            f"Week: {job['week']}"
        )


# ================================================================
# STEP 3
# PROCESS ONE JOB END TO END
#
# fetch remaining transcripts -> clean remaining transcripts
#   -> scrape + classify assignment
#   -> if fully cleaned: solve, then mark completed
# ================================================================

def process_job(job_record):

    course = job_record["course"]
    week = job_record["week"]

    print("\n" + "#" * 80)

    print(
        f"JOB: {course} — Week {week}"
    )

    print("#" * 80)

    if job_record.get("completed"):

        print(
            "\nAlready completed — skipping."
        )

        return

    # --------------------------------------------------------
    # Step A: fetch remaining transcripts (best effort).
    #
    # Own persistent browser context, opened and closed just
    # for this step — same one-context-per-operation pattern
    # as auth/setup_login.py and ingestion/assignment.py's
    # launch_browser(), so this never overlaps with the
    # separate context the assignment step below opens on the
    # same browser profile.
    #
    # Deliberately does NOT clean — cleaning always happens in
    # step B below, once, regardless of whether fetching ran,
    # was skipped, or failed partway.
    # --------------------------------------------------------

    content_url = job_record.get("contentUrl")

    already_fully_fetched = (
        job_record["numberOfLectures"] > 0
        and len(job_record["transcriptsFetched"])
            >= job_record["numberOfLectures"]
    )

    if already_fully_fetched:

        print(
            "\nAll lectures already fetched for this week — "
            "nothing remaining, skipping the video/transcript "
            "scrape entirely."
        )

    elif not content_url:

        print(
            "\nNo content URL on record for this job — "
            "cannot fetch transcripts until a fresh email "
            "supplies one. Skipping the scrape."
        )

    elif is_browser_blocked():

        print(
            "\nNPTEL login not verified this run — "
            "skipping the scrape."
        )

    elif is_fetch_blocked():

        print(
            "\nTranscript fetching is blocked for the rest of "
            "this run (a prior job hit a systemic failure) — "
            "skipping the scrape."
        )

    else:

        try:

            with sync_playwright() as p:

                context = (
                    p.chromium.launch_persistent_context(
                        user_data_dir=str(
                            BROWSER_PROFILE_DIR
                        ),
                        headless=HEADLESS,
                        viewport={
                            "width": 1400,
                            "height": 900
                        }
                    )
                )

                try:

                    page = get_page(context)

                    working_job = {
                        "course": course,
                        "week": week,
                        "content_url": content_url
                    }

                    enrich_job_with_videos(
                        page,
                        working_job
                    )

                    lecture_transcripts = fetch_week_transcripts(
                        working_job
                    )

                    save_week_transcripts(
                        working_job,
                        lecture_transcripts
                    )

                finally:

                    context.close()

        except NotLoggedInError as e:

            _block_browser(str(e))

        except Exception as e:

            print("\n" + "!" * 80)
            print("TRANSCRIPT STEP FAILED")
            print("!" * 80)

            print(
                f"{type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # Step B: clean whatever's fetched but not cleaned yet —
    # always attempted, whatever step A did or didn't do, so a
    # lecture fetched in a previous (or this) run never sits
    # uncleaned just because this run's fetch was skipped/failed.
    # --------------------------------------------------------

    try:

        clean_week_transcripts(
            course,
            week
        )

    except Exception as e:

        print("\n" + "!" * 80)
        print("CLEANING STEP FAILED")
        print("!" * 80)

        print(
            f"{type(e).__name__}: {e}"
        )

    # Reload persisted state — the steps above have been
    # updating jobs.json as they went.

    job_record = find_job(
        load_jobs(),
        course,
        week
    )

    # --------------------------------------------------------
    # Assignment (scrape once, skipped if already on disk)
    # --------------------------------------------------------

    if is_browser_blocked():

        print(
            "\nNPTEL login not verified this run — "
            "skipping the assignment scrape."
        )

    else:

        try:

            ensure_assignment_scraped(
                course,
                week,
                job_record.get("assignmentUrl")
            )

        except NotLoggedInError as e:

            _block_browser(str(e))

        except Exception as e:

            print("\n" + "!" * 80)
            print("ASSIGNMENT STEP FAILED")
            print("!" * 80)

            print(
                f"{type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # Classification — once questions.json exists. Needs no
    # browser, so it runs even when login isn't verified; skipped
    # when every question already has a solving strategy.
    # --------------------------------------------------------

    try:

        ensure_assignment_classified(
            course,
            week
        )

    except Exception as e:

        print("\n" + "!" * 80)
        print("CLASSIFICATION STEP FAILED")
        print("!" * 80)

        print(
            f"{type(e).__name__}: {e}"
        )

    # --------------------------------------------------------
    # Solving — gated on every lecture for this week being
    # cleaned. Embedding + storage already happened above, as
    # part of the transcript-cleaning hand-off (see
    # rag/indexing/index.py's receive_transcript_job) — main.py
    # doesn't need to know that exists, only whether cleaning is
    # complete enough to solve from.
    # --------------------------------------------------------

    fully_cleaned = (
        job_record["numberOfLectures"] > 0
        and len(job_record["transcriptsCleaned"])
            >= job_record["numberOfLectures"]
    )

    if not fully_cleaned:

        print(
            "\nTranscripts not fully cleaned yet for this "
            "week — skipping solving for now."
        )

        return

    print(
        "\nAll transcripts cleaned for this week — ready to solve."
    )

    try:

        solved = solve_assignment(
            course,
            week
        )

    except Exception as e:

        print("\n" + "!" * 80)
        print("SOLVING STEP FAILED")
        print("!" * 80)

        print(
            f"{type(e).__name__}: {e}"
        )

        return

    if solved:

        mark_completed(
            course,
            week
        )

        print(
            "\n✓ Every question answered — job marked completed."
        )

    else:

        print(
            "\nSome questions are still unanswered — "
            "they'll be retried next run."
        )


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():

    print("\n")
    print("#" * 80)
    print("NPTEL COURSEWORK AGENT")
    print("#" * 80)

    # ------------------------------------------------------------
    # Gmail -> register any new jobs
    # ------------------------------------------------------------

    try:

        emails = fetch_emails()

    except Exception as e:

        print(
            f"\n✗ EMAIL FETCH FAILED: "
            f"{type(e).__name__}: {e}"
        )

        emails = []

    if emails:

        register_new_jobs(emails)

    else:

        print(
            "\nNo NPTEL announcement emails found this run — "
            "continuing with existing tracked jobs."
        )

    jobs = load_jobs()

    if not jobs:

        print(
            "\nNo jobs to process."
        )

        return

    # ------------------------------------------------------------
    # Process every tracked job
    # ------------------------------------------------------------

    # Browser steps need a positively verified login; cleaning and
    # indexing of already-fetched transcripts don't, so the run
    # continues either way.
    try:

        setup_login()

    except Exception as e:

        _block_browser(
            f"NPTEL login not verified: "
            f"{type(e).__name__}: {e}"
        )

    for job_record in jobs:

        try:

            process_job(
                job_record
            )

        except Exception as e:

            print("\n" + "!" * 80)
            print("JOB FAILED")
            print("!" * 80)

            print(
                f"Course: {job_record.get('course')}"
            )

            print(
                f"Week: {job_record.get('week')}"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

    # ------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    print("\nTRACKED JOBS:")

    print(
        json.dumps(
            load_jobs(),
            indent=4,
            ensure_ascii=False
        )
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()

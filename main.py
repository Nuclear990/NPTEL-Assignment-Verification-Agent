import json

from ingestion.gmail_watcher import get_recent_nptel_emails
from ingestion.email_parser import parse_nptel_email
from ingestion.videos import enrich_jobs_with_videos
from ingestion.transcripts import enrich_jobs_with_transcript_files
from ingestion.assignment import enrich_jobs_with_assignments
from auth.setup_login import setup_login
#from solver import solve_jobs

# ================================================================
# CONFIG
# ================================================================

MAX_EMAIL_RESULTS = 15


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
# PARSE EMAILS
# ================================================================

def parse_emails(emails):

    print("\n" + "=" * 80)
    print("STEP 2: PARSING NPTEL EMAILS")
    print("=" * 80)

    jobs = []

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

            # Preserve source Gmail ID
            job["email_id"] = email["id"]

            jobs.append(job)

            print("\n✓ PARSED")

            print(
                f"Course: {job['course']}"
            )

            print(
                f"Week: {job['week']}"
            )

        except Exception as e:

            print(
                f"\n✗ SKIPPED: "
                f"{type(e).__name__}: {e}"
            )

    print(
        f"\nValid jobs created: {len(jobs)}"
    )

    return jobs


# ================================================================
# STEP 3
# EXTRACT NPTEL LECTURES + VIDEOS
# ================================================================

def process_course_content(jobs):

    print("\n" + "=" * 80)
    print("STEP 3: EXTRACTING COURSE VIDEOS")
    print("=" * 80)

    return enrich_jobs_with_videos(
        jobs
    )


# ================================================================
# STEP 4
# FETCH + SAVE TRANSCRIPTS
# ================================================================

def process_transcripts(jobs):

    print("\n" + "=" * 80)
    print("STEP 4: FETCHING TRANSCRIPTS")
    print("=" * 80)

    return enrich_jobs_with_transcript_files(
        jobs
    )


# ================================================================
# STEP 5
# EXTRACT + SAVE ASSIGNMENTS
# ================================================================

def process_assignments(jobs):

    print("\n" + "=" * 80)
    print("STEP 5: PROCESSING ASSIGNMENTS")
    print("=" * 80)

    return enrich_jobs_with_assignments(
        jobs
    )
    
# ================================================================
# STEP 6
# SOLVE ASSIGNMENTS
# ================================================================

def process_solutions(jobs):

    print("\n" + "=" * 80)
    print("STEP 6: SOLVING ASSIGNMENTS")
    print("=" * 80)

    return solve_jobs(
        jobs
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
    # Gmail
    # ------------------------------------------------------------

    emails = fetch_emails()

    if not emails:

        print(
            "\nNo NPTEL announcement emails found."
        )

        return

    # ------------------------------------------------------------
    # Email parsing
    # ------------------------------------------------------------

    jobs = parse_emails(
        emails
    )

    if not jobs:

        print(
            "\nNo valid jobs could be created."
        )

        return
        
    setup_login()
    # ------------------------------------------------------------
    # NPTEL content extraction
    # ------------------------------------------------------------

    jobs = process_course_content(
        jobs
    )

    # ------------------------------------------------------------
    # Transcript extraction
    # ------------------------------------------------------------

    jobs = process_transcripts(
        jobs
    )

    # ------------------------------------------------------------
    # Assignment processing
    # ------------------------------------------------------------

    jobs = process_assignments(
    	jobs
    )
	# ------------------------------------------------------------
	# Solve assignments
	# ------------------------------------------------------------

    '''jobs = process_solutions(
	 jobs
    )'''

    # ------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    print("\nFINAL JOBS:")

    print(
        json.dumps(
            jobs,
            indent=4,
            ensure_ascii=False
        )
    )



# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()

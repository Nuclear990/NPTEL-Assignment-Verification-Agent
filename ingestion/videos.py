import re

from playwright.sync_api import sync_playwright

from auth.setup_login import NotLoggedInError, assert_logged_in
from config import BROWSER_PROFILE_DIR, HEADLESS


# ============================================================
# CONFIG
# ============================================================

VIDEO_LOAD_RETRIES = 3
VIDEO_LOAD_WAIT_MS = 5000


# ============================================================
# INTERNAL: GET PAGE
# ============================================================

def get_page(context):

    if context.pages:
        return context.pages[0]

    return context.new_page()


# ============================================================
# EXTRACT WEEK LECTURE TITLES
# ============================================================

def get_week_lecture_titles(page, week):

    print("\n" + "=" * 80)
    print(f"EXTRACTING WEEK {week} CONTENT")
    print("=" * 80)

    week_pattern = re.compile(
        rf"^\s*Week {week}\b",
        re.IGNORECASE
    )

    week_buttons = (
        page
        .locator("button")
        .filter(has_text=week_pattern)
    )

    if week_buttons.count() == 0:
        raise RuntimeError(
            f"Could not find Week {week} button"
        )

    week_button = week_buttons.first

    print("\nFound week:")
    print(week_button.inner_text().strip())

    # --------------------------------------------------------
    # Expand week
    # --------------------------------------------------------

    expanded = week_button.get_attribute(
        "aria-expanded"
    )

    print(f"\naria-expanded: {expanded}")

    if expanded == "false":

        print("Week is collapsed. Expanding...")

        week_button.click()

        page.wait_for_timeout(1000)

    else:

        print("Week is already expanded.")

    # --------------------------------------------------------
    # Get container
    # --------------------------------------------------------

    container_id = week_button.get_attribute(
        "aria-controls"
    )

    if not container_id:
        raise RuntimeError(
            f"Week {week} has no aria-controls"
        )

    print("\nWeek container ID:")
    print(container_id)

    week_container = page.locator(
        f"#{container_id}"
    )

    if week_container.count() == 0:
        raise RuntimeError(
            f"Could not find #{container_id}"
        )

    lesson_buttons = week_container.locator(
        "button"
    )

    button_count = lesson_buttons.count()

    print(
        f"\nButtons found inside Week {week}: "
        f"{button_count}"
    )

    lecture_titles = []

    print("\n" + "-" * 80)
    print("ANALYZING WEEK ITEMS")
    print("-" * 80)

    for i in range(button_count):

        button = lesson_buttons.nth(i)

        try:
            text = button.inner_text().strip()

        except Exception as e:

            print(
                f"[{i}] Could not read button: {e}"
            )

            continue

        if not text:
            continue

        print(f"\n[{i}] {text}")

        # Feedback forms are video-less lesson pages that some courses
        # list before the quiz, so the assignment stop in
        # enrich_job_with_videos can't catch them. Week items carry no
        # type marker in the DOM, hence the (deliberately narrow) title
        # match; everything else is kept.
        if re.search(
            r"^\s*Week\s*\d+\s*Feedback Form",
            text,
            re.IGNORECASE
        ):

            print("    -> SKIPPED (feedback form)")

            continue

        lecture_titles.append(text)

    if not lecture_titles:

        raise RuntimeError(
            f"No content items found for Week {week}"
        )

    return lecture_titles


# ============================================================
# GET LECTURE URL
# ============================================================

def get_lecture_url(page, week, lecture_title):

    week_pattern = re.compile(
        rf"^\s*Week {week}\b",
        re.IGNORECASE
    )

    week_button = (
        page
        .locator("button")
        .filter(has_text=week_pattern)
        .first
    )

    if week_button.count() == 0:

        raise RuntimeError(
            f"Could not re-find Week {week}"
        )

    # Ensure expanded
    expanded = week_button.get_attribute(
        "aria-expanded"
    )

    if expanded == "false":

        week_button.click()

        page.wait_for_timeout(700)

    # Get container again
    container_id = week_button.get_attribute(
        "aria-controls"
    )

    if not container_id:

        raise RuntimeError(
            "Week container ID missing"
        )

    week_container = page.locator(
        f"#{container_id}"
    )

    # Exact lecture match.
    # NPTEL's own markup occasionally has stray leading/trailing
    # whitespace inside the title element (e.g. "Ocean Engineering "),
    # which survives into has_text's raw text content even though
    # inner_text() elsewhere normalizes it away. Tolerate it here.
    lecture_pattern = re.compile(
        rf"^\s*{re.escape(lecture_title)}\s*$"
    )

    lecture_buttons = (
        week_container
        .locator("button")
        .filter(has_text=lecture_pattern)
    )

    # The lecture list can render progressively after the week
    # is expanded, so actively wait for the button to appear
    # instead of checking count() once after a fixed sleep.
    try:

        lecture_buttons.first.wait_for(
            state="visible",
            timeout=8000
        )

    except Exception:

        raise RuntimeError(
            f"Could not find lecture: "
            f"{lecture_title}"
        )

    lecture_button = lecture_buttons.first

    old_url = page.url

    lecture_button.click()

    try:

        page.wait_for_function(
            """
            oldUrl => window.location.href !== oldUrl
            """,
            arg=old_url,
            timeout=5000
        )

    except Exception:
        # Happens if already on this lecture
        pass

    page.wait_for_timeout(1500)

    return page.url


# ============================================================
# EXTRACT YOUTUBE VIDEO
# ============================================================

def extract_youtube_video(page, lecture_url):

    print("\nOpening lecture page:")
    print(lecture_url)

    # --------------------------------------------------------
    # Retry because NPTEL sometimes fails to render
    # the iframe on the first SPA load.
    # --------------------------------------------------------

    for attempt in range(
        1,
        VIDEO_LOAD_RETRIES + 1
    ):

        print(
            f"\nVideo extraction attempt "
            f"{attempt}/{VIDEO_LOAD_RETRIES}"
        )

        try:

            page.goto(
                lecture_url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            # Wait longer than before
            page.wait_for_timeout(
                VIDEO_LOAD_WAIT_MS
            )

            youtube_iframes = page.locator(
                'iframe[src*="youtube.com/embed"], '
                'iframe[src*="youtube-nocookie.com/embed"]'
            )

            iframe_count = youtube_iframes.count()

            print(
                f"YouTube iframes found: "
                f"{iframe_count}"
            )

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            if iframe_count > 0:

                iframe = youtube_iframes.first

                embed_url = iframe.get_attribute(
                    "src"
                )

                if not embed_url:

                    raise RuntimeError(
                        "YouTube iframe has no src"
                    )

                print("\nYouTube embed URL:")
                print(embed_url)

                match = re.search(
                    r"youtube(?:-nocookie)?\.com/embed/([^?&#/]+)",
                    embed_url
                )

                if not match:

                    raise RuntimeError(
                        "Could not extract video ID"
                    )

                video_id = match.group(1)

                youtube_url = (
                    "https://www.youtube.com/"
                    f"watch?v={video_id}"
                )

                print(f"\nVideo ID: {video_id}")

                return {
                    "video_id": video_id,
                    "youtube_url": youtube_url
                }

            # ------------------------------------------------
            # Debug info
            # ------------------------------------------------

            total_iframes = page.locator(
                "iframe"
            ).count()

            print(
                f"Total iframes found: "
                f"{total_iframes}"
            )

            # ------------------------------------------------
            # Retry
            # ------------------------------------------------

            if attempt < VIDEO_LOAD_RETRIES:

                print(
                    "Video iframe not loaded. "
                    "Retrying..."
                )

                page.wait_for_timeout(2000)

        except Exception as e:

            print(
                f"Attempt {attempt} failed: "
                f"{type(e).__name__}: {e}"
            )

            if attempt < VIDEO_LOAD_RETRIES:

                page.wait_for_timeout(2000)

            else:
                raise

    # --------------------------------------------------------
    # All retries failed
    # --------------------------------------------------------

    raise RuntimeError(
        f"No YouTube iframe found after "
        f"{VIDEO_LOAD_RETRIES} attempts"
    )


# ============================================================
# PROCESS ONE NPTEL JOB
# ============================================================

def enrich_job_with_videos(page, job):

    course = job["course"]
    week = job["week"]
    content_url = job["content_url"]

    print("\n" + "=" * 80)
    print("PROCESSING NPTEL JOB")
    print("=" * 80)

    print(f"\nCourse: {course}")
    print(f"Week: {week}")

    print("\nContent URL:")
    print(content_url)

    # --------------------------------------------------------
    # Open course
    # --------------------------------------------------------

    print("\nOpening course page...")

    page.goto(
        content_url,
        wait_until="domcontentloaded",
        timeout=60000
    )

    page.wait_for_timeout(3000)

    assert_logged_in(page)

    # --------------------------------------------------------
    # Get lecture titles
    # --------------------------------------------------------

    lecture_titles = get_week_lecture_titles(
        page,
        week
    )

    print("\nLectures found:")

    for title in lecture_titles:

        print(f"- {title}")

    enriched_lectures = []

    total = len(lecture_titles)

    # ========================================================
    # PROCESS EACH LECTURE INDEPENDENTLY
    #
    # IMPORTANT:
    # One broken lecture must NOT kill the whole course.
    # ========================================================

    for index, title in enumerate(
        lecture_titles,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"PROCESSING LECTURE "
            f"{index}/{total}"
        )

        print(f"Title: {title}")

        try:

            # --------------------------------------------
            # IMPORTANT:
            # We may currently be on a previous lecture.
            # Open original course URL again before finding
            # the next lecture.
            # --------------------------------------------

            page.goto(
                content_url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            page.wait_for_timeout(2000)

            assert_logged_in(page)

            # --------------------------------------------
            # Get lecture URL
            # --------------------------------------------

            lecture_url = get_lecture_url(
                page,
                week,
                title
            )

            print("\nLecture URL:")
            print(lecture_url)

            # The assignment always follows a week's lectures, so the
            # first item that opens it ends the lecture list.
            if "assessmentId=" in lecture_url:

                print(
                    "\nReached the week's assignment — "
                    "no more lectures."
                )

                break

            # --------------------------------------------
            # Extract video
            # --------------------------------------------

            video_data = extract_youtube_video(
                page,
                lecture_url
            )

            enriched_lectures.append(
                {
                    "title": title,
                    "lecture_url": lecture_url,
                    "video_id": (
                        video_data["video_id"]
                    ),
                    "youtube_url": (
                        video_data["youtube_url"]
                    )
                }
            )

        except NotLoggedInError:

            # Not a lecture failure — every remaining lecture would
            # hit the same logged-out page; abort without recording.
            raise

        except Exception as e:

            # --------------------------------------------
            # DO NOT KILL ENTIRE JOB
            # --------------------------------------------

            print("\n" + "!" * 80)

            print(
                f"LECTURE FAILED: {title}"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            print(
                "Continuing with remaining lectures..."
            )

            print("!" * 80)

            # Preserve the lecture in output so we know
            # exactly what failed.

            enriched_lectures.append(
                {
                    "title": title,
                    "video_error": str(e)
                }
            )

    # --------------------------------------------------------
    # Add lectures even if some failed
    # --------------------------------------------------------

    job["lectures"] = enriched_lectures

    successful = sum(
        1
        for lecture in enriched_lectures
        if "youtube_url" in lecture
    )

    failed = len(enriched_lectures) - successful

    job["video_extraction_summary"] = {
        "total_lectures": len(enriched_lectures),
        "videos_found": successful,
        "videos_failed": failed
    }

    return job


# ============================================================
# PUBLIC API
# ============================================================

def enrich_jobs_with_videos(jobs):

    print("\n" + "=" * 80)
    print("STARTING NPTEL VIDEO ENRICHMENT")
    print("=" * 80)

    print(f"\nJobs to process: {len(jobs)}")

    enriched_jobs = []

    with sync_playwright() as p:

        print(
            "\nStarting persistent Chromium..."
        )

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

        page = get_page(context)

        try:

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
                        enrich_job_with_videos(
                            page,
                            job
                        )
                    )

                    enriched_jobs.append(
                        enriched_job
                    )

                except NotLoggedInError:

                    raise

                except Exception as e:

                    # This is now ONLY for failures at
                    # course/job level.

                    print("\n" + "=" * 80)
                    print("JOB FAILED")
                    print("=" * 80)

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

                    job["nptel_error"] = str(e)

                    enriched_jobs.append(job)

        finally:

            context.close()

            print(
                "\nPersistent browser closed."
            )

    return enriched_jobs


# ============================================================
# OPTIONAL DIRECT TEST
# ============================================================

if __name__ == "__main__":

    import json

    test_jobs = [
        {
            "course": (
                "Governance of Artificial Intelligence"
            ),
            "week": 2,
            "content_url": (
                "https://onlinecourses.nptel.ac.in/"
                "e-learning/course/noc26_lw24"
                "?unitId=21&lessonId=47"
            ),
            "assignment_url": "TEST",
            "email_id": "TEST"
        }
    ]

    results = enrich_jobs_with_videos(
        test_jobs
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        json.dumps(
            results,
            indent=4,
            ensure_ascii=False
        )
    )

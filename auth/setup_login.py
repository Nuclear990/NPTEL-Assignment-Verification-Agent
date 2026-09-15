import re
import time

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from config import BROWSER_PROFILE_DIR
from jobs import load_jobs


NPTEL_URL = (
    "https://onlinecourses.nptel.ac.in/"
)

# How long to wait for positive proof of a logged-in course page.
VERIFY_TIMEOUT_S = 30

# How long a "Sign In" button may stay visible on a scraped page before
# it counts as logged out (it can flash while the SPA hydrates).
SIGN_IN_GRACE_S = 3

# Logged-out visitors are bounced to the public course preview
# (/e-learning/preview/<course>) or a login / Google sign-in page. An
# expired session does NOT redirect /my/, so that page proves nothing.
LOGGED_OUT_URL_MARKERS = (
    "/preview/",
    "login",
    "signin",
    "accounts.google.com"
)

WEEK_BUTTON_PATTERN = re.compile(r"^\s*Week\s*\d+", re.IGNORECASE)
SIGN_IN_PATTERN = re.compile(r"^\s*sign\s*in\s*$", re.IGNORECASE)


class NotLoggedInError(RuntimeError):
    """The persistent NPTEL browser session is not authenticated."""


# ================================================================
# CHECKS ON AN ALREADY-OPEN PAGE
# ================================================================

def _has_logged_out_url(page) -> bool:

    url = page.url.lower()

    return any(
        marker in url
        for marker in LOGGED_OUT_URL_MARKERS
    )


def _sign_in_visible(page) -> bool:

    return (
        page.locator("button, a")
        .filter(has_text=SIGN_IN_PATTERN)
        .filter(visible=True)
        .count()
        > 0
    )


def assert_logged_in(page) -> None:
    """
    Raise NotLoggedInError if the page NPTEL just served is the
    logged-out view. Call right after navigating to any course,
    lecture or assignment URL, before scraping it.
    """

    if _has_logged_out_url(page):

        raise NotLoggedInError(
            f"NPTEL session is not logged in "
            f"(redirected to {page.url})"
        )

    deadline = time.monotonic() + SIGN_IN_GRACE_S

    while _sign_in_visible(page):

        if time.monotonic() >= deadline:

            raise NotLoggedInError(
                f"NPTEL shows a Sign In button on {page.url}"
            )

        page.wait_for_timeout(250)


def wait_for_login_state(page) -> bool:
    """
    Fail-closed verdict for a page just sent to a course content URL.

    True only on positive proof of an authenticated session: still on
    /e-learning/course/, the week accordion has rendered, and no Sign
    In button is visible. A logged-out redirect, or no proof within
    VERIFY_TIMEOUT_S, is False.
    """

    deadline = time.monotonic() + VERIFY_TIMEOUT_S

    while time.monotonic() < deadline:

        try:

            if _has_logged_out_url(page):
                return False

            if (
                "/e-learning/course/" in page.url
                and page.locator("button")
                .filter(has_text=WEEK_BUTTON_PATTERN)
                .count() > 0
                and not _sign_in_visible(page)
            ):
                return True

        except PlaywrightError:

            # Mid-navigation (e.g. a client-side redirect) — undecided.
            pass

        page.wait_for_timeout(250)

    return False


# ================================================================
# SESSION CHECK
# ================================================================

def default_check_url() -> str | None:
    """A tracked course content URL to verify the login against."""

    return next(
        (
            job["contentUrl"]
            for job in load_jobs()
            if job.get("contentUrl")
        ),
        None
    )


def is_logged_in(check_url: str) -> bool:
    """
    Open check_url (a course content URL) in the persistent profile
    and return wait_for_login_state's fail-closed verdict. Navigation
    errors propagate — an unreachable NPTEL proves nothing either way.
    """

    with sync_playwright() as p:

        context = p.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_PROFILE_DIR),
            headless=True,
            viewport={
                "width": 1400,
                "height": 900
            }
        )

        try:

            page = context.pages[0] if context.pages else context.new_page()

            page.goto(
                check_url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            return wait_for_login_state(page)

        finally:

            context.close()


def setup_login(check_url: str = None):
    """
    Make sure the persistent profile is logged into NPTEL, falling
    back to a manual login. Raises NotLoggedInError unless the login
    is positively verified — no browser step may run in that case.
    """

    check_url = check_url or default_check_url()

    if not check_url:

        raise NotLoggedInError(
            "No course content URL in jobs.json to verify "
            "the NPTEL login against."
        )

    if is_logged_in(check_url):

        print("NPTEL login verified - skipping manual login.")
        return

    print("NPTEL session is not logged in (or not enrolled in this course).")

    print("Starting persistent browser...")

    with sync_playwright() as p:

        # This is the important part.
        # Everything you do in this browser is saved in:
        # browser_profile/
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_PROFILE_DIR),
            headless=False,
            viewport={
                "width": 1400,
                "height": 900
            }
        )

        page = context.pages[0]

        print("Opening NPTEL...")

        page.goto(
            check_url,
            wait_until="domcontentloaded"
        )

        print("\n" + "=" * 60)
        print("MANUAL LOGIN REQUIRED")
        print("=" * 60)

        print("""
1. Log into NPTEL
2. Click Google login
3. Select your college Google account
4. Complete login
5. Make sure you can access your course
6. Return here
7. Press ENTER to save the session
        """)

        input("Press ENTER after successful login...")

        print("Saving browser session...")

        context.close()

    print("Verifying login...")

    if not is_logged_in(check_url):

        raise NotLoggedInError(
            "NPTEL login could not be verified after manual login — "
            f"{check_url} still shows the logged-out view."
        )

    print("Done.")
    print(
        f"Persistent browser profile saved at:\n"
        f"{BROWSER_PROFILE_DIR}"
    )


if __name__ == "__main__":
    setup_login()

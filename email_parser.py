import re

from bs4 import BeautifulSoup


# ================================================================
# CLEAN EMAIL HTML
# ================================================================

def clean_email_html(html: str) -> str:

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    for tag in soup(
        ["script", "style", "img"]
    ):
        tag.decompose()

    text = soup.get_text("\n")

    text = re.sub(
        r"\n\s*\n+",
        "\n",
        text
    )

    return text.strip()


# ================================================================
# PARSE NPTEL EMAIL
# ================================================================

def parse_nptel_email(email_text: str) -> dict:

    text = clean_email_html(
        email_text
    )

    # ------------------------------------------------------------
    # COURSE
    # ------------------------------------------------------------

    course_match = re.search(
        r'course\s+["“]([^"”]+)["”]',
        text,
        re.IGNORECASE
    )

    course = (
        course_match.group(1).strip()
        if course_match
        else None
    )

    # ------------------------------------------------------------
    # WEEK
    # ------------------------------------------------------------

    week_match = re.search(
        r'\bWeek\s*-?\s*(\d+)\b',
        text,
        re.IGNORECASE
    )

    week = (
        int(week_match.group(1))
        if week_match
        else None
    )

    # ------------------------------------------------------------
    # URLs
    # ------------------------------------------------------------

    urls = re.findall(
        r'https://onlinecourses\.nptel\.ac\.in/'
        r'e-learning/course/[^\s]+',
        text
    )

    urls = list(
        dict.fromkeys(urls)
    )

    content_url = None
    assignment_url = None

    for url in urls:

        if "lessonId=" in url:
            content_url = url

        elif "assessmentId=" in url:
            assignment_url = url

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    if not content_url:
        raise ValueError(
            "Could not find content URL"
        )

    if not assignment_url:
        raise ValueError(
            "Could not find assignment URL"
        )

    if week is None:
        raise ValueError(
            "Could not find week number"
        )

    if course is None:
        raise ValueError(
            "Could not find course name"
        )

    return {
        "course": course,
        "week": week,
        "content_url": content_url,
        "assignment_url": assignment_url
    }

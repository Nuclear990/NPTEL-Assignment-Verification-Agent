import os
import re
import json
import time

from urllib.parse import urlparse, parse_qs

from playwright.sync_api import sync_playwright

from config import BROWSER_PROFILE_DIR


# ================================================================
# CONFIGURATION
# ================================================================

ASSIGNMENTS_DIR = "data/assignments"

PAGE_LOAD_TIMEOUT = 30000
QUESTION_WAIT_TIMEOUT = 30000


# ================================================================
# DIRECTORY HELPERS
# ================================================================

def slugify(text: str):
    """
    Convert course name into filesystem-safe format.

    Example:

        Research Methodology

            ->

        research_methodology
    """

    if not text:
        return "unknown"

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


def build_assignment_file_path(
    course: str,
    week: int
):
    """
    Build assignment path.

    Example:

    data/assignments/
        research_methodology/
            week_6.json
    """

    course_slug = slugify(
        course
    )

    course_dir = os.path.join(
        ASSIGNMENTS_DIR,
        course_slug
    )

    os.makedirs(
        course_dir,
        exist_ok=True
    )

    filename = (
        f"week_{week}.json"
    )

    return os.path.join(
        course_dir,
        filename
    )


# ================================================================
# URL HELPERS
# ================================================================

def get_assessment_id(
    url: str
):
    """
    Extract assessmentId from NPTEL assignment URL.
    """

    parsed = urlparse(
        url
    )

    query = parse_qs(
        parsed.query
    )

    assessment_ids = query.get(
        "assessmentId",
        []
    )

    if not assessment_ids:
        return None

    return assessment_ids[0]


# ================================================================
# BROWSER
# ================================================================

def launch_browser(
    playwright
):
    """
    Launch persistent Chromium.

    Uses the same browser profile created
    during NPTEL login setup.
    """

    context = (
        playwright.chromium
        .launch_persistent_context(
            user_data_dir=str(
                BROWSER_PROFILE_DIR
            ),
            headless=False,
            viewport={
                "width": 1400,
                "height": 900
            }
        )
    )

    page = context.pages[0]

    return context, page


# ================================================================
# EXTRACT QUESTIONS
# ================================================================

def extract_questions_from_page(
    page
):
    """
    Extract questions from rendered NPTEL assignment page.

    Question types:

        radio
            ->
        single_correct

        checkbox
            ->
        multiple_correct


    Returns:

    [
        {
            "id": "q1",
            "type": "single_correct",
            "question": "...",
            "options": [
                {
                    "id": "q1_o1",
                    "text": "..."
                }
            ]
        }
    ]
    """

    print(
        "\nWaiting for assignment questions..."
    )

    page.wait_for_selector(
        "main.practice-questions",
        timeout=QUESTION_WAIT_TIMEOUT
    )

    time.sleep(2)

    print(
        "Assignment questions container found."
    )

    questions = page.evaluate(
        """
        () => {

            // ====================================================
            // CLEAN TEXT
            // ====================================================

            function cleanText(text) {

                if (!text) {
                    return "";
                }

                return text
                    .replace(/\\s+/g, " ")
                    .trim();

            }


            // ====================================================
            // FIND ASSIGNMENT CONTAINER
            // ====================================================

            const container =
                document.querySelector(
                    "main.practice-questions"
                );


            if (!container) {
                return [];
            }


            // ====================================================
            // QUESTION SECTIONS
            //
            // Use sections directly.
            // ====================================================

            const questionSections =
                Array.from(
                    container.querySelectorAll(
                        "section"
                    )
                );


            // ====================================================
            // EXTRACT QUESTIONS
            // ====================================================

            return questionSections.map(
                (
                    section,
                    questionIndex
                ) => {


                    const questionId =
                        `q${questionIndex + 1}`;


                    // ------------------------------------------------
                    // QUESTION TEXT
                    // ------------------------------------------------

                    const questionElement =
                        section.querySelector(
                            '[class*="question-content"]'
                        );


                    let questionText = "";


                    if (questionElement) {

                        questionText =
                            cleanText(
                                questionElement.innerText
                            );

                    }


                    // Remove visible question number.
                    //
                    // Example:
                    //
                    // "1. What is research?"
                    //
                    // becomes:
                    //
                    // "What is research?"

                    questionText =
                        questionText.replace(
                            /^\\s*\\d+\\s*[.\\)]\\s*/,
                            ""
                        );


                    // ------------------------------------------------
                    // QUESTION TYPE
                    // ------------------------------------------------
                    //
                    // Circle / radio:
                    //     single_correct
                    //
                    // Square / checkbox:
                    //     multiple_correct
                    // ------------------------------------------------

                    const firstInput =
                        section.querySelector(
                            'input[type="radio"], input[type="checkbox"]'
                        );


                    let questionType =
                        "unknown";


                    if (firstInput) {

                        if (
                            firstInput.type === "radio"
                        ) {

                            questionType =
                                "single_correct";

                        }

                        else if (
                            firstInput.type === "checkbox"
                        ) {

                            questionType =
                                "multiple_correct";

                        }

                    }


                    // ------------------------------------------------
                    // OPTIONS
                    // ------------------------------------------------

                    const labels =
                        Array.from(
                            section.querySelectorAll(
                                "label"
                            )
                        );


                    const options = [];


                    labels.forEach(
                        label => {

                            const input =
                                label.querySelector(
                                    'input[type="radio"], input[type="checkbox"]'
                                );


                            if (!input) {
                                return;
                            }


                            // Clone label and remove input
                            // so only option text remains.

                            const clone =
                                label.cloneNode(
                                    true
                                );


                            clone
                                .querySelectorAll(
                                    "input"
                                )
                                .forEach(
                                    element => {

                                        element.remove();

                                    }
                                );


                            const optionText =
                                cleanText(
                                    clone.innerText
                                );


                            if (!optionText) {
                                return;
                            }


                            options.push({

                                id:
                                    `${questionId}_o${options.length + 1}`,

                                text:
                                    optionText

                            });

                        }
                    );


                    return {

                        id:
                            questionId,

                        type:
                            questionType,

                        question:
                            questionText,

                        options:
                            options

                    };

                }
            );

        }
        """
    )

    return questions


# ================================================================
# SAVE QUESTIONS
# ================================================================

def save_questions_to_file(
    questions,
    course,
    week
):
    """
    Save ONLY assignment questions.

    Structure:

    data/assignments/
        course_name/
            week_X.json
    """

    file_path = (
        build_assignment_file_path(
            course=course,
            week=week
        )
    )

    payload = {
        "questions": questions
    }

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            separators=(",", ":")
        )

    print(
        "\nQuestions saved:"
    )

    print(
        file_path
    )

    return file_path


# ================================================================
# EXTRACT ASSIGNMENT
# ================================================================

def extract_assignment(
    assignment_url
):
    """
    Open NPTEL assignment page and extract questions.

    Does NOT:

    - click answers
    - modify answers
    - submit assignment
    - save HTML

    Returns:

    {
        "assignment_id": "...",
        "questions": [...]
    }
    """

    if not assignment_url:

        raise ValueError(
            "assignment_url is required"
        )

    assessment_id = (
        get_assessment_id(
            assignment_url
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "EXTRACTING NPTEL ASSIGNMENT"
    )

    print(
        "=" * 80
    )

    print(
        f"\nAssignment ID: "
        f"{assessment_id}"
    )

    with sync_playwright() as playwright:

        context, page = (
            launch_browser(
                playwright
            )
        )

        try:

            print(
                "\nOpening assignment page..."
            )

            page.goto(
                assignment_url,
                wait_until="domcontentloaded",
                timeout=PAGE_LOAD_TIMEOUT
            )

            time.sleep(3)

            print(
                "\nCurrent URL:"
            )

            print(
                page.url
            )

            # ----------------------------------------------------
            # AUTHENTICATION CHECK
            # ----------------------------------------------------

            if (
                "login" in page.url.lower()
                or
                "signin" in page.url.lower()
            ):

                raise RuntimeError(
                    "NPTEL authentication required."
                )


            # ----------------------------------------------------
            # EXTRACT QUESTIONS
            # ----------------------------------------------------

            questions = (
                extract_questions_from_page(
                    page
                )
            )

            if not questions:

                raise RuntimeError(
                    "No assignment questions found."
                )

            print(
                f"\nTotal questions extracted: "
                f"{len(questions)}"
            )

            return {
                "assignment_id":
                    assessment_id,

                "questions":
                    questions
            }

        finally:

            context.close()


# ================================================================
# ENRICH JOBS WITH ASSIGNMENTS
# ================================================================

def enrich_jobs_with_assignments(
    jobs
):
    """
    Pipeline stage:

    jobs
        ↓
    extract assignment questions
        ↓
    save questions to file
        ↓
    add assignment_file to job


    Input:

    {
        "course": "...",
        "week": 6,
        "assignment_url": "..."
    }


    Output:

    {
        "course": "...",
        "week": 6,
        "assignment_url": "...",
        "assignment_file":
            "data/assignments/course_name/week_6.json"
    }
    """

    if not jobs:

        return []


    enriched_jobs = []


    print(
        "\n" + "=" * 80
    )

    print(
        "EXTRACTING ASSIGNMENTS"
    )

    print(
        "=" * 80
    )


    for index, job in enumerate(
        jobs,
        start=1
    ):

        print(
            "\n" + "#" * 80
        )

        print(
            f"PROCESSING ASSIGNMENT "
            f"{index}/{len(jobs)}"
        )

        print(
            "#" * 80
        )


        assignment_url = (
            job.get(
                "assignment_url"
            )
        )


        if not assignment_url:

            raise ValueError(
                "Job has no assignment_url"
            )


        # --------------------------------------------------------
        # EXTRACT QUESTIONS
        # --------------------------------------------------------

        assignment = (
            extract_assignment(
                assignment_url
            )
        )


        # --------------------------------------------------------
        # SAVE QUESTIONS
        # --------------------------------------------------------

        assignment_file = (
            save_questions_to_file(
                questions=
                    assignment["questions"],

                course=
                    job["course"],

                week=
                    job["week"]
            )
        )


        # --------------------------------------------------------
        # CREATE ENRICHED JOB
        # --------------------------------------------------------

        enriched_job = (
            job.copy()
        )


        enriched_job[
            "assignment_file"
        ] = assignment_file


        enriched_jobs.append(
            enriched_job
        )


        print(
            "\nAssignment processing complete."
        )


    return enriched_jobs


# ================================================================
# TEST CONFIG
# ================================================================

TEST_JOB = {
    "course": "Research Methodology",
    "week": 1,
    "assignment_url": (
        "https://onlinecourses.nptel.ac.in/"
        "e-learning/course/noc26_ge79"
        "?unitId=16&assessmentId=211"
    )
}


# ================================================================
# TEST MAIN
# ================================================================

def main():

    print("\n" + "#" * 80)

    print(
        "TESTING ASSIGNMENT EXTRACTION"
    )

    print(
        "#" * 80
    )


    print("\nCourse:")

    print(
        TEST_JOB["course"]
    )


    print("\nWeek:")

    print(
        TEST_JOB["week"]
    )


    print("\nAssignment URL:")

    print(
        TEST_JOB["assignment_url"]
    )


    # ------------------------------------------------------------
    # Run complete assignment pipeline
    #
    # Job
    #   ↓
    # Extract questions
    #   ↓
    # Save questions to file
    #   ↓
    # Add assignment_file to job
    # ------------------------------------------------------------

    enriched_jobs = (
        enrich_jobs_with_assignments(
            [TEST_JOB]
        )
    )


    print("\n")

    print(
        "=" * 80
    )

    print(
        "ASSIGNMENT PROCESSING COMPLETE"
    )

    print(
        "=" * 80
    )


    print(
        "\nFinal enriched job(s):"
    )

    print(
        json.dumps(
            enriched_jobs,
            indent=4,
            ensure_ascii=False
        )
    )


    print("\n" + "=" * 80)

    print(
        "FILES CREATED"
    )

    print(
        "=" * 80
    )


    for job in enriched_jobs:

        assignment_file = job.get(
            "assignment_file"
        )


        if assignment_file:

            print(
                f"\nAssignment file:\n"
                f"{assignment_file}"
            )


            if os.path.exists(
                assignment_file
            ):

                print(
                    "Status: EXISTS"
                )

                print(
                    f"Size: "
                    f"{os.path.getsize(assignment_file)} bytes"
                )

            else:

                print(
                    "Status: NOT FOUND"
                )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()

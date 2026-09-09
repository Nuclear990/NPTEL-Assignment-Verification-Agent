import os
import re
import json
import time
import requests

from ingestion.img_to_txt import image_to_text

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
    """Convert course name into filesystem-safe format."""

    if not text:
        return "unknown"

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_assignment_directory(course: str, week: int):
    """
    Build and create the directory for one assignment.

    Example:

    data/assignments/
        research_methodology/
            week_1/
                questions.json
                p1.txt
                p2.txt
    """

    course_slug = slugify(course)

    assignment_dir = os.path.join(
        ASSIGNMENTS_DIR,
        course_slug,
        f"week_{week}"
    )

    os.makedirs(
        assignment_dir,
        exist_ok=True
    )

    return assignment_dir


def build_questions_file_path(course: str, week: int):
    """Return the path to questions.json for an assignment."""

    return os.path.join(
        build_assignment_directory(course, week),
        "questions.json"
    )


def build_question_image_directory(
    course: str,
    week: int,
    question_id: str
):
    """Return and create the directory for one question's images."""

    image_dir = os.path.join(
        build_assignment_directory(course, week),
        "images",
        question_id
    )

    os.makedirs(
        image_dir,
        exist_ok=True
    )

    return image_dir


# ================================================================
# URL HELPERS
# ================================================================

def get_assessment_id(url: str):
    """Extract assessmentId from an NPTEL assignment URL."""

    parsed = urlparse(url)

    query = parse_qs(parsed.query)

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

def launch_browser(playwright):
    """Launch persistent Chromium using the existing NPTEL profile."""

    context = (
        playwright.chromium
        .launch_persistent_context(
            user_data_dir=str(BROWSER_PROFILE_DIR),
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
# EXTRACT ASSIGNMENT ELEMENTS
# ================================================================

def extract_questions_from_page(
    page,
    course,
    week
):
    """
    Extract passages and questions in DOM appearance order.

    Questions with images:
        image URL -> local image path -> image_to_text(image_path)
        -> element["question"]

    Passages:
        complete passage is stored directly in element["passage"].
    """

    print("\nWaiting for assignment questions...")

    page.wait_for_selector(
        "main.practice-questions",
        timeout=QUESTION_WAIT_TIMEOUT
    )

    time.sleep(2)

    print("Assignment questions container found.")

    try:
        page.wait_for_selector(
            "main.practice-questions input[type='radio'], "
            "main.practice-questions input[type='checkbox'], "
            "main.practice-questions [class*='question-html-content'], "
            "main.practice-questions [class*='question-content']",
            timeout=QUESTION_WAIT_TIMEOUT
        )
    except Exception:
        print(
            "WARNING: Assignment container exists, but the expected "
            "question/passage content selectors were not found."
        )

    result = page.evaluate(
        r"""
        () => {
            function cleanInlineText(text) {
                if (!text) {
                    return "";
                }

                return text
                    .replace(/\u00a0/g, " ")
                    .replace(/\s+/g, " ")
                    .trim();
            }

            function cleanBlockText(text) {
                if (!text) {
                    return "";
                }

                return text
                    .replace(/\u00a0/g, " ")
                    .split(/\n+/)
                    .map(line => line.replace(/\s+/g, " ").trim())
                    .filter(Boolean)
                    .join("\n")
                    .trim();
            }

            const container = document.querySelector(
                "main.practice-questions"
            );

            if (!container) {
                return {
                    elements: [],
                    sectionCount: 0,
                    inputCount: 0,
                    questionContentCount: 0,
                    passageContentCount: 0
                };
            }

            const sections = Array.from(
                container.querySelectorAll("section")
            );

            let passageCount = 0;
            let questionCount = 0;

            const elements = [];

            sections.forEach(section => {
                const inputs = Array.from(
                    section.querySelectorAll(
                        'input[type="radio"], input[type="checkbox"]'
                    )
                );

                if (inputs.length > 0) {
                    questionCount += 1;

                    const questionId = `q${questionCount}`;

                    const answerFormat = inputs.some(
                        input => input.type === "checkbox"
                    )
                        ? "multiple_correct"
                        : "single_correct";

                    const questionElement = section.querySelector(
                        '[class*="question-content"]'
                    );

                    let questionText = questionElement
                        ? cleanInlineText(questionElement.innerText)
                        : "";

                    const questionImages = questionElement
                        ? Array.from(
                            questionElement.querySelectorAll("img")
                        )
                            .map(
                                img => img.currentSrc || img.src
                            )
                            .filter(Boolean)
                        : [];

                    questionText = questionText.replace(
                        /^\s*\d+\s*[.\)]\s*/,
                        ""
                    );

                    const options = [];

                    Array.from(
                        section.querySelectorAll("label")
                    ).forEach(label => {
                        const input = label.querySelector(
                            'input[type="radio"], input[type="checkbox"]'
                        );

                        if (!input) {
                            return;
                        }

                        const clone = label.cloneNode(true);

                        clone
                            .querySelectorAll("input")
                            .forEach(element => element.remove());

                        const optionText = cleanInlineText(
                            clone.innerText
                        );

                        if (!optionText) {
                            return;
                        }

                        options.push({
                            id: `${questionId}_o${options.length + 1}`,
                            text: optionText
                        });
                    });

                    elements.push({
                        element: "question",
                        id: questionId,
                        question_type: {
                            answer_format: answerFormat,
                            solving_strategy: null
                        },
                        question: questionText,
                        question_images: questionImages,
                        options: options,
                        passage_id: ""
                    });

                    return;
                }

                const passageElement = section.querySelector(
                    '[class*="question-html-content"]'
                );

                if (!passageElement) {
                    return;
                }

                const passageText = cleanBlockText(
                    passageElement.innerText
                );

                if (!passageText) {
                    return;
                }

                passageCount += 1;

                elements.push({
                    element: "passage",
                    id: `p${passageCount}`,
                    passage: passageText
                });
            });

            return {
                elements: elements,
                sectionCount: sections.length,
                inputCount: container.querySelectorAll(
                    'input[type="radio"], input[type="checkbox"]'
                ).length,
                questionContentCount: container.querySelectorAll(
                    '[class*="question-content"]'
                ).length,
                passageContentCount: container.querySelectorAll(
                    '[class*="question-html-content"]'
                ).length
            };
        }
        """
    )

    print(
        f"DOM sections found: {result['sectionCount']}"
    )
    print(
        f"Radio/checkbox inputs found: {result['inputCount']}"
    )
    print(
        f"Question-content elements found: "
        f"{result['questionContentCount']}"
    )
    print(
        f"Passage-content elements found: "
        f"{result['passageContentCount']}"
    )

    elements = result["elements"]

    for element in elements:
        if element["element"] != "question":
            continue

        image_urls = element.pop(
            "question_images",
            []
        )

        if not image_urls:
            continue

        image_dir = os.path.join(
            ASSIGNMENTS_DIR,
            slugify(course),
            f"week_{week}",
            "images",
            element["id"]
        )

        os.makedirs(
            image_dir,
            exist_ok=True
        )

        image_paths = []
        descriptions = []

        for image_index, image_url in enumerate(
            image_urls,
            start=1
        ):
            try:
                response = requests.get(
                    image_url,
                    timeout=20
                )
                response.raise_for_status()

                image_path = os.path.join(
                    image_dir,
                    f"image_{image_index}.png"
                )

                with open(
                    image_path,
                    "wb"
                ) as image_file:
                    image_file.write(
                        response.content
                    )

                image_paths.append(
                    image_path
                )

                print(
                    f"  Image saved: {image_path}"
                )


                description = image_to_text(
                    image_path
                )
                element["question"] += str(description)
     
            except Exception as exc:
                print(
                    f"  WARNING: Image processing failed "
                    f"for {element['id']}: {exc}"
                )
            

  
        element["question_images"] = image_paths

    return elements


# ================================================================
# SAVE ASSIGNMENT ELEMENTS
# ================================================================

def save_assignment_elements(
    elements,
    course,
    week
):
    """
    Save the complete ordered assignment structure to questions.json.

    Passages are stored directly as:
        {"element": "passage", "id": "...", "passage": "..."}
    """

    build_assignment_directory(
        course=course,
        week=week
    )

    payload = {
        "elements": elements
    }

    file_path = build_questions_file_path(
        course=course,
        week=week
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(
        "\nAssignment elements saved:"
    )
    print(
        file_path
    )

    return file_path


# ================================================================
# EXTRACT ASSIGNMENT
# ================================================================

def extract_assignment(
    assignment_url,
    course,
    week
):
    """
    Open an NPTEL assignment page and extract passages/questions.
    """

    if not assignment_url:
        raise ValueError(
            "assignment_url is required"
        )

    assessment_id = get_assessment_id(
        assignment_url
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
        f"\nAssignment ID: {assessment_id}"
    )

    with sync_playwright() as playwright:
        context, page = launch_browser(
            playwright
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

            if (
                "login" in page.url.lower()
                or
                "signin" in page.url.lower()
            ):
                raise RuntimeError(
                    "NPTEL authentication required."
                )

            elements = extract_questions_from_page(
                page=page,
                course=course,
                week=week
            )

            if not elements:
                raise RuntimeError(
                    "No assignment passages or questions found."
                )

            question_count = sum(
                1
                for element in elements
                if element["element"] == "question"
            )

            passage_count = sum(
                1
                for element in elements
                if element["element"] == "passage"
            )

            print(
                f"\nTotal passages extracted: {passage_count}"
            )
            print(
                f"Total questions extracted: {question_count}"
            )

            return {
                "assignment_id": assessment_id,
                "elements": elements
            }

        finally:
            context.close()


# ================================================================
# ENRICH JOBS WITH ASSIGNMENTS
# ================================================================

def enrich_jobs_with_assignments(jobs):
    """
    Pipeline stage:

    jobs
        -> extract assignment passages/questions
        -> save questions.json + passage .txt files
        -> add assignment_file to job

    Input:

    {
        "course": "...",
        "week": 5,
        "assignment_url": "..."
    }

    Output retains the input job and adds:

    {
        "assignment_file":
            "data/assignments/course_name/week_6/questions.json"
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
            f"PROCESSING ASSIGNMENT {index}/{len(jobs)}"
        )

        print(
            "#" * 80
        )

        assignment_url = job.get(
            "assignment_url"
        )

        if not assignment_url:

            raise ValueError(
                "Job has no assignment_url"
            )


        # --------------------------------------------------------
        # EXTRACT PASSAGES + QUESTIONS
        # --------------------------------------------------------

        assignment = extract_assignment(
            assignment_url=assignment_url,
            course=job["course"],
            week=job["week"]
        )


        # --------------------------------------------------------
        # SAVE QUESTIONS + PASSAGES
        # --------------------------------------------------------

        assignment_file = save_assignment_elements(
            elements=assignment["elements"],
            course=job["course"],
            week=job["week"]
        )


        # --------------------------------------------------------
        # CREATE ENRICHED JOB
        # --------------------------------------------------------

        enriched_job = job.copy()

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
    "course": "Advanced Algorithmic Trading and Portfolio Management",
    "week": 5,
    "assignment_url": (
        "https://onlinecourses.nptel.ac.in/"
        "e-learning/course/noc24_mg83"
        "?unitId=60&assessmentId=110"
    )
}


# ================================================================
# TEST MAIN
# ================================================================

def main():

    print(
        "\n" + "#" * 80
    )

    print(
        "TESTING ASSIGNMENT EXTRACTION"
    )

    print(
        "#" * 80
    )

    print(
        "\nCourse:"
    )

    print(
        TEST_JOB["course"]
    )

    print(
        "\nWeek:"
    )

    print(
        TEST_JOB["week"]
    )

    print(
        "\nAssignment URL:"
    )

    print(
        TEST_JOB["assignment_url"]
    )


    # ------------------------------------------------------------
    # Run complete assignment pipeline
    #
    # Job
    #   -> extract ordered elements
    #   -> save questions.json + passage files
    #   -> add assignment_file to job
    # ------------------------------------------------------------

    enriched_jobs = enrich_jobs_with_assignments(
        [TEST_JOB]
    )

    print(
        "\n"
    )

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

    print(
        "\n" + "=" * 80
    )

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
                f"\nAssignment file:\n{assignment_file}"
            )

            if os.path.exists(
                assignment_file
            ):

                print(
                    "Status: EXISTS"
                )

                print(
                    f"Size: {os.path.getsize(assignment_file)} bytes"
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





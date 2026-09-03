import os
import json
import time
from pathlib import Path

from groq import Groq


# ================================================================
# CONFIG
# ================================================================

MODEL = "openai/gpt-oss-120b"

JOB_DELAY_SECONDS = 65


# ================================================================
# GROQ CLIENT
# ================================================================

def get_groq_client():
    """
    Create and return Groq client.

    Expects GROQ_API_KEY to already exist
    in environment variables.
    """

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable not found."
        )

    return Groq(
        api_key=api_key
    )


# ================================================================
# FILE LOADING
# ================================================================

def load_transcript(transcript_file):
    """
    Load the complete week's transcript.
    """

    path = Path(
        transcript_file
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Transcript file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def load_assignment(assignment_file):
    """
    Load assignment questions JSON.
    """

    path = Path(
        assignment_file
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Assignment file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ================================================================
# PROMPT BUILDING
# ================================================================

def build_solver_prompt(
    transcript,
    questions
):
    """
    Build minimal solver prompt.
    """

    questions_json = json.dumps(
        questions,
        ensure_ascii=False
    )

    return f"""COURSE TRANSCRIPT:
{transcript}

QUESTIONS:
{questions_json}

Solve all questions based on the transcript.

Return valid JSON only in this format:

{{
  "solutions": [
    {{
      "question_id": "q1",
      "selected_option_ids": ["q1_o1"]
    }}
  ]
}}

For each question, return its question_id and the selected option ID(s).
"""


# ================================================================
# LLM RESPONSE PARSING
# ================================================================

def extract_json_from_response(
    response_text
):
    """
    Extract JSON safely from model output.

    The prompt asks for pure JSON, but this function
    handles accidental markdown fences as well.
    """

    text = response_text.strip()

    # Remove markdown fences if model adds them
    if text.startswith(
        "```json"
    ):

        text = text[
            len("```json"):
        ]

        if text.endswith(
            "```"
        ):

            text = text[
                :-3
            ]

    elif text.startswith(
        "```"
    ):

        text = text[3:]

        if text.endswith(
            "```"
        ):

            text = text[
                :-3
            ]

    text = text.strip()

    return json.loads(
        text
    )


# ================================================================
# VALIDATE SOLUTIONS
# ================================================================

def validate_solutions(
    assignment_questions,
    solver_result
):
    """
    Validate that:

    - Every assignment question has an answer
    - No unknown question IDs exist
    - Option IDs belong to the correct question
    - single_correct has exactly one option
    - multi_correct has at least one option
    """

    if "solutions" not in solver_result:

        raise ValueError(
            "Solver response does not contain 'solutions'."
        )

    solutions = solver_result[
        "solutions"
    ]

    if not isinstance(
        solutions,
        list
    ):

        raise ValueError(
            "'solutions' must be a list."
        )

    # ------------------------------------------------------------
    # Build lookup tables
    # ------------------------------------------------------------

    question_map = {}

    for question in assignment_questions:

        question_map[
            question["id"]
        ] = question

    solution_question_ids = set()

    validated_solutions = []

    # ------------------------------------------------------------
    # Validate every returned solution
    # ------------------------------------------------------------

    for solution in solutions:

        question_id = solution.get(
            "question_id"
        )

        selected_option_ids = solution.get(
            "selected_option_ids"
        )

        reasoning = solution.get(
            "reasoning"
        )

        confidence = solution.get(
            "confidence"
        )

        if question_id not in question_map:

            raise ValueError(
                f"Unknown question ID returned: "
                f"{question_id}"
            )

        if question_id in solution_question_ids:

            raise ValueError(
                f"Duplicate solution for question: "
                f"{question_id}"
            )

        solution_question_ids.add(
            question_id
        )

        if not isinstance(
            selected_option_ids,
            list
        ):

            raise ValueError(
                f"{question_id}: "
                f"'selected_option_ids' must be a list."
            )

        if not selected_option_ids:

            raise ValueError(
                f"{question_id}: "
                f"No options selected."
            )

        # --------------------------------------------------------
        # Validate option IDs
        # --------------------------------------------------------

        question = question_map[
            question_id
        ]

        valid_option_ids = {

            option["id"]

            for option in question[
                "options"
            ]

        }

        for option_id in selected_option_ids:

            if option_id not in valid_option_ids:

                raise ValueError(
                    f"{question_id}: "
                    f"Invalid option ID returned: "
                    f"{option_id}"
                )

        # --------------------------------------------------------
        # Validate question type
        # --------------------------------------------------------

        question_type = question.get(
            "type"
        )

        if (
            question_type == "single_correct"
            and len(selected_option_ids) != 1
        ):

            raise ValueError(
                f"{question_id}: "
                f"single_correct question must have "
                f"exactly one selected option."
            )

        if (
            question_type == "multi_correct"
            and len(selected_option_ids) < 1
        ):

            raise ValueError(
                f"{question_id}: "
                f"multi_correct question must have "
                f"at least one selected option."
            )

        # --------------------------------------------------------
        # Validate confidence
        # --------------------------------------------------------

        if confidence is not None:

            try:

                confidence = float(
                    confidence
                )

            except (
                TypeError,
                ValueError
            ):

                raise ValueError(
                    f"{question_id}: "
                    f"Invalid confidence value."
                )

            if not (
                0 <= confidence <= 1
            ):

                raise ValueError(
                    f"{question_id}: "
                    f"Confidence must be between 0 and 1."
                )

        # --------------------------------------------------------
        # Store validated solution
        # --------------------------------------------------------

        validated_solutions.append(
            {
                "question_id": question_id,

                "selected_option_ids":
                    selected_option_ids,

                "reasoning":
                    reasoning,

                "confidence":
                    confidence
            }
        )

    # ------------------------------------------------------------
    # Ensure every question was solved
    # ------------------------------------------------------------

    expected_question_ids = set(
        question_map.keys()
    )

    missing_questions = (
        expected_question_ids
        -
        solution_question_ids
    )

    if missing_questions:

        raise ValueError(
            "Missing solutions for questions: "
            f"{sorted(missing_questions)}"
        )

    return validated_solutions


# ================================================================
# OUTPUT PATH
# ================================================================

def get_solution_file_path(
    job
):
    """
    Generate:

    data/solutions/<course_slug>/week_N.json
    """

    course_slug = (
        job["course"]
        .lower()
        .replace(" ", "_")
    )

    week = job["week"]

    output_dir = (
        Path("data")
        / "solutions"
        / course_slug
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        output_dir
        / f"week_{week}.json"
    )


# ================================================================
# SOLVE ONE JOB
# ================================================================

def solve_job(
    job,
    client=None
):
    """
    Solve one week's assignment.

    ONE LLM CALL:

    complete transcript
        +
    all assignment questions
        ↓
    model
        ↓
    validated solutions
        ↓
    saved JSON
    """

    print("\n" + "=" * 80)
    print("SOLVING ASSIGNMENT")
    print("=" * 80)

    print(
        f"\nCourse: {job['course']}"
    )

    print(
        f"Week: {job['week']}"
    )

    # ------------------------------------------------------------
    # Load files
    # ------------------------------------------------------------

    transcript = load_transcript(
        job["transcript_file"]
    )

    assignment = load_assignment(
        job["assignment_file"]
    )

    questions = assignment.get(
        "questions",
        []
    )

    if not questions:

        raise ValueError(
            "No questions found in assignment file."
        )

    print(
        f"\nTranscript characters: "
        f"{len(transcript):,}"
    )

    print(
        f"Questions: "
        f"{len(questions)}"
    )

    # ------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------

    prompt = build_solver_prompt(
        transcript=transcript,
        questions=questions
    )

    print(
        f"Prompt characters: "
        f"{len(prompt):,}"
    )

    # ------------------------------------------------------------
    # Create client if necessary
    # ------------------------------------------------------------

    if client is None:

        client = get_groq_client()

    # ------------------------------------------------------------
    # LLM CALL
    # ------------------------------------------------------------

    print("\nCalling Groq...")
    print(
        "Sending complete transcript + all questions "
        "in ONE request..."
    )

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,

        response_format={
            "type": "json_object"
        }
    )

    response_text = (
        response
        .choices[0]
        .message
        .content
    )

    if not response_text:

        raise ValueError(
            "Groq returned an empty response."
        )

    # ------------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------------

    print(
        "Parsing model response..."
    )

    solver_result = (
        extract_json_from_response(
            response_text
        )
    )

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    print(
        "Validating solutions..."
    )

    validated_solutions = (
        validate_solutions(
            questions,
            solver_result
        )
    )

    # ------------------------------------------------------------
    # Build final output
    # ------------------------------------------------------------

    result = {

        "course":
            job["course"],

        "week":
            job["week"],

        "transcript_file":
            job["transcript_file"],

        "assignment_file":
            job["assignment_file"],

        "model":
            MODEL,

        "total_questions":
            len(questions),

        "solutions":
            validated_solutions
    }

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_path = (
        get_solution_file_path(
            job
        )
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\n✓ Solutions saved:"
    )

    print(
        output_path
    )

    # Enrich job for later pipeline stages
    job["solution_file"] = str(
        output_path
    )

    job["solution_summary"] = {

        "total_questions":
            len(questions),

        "solved_questions":
            len(validated_solutions),

        "model":
            MODEL
    }

    return job


# ================================================================
# SOLVE MULTIPLE JOBS
# ================================================================

def solve_jobs(
    jobs
):
    """
    Solve multiple NPTEL jobs.

    Important:

    - ONE LLM call per job
    - Wait 65 seconds between jobs
    - Last job does not wait
    """

    if not jobs:

        print(
            "\nNo jobs provided to solver."
        )

        return []

    client = get_groq_client()

    solved_jobs = []

    total_jobs = len(
        jobs
    )

    for index, job in enumerate(
        jobs,
        start=1
    ):

        print("\n")
        print("#" * 80)

        print(
            f"SOLVER JOB "
            f"{index}/{total_jobs}"
        )

        print("#" * 80)

        try:

            solved_job = solve_job(
                job=job,
                client=client
            )

            solved_jobs.append(
                solved_job
            )

            print(
                "\n✓ JOB SOLVED SUCCESSFULLY"
            )

        except Exception as e:

            print(
                "\n✗ SOLVER FAILED"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            # Keep pipeline job even if solving failed
            job["solver_error"] = (
                f"{type(e).__name__}: {e}"
            )

            solved_jobs.append(
                job
            )

        # --------------------------------------------------------
        # WAIT BEFORE NEXT JOB
        # --------------------------------------------------------

        if index < total_jobs:

            print("\n" + "-" * 80)

            print(
                f"Waiting {JOB_DELAY_SECONDS} seconds "
                f"before next job..."
            )

            print(
                "This helps avoid Groq "
                "tokens-per-minute rate limits."
            )

            print("-" * 80)

            time.sleep(
                JOB_DELAY_SECONDS
            )

    return solved_jobs


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    TEST_JOB = {

        "course":
            "Research Methodology",

        "week":
            6,

        "transcript_file":
            "data/transcripts/"
            "research_methodology/"
            "week_6.txt",

        "assignment_file":
            "data/assignments/"
            "research_methodology/"
            "week_6.json"
    }

    solve_jobs(
        [TEST_JOB]
    )

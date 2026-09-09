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

def load_assignment(
    assignment_file
):
    """
    Load questions.json.

    Expected structure:

    {
        "elements": [
            {
                "element": "passage",
                "id": "p1",
                "preview": "..."
            },
            {
                "element": "question",
                "id": "q1",
                "question_type": {
                    "answer_format": "single_correct",
                    "solving_strategy": null
                },
                "question": "...",
                "options": [...],
                "passage_id": ""
            }
        ]
    }
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

        assignment = json.load(
            file
        )

    if not isinstance(
        assignment,
        dict
    ):
        raise ValueError(
            "Assignment JSON must contain an object."
        )

    elements = assignment.get(
        "elements"
    )

    if not isinstance(
        elements,
        list
    ):
        raise ValueError(
            "Assignment JSON must contain an 'elements' list."
        )

    return assignment


# ================================================================
# PROMPT BUILDING
# ================================================================

def build_classifier_prompt(
    elements
):
    """
    Build the classification prompt.

    The complete ordered elements array is provided to the LLM.
    """

    elements_json = json.dumps(
        elements,
        ensure_ascii=False,
        indent=2
    )

    return f"""You are classifying questions from an NPTEL assignment.

The assignment is provided as an ORDERED sequence of elements.

Each element is either:

- "passage"
- "question"

The order of elements is important.

ASSIGNMENT ELEMENTS:
{elements_json}



TASK

For EVERY question element, classify its solving strategy as
exactly one of:

1. "conceptual"

   The question is primarily theoretical/conceptual.

   It can be answered without:
   - performing a numerical calculation
   - relying on a specific passage in the assignment

2. "numerical"

   The question requires:
   - calculation
   - mathematical reasoning
   - applying a formula
   - numerical manipulation
   - interpreting numerical data
   - or another form of quantitative reasoning

   It is NOT primarily a passage-based question.

3. "passage"

   The question is intended to be answered using information
   from a passage appearing in the assignment.


PASSAGE ASSOCIATION

Passage association is determined from the passage's instructions
AND the ordered structure of the assignment.

There are two cases.

CASE 1 — EXPLICIT QUESTION REFERENCES

If the passage explicitly specifies question numbers or a range,
follow those references exactly.

Examples:

"Answer questions 2 and 3 based on the passage."

    q2 -> passage
    q3 -> passage

"Answer questions 4-6 based on the passage."

    q4 -> passage
    q5 -> passage
    q6 -> passage


CASE 2 — NO QUESTION NUMBERS / COUNT

Sometimes a passage says something like:

"Read the passage below and answer the questions based on
the passage."

or:

"Answer the following questions based on the passage."

without specifying the question numbers or number of questions.

In this case:

- Associate the passage with the contiguous question elements
  immediately following that passage.
- Continue until the next passage element OR the end of the
  assignment.
- All those immediately following questions are passage questions.
- Multiple questions may share the same passage_id.

Example:

p1:
"Read the passage below and answer the questions based on
the passage."

q7
q8

p2:
"Read the passage below and answer the questions based on
the passage."

q9
q10

Therefore:

q7 -> passage p1
q8 -> passage p1
q9 -> passage p2
q10 -> passage p2


CASE 3 — EXPLICIT NUMBER OF NEXT QUESTIONS

If the passage says:

"Answer the next 2 questions based on the passage."

associate exactly the next 2 question elements with that passage.

Do NOT continue beyond those 2 questions.

Example:

p1:
"Answer the next 2 questions based on the passage."

q1
q2
q3

Result:

q1 -> passage p1
q2 -> passage p1
q3 -> classify independently


IMPORTANT:

An explicit question number/range always takes precedence over
the positional fallback.

Do not associate questions appearing before a passage with that
passage.

Do not associate unrelated questions after a passage when the
passage explicitly specifies a smaller range.


OUTPUT FORMAT

Return VALID JSON only.

Return exactly ONE entry for EVERY question.

Required format:

{{
  "q1": {{
    "solving_strategy": "conceptual"
  }},
  "q2": {{
    "solving_strategy": "numerical"
  }},
  "q3": {{
    "solving_strategy": "passage",
    "passage_id": "p1"
  }}
}}


OUTPUT RULES

- Every question ID must appear exactly once.

- Do not include passage IDs as top-level keys.

- Do not include passage elements as output entries.

- Do not include question text.

- Do not include options.

- Do not provide explanations.

- Do not provide answers.

- Do not provide selected option IDs.

- "solving_strategy" must be exactly one of:

    "conceptual"
    "numerical"
    "passage"

- conceptual or numerical questions MUST NOT have passage_id.

- passage questions MUST have passage_id.

- passage_id must be the ID of an actual passage element.
"""


# ================================================================
# LLM RESPONSE PARSING
# ================================================================

def extract_json_from_response(
    response_text
):
    """
    Extract JSON safely from model output.

    Handles accidental markdown fences.
    """

    text = response_text.strip()

    # ------------------------------------------------------------
    # Remove markdown fences
    # ------------------------------------------------------------

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

        text = text[
            3:
        ]

        if text.endswith(
            "```"
        ):

            text = text[
                :-3
            ]

    text = text.strip()

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            "LLM returned invalid JSON: "
            f"{e}"
        )


# ================================================================
# VALIDATE CLASSIFICATIONS
# ================================================================

def validate_classifications(
    elements,
    classifier_result
):
    """
    Validate the classifier output.

    Checks:

    - Every question is classified.
    - No unknown question IDs exist.
    - Strategy is valid.
    - Passage questions have passage_id.
    - Passage IDs actually exist.
    - Non-passage questions do not have passage_id.
    """

    if not isinstance(
        classifier_result,
        dict
    ):

        raise ValueError(
            "Classifier response must be a JSON object."
        )

    # ------------------------------------------------------------
    # Extract questions and passages
    # ------------------------------------------------------------

    questions = {}

    passages = {}

    for element in elements:

        if not isinstance(
            element,
            dict
        ):
            continue

        element_type = element.get(
            "element"
        )

        element_id = element.get(
            "id"
        )

        if not element_id:
            continue

        if element_type == "question":

            questions[
                element_id
            ] = element

        elif element_type == "passage":

            passages[
                element_id
            ] = element

    # ------------------------------------------------------------
    # Expected / returned IDs
    # ------------------------------------------------------------

    expected_question_ids = set(
        questions.keys()
    )

    returned_question_ids = set(
        classifier_result.keys()
    )

    # ------------------------------------------------------------
    # Missing questions
    # ------------------------------------------------------------

    missing_questions = (
        expected_question_ids
        -
        returned_question_ids
    )

    if missing_questions:

        raise ValueError(
            "Missing classifications for questions: "
            f"{sorted(missing_questions)}"
        )

    # ------------------------------------------------------------
    # Unknown question IDs
    # ------------------------------------------------------------

    unknown_questions = (
        returned_question_ids
        -
        expected_question_ids
    )

    if unknown_questions:

        raise ValueError(
            "Unknown question IDs returned: "
            f"{sorted(unknown_questions)}"
        )

    # ------------------------------------------------------------
    # Validate individual classifications
    # ------------------------------------------------------------

    validated = {}

    valid_strategies = {
        "conceptual",
        "numerical",
        "passage"
    }

    for question_id in expected_question_ids:

        classification = (
            classifier_result[
                question_id
            ]
        )

        if not isinstance(
            classification,
            dict
        ):

            raise ValueError(
                f"{question_id}: "
                "classification must be an object."
            )

        strategy = (
            classification.get(
                "solving_strategy"
            )
        )

        # --------------------------------------------------------
        # Validate strategy
        # --------------------------------------------------------

        if strategy not in valid_strategies:

            raise ValueError(
                f"{question_id}: invalid "
                f"solving_strategy '{strategy}'."
            )

        passage_id = (
            classification.get(
                "passage_id"
            )
        )

        # --------------------------------------------------------
        # Passage question
        # --------------------------------------------------------

        if strategy == "passage":

            if not passage_id:

                raise ValueError(
                    f"{question_id}: passage question "
                    "must have passage_id."
                )

            if passage_id not in passages:

                raise ValueError(
                    f"{question_id}: invalid passage_id "
                    f"'{passage_id}'."
                )

            validated[
                question_id
            ] = {
                "solving_strategy":
                    "passage",

                "passage_id":
                    passage_id
            }

        # --------------------------------------------------------
        # Conceptual / numerical
        # --------------------------------------------------------

        else:

            if passage_id is not None:

                raise ValueError(
                    f"{question_id}: "
                    f"{strategy} question must not "
                    "have passage_id."
                )

            validated[
                question_id
            ] = {
                "solving_strategy":
                    strategy
            }

    return validated


# ================================================================
# OUTPUT PATH
# ================================================================

def get_classification_file_path(
    assignment_file
):
    """
    Generate:

    data/assignments/<course>/week_N/classification.json

    based on the location of questions.json.
    """

    assignment_path = Path(
        assignment_file
    )

    return (
        assignment_path.parent
        / "classification.json"
    )


# ================================================================
# CLASSIFY ONE ASSIGNMENT
# ================================================================

def classify_assignment(
    assignment_file,
    client=None
):
    """
    Classify all questions in questions.json.

    ONE LLM CALL:

        questions.json
             ↓
           LLM
             ↓
       classifications
             ↓
         validation
             ↓
      classification.json
    """

    print(
        "\n" + "=" * 80
    )

    print(
        "CLASSIFYING ASSIGNMENT QUESTIONS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nAssignment file:"
    )

    print(
        assignment_file
    )

    # ------------------------------------------------------------
    # Load assignment
    # ------------------------------------------------------------

    assignment = load_assignment(
        assignment_file
    )

    elements = assignment[
        "elements"
    ]

    # ------------------------------------------------------------
    # Count elements
    # ------------------------------------------------------------

    question_count = sum(
        1
        for element in elements
        if element.get("element") == "question"
    )

    passage_count = sum(
        1
        for element in elements
        if element.get("element") == "passage"
    )

    print(
        f"\nTotal elements: "
        f"{len(elements)}"
    )

    print(
        f"Questions: "
        f"{question_count}"
    )

    print(
        f"Passages: "
        f"{passage_count}"
    )

    if question_count == 0:

        raise ValueError(
            "No questions found in assignment."
        )

    # ------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------

    prompt = build_classifier_prompt(
        elements
    )

    print(
        f"\nPrompt characters: "
        f"{len(prompt):,}"
    )

    # ------------------------------------------------------------
    # Create client
    # ------------------------------------------------------------

    if client is None:

        client = get_groq_client()

    # ------------------------------------------------------------
    # LLM CALL
    # ------------------------------------------------------------

    print(
        "\nCalling Groq..."
    )

    print(
        "Sending questions.json "
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

        temperature=0,

        response_format={
            "type": "json_object"
        }
    )
    usage = response.usage

    print("TOKEN USAGE")


    print(
        f"Prompt tokens:     {usage.prompt_tokens:,}"
    )
    print(
        f"Completion tokens: {usage.completion_tokens:,}"
    )
    print(
        f"Total tokens:      {usage.total_tokens:,}"
    )
    # ------------------------------------------------------------
    # Extract response
    # ------------------------------------------------------------

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
        "Parsing classifier response..."
    )

    classifier_result = (
        extract_json_from_response(
            response_text
        )
    )

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    print(
        "Validating classifications..."
    )

    classifications = (
        validate_classifications(
            elements=elements,
            classifier_result=classifier_result
        )
    )
    # ------------------------------------------------------------
    # Assign classifications to questions.json
    # ------------------------------------------------------------

    assign_classification(
        assignment_file=assignment_file,
        classifications=classifications
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_path = (
        get_classification_file_path(
            assignment_file
        )
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            classifications,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\n✓ Classifications saved:"
    )

    print(
        output_path
    )

    # ------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------

    print(
        "\nClassification:"
    )

    print(
        json.dumps(
            classifications,
            indent=2,
            ensure_ascii=False
        )
    )

    return classifications


# ================================================================
# CLASSIFY MULTIPLE JOBS
# ================================================================

def classify_jobs(
    jobs
):
    """
    Classify multiple assignment jobs.

    Important:

    - ONE LLM call per assignment
    - Wait 65 seconds between jobs
    - Last job does not wait
    """

    if not jobs:

        print(
            "\nNo jobs provided to classifier."
        )

        return []

    client = get_groq_client()

    classified_jobs = []

    total_jobs = len(
        jobs
    )

    for index, job in enumerate(
        jobs,
        start=1
    ):

        print(
            "\n"
        )

        print(
            "#" * 80
        )

        print(
            f"CLASSIFIER JOB "
            f"{index}/{total_jobs}"
        )

        print(
            "#" * 80
        )

        try:

            classifications = (
                classify_assignment(
                    assignment_file=
                        job["assignment_file"],

                    client=
                        client
                )
            )

            classified_job = (
                job.copy()
            )

            classified_job[
                "classification_file"
            ] = str(
                get_classification_file_path(
                    job["assignment_file"]
                )
            )

            classified_job[
                "classification_summary"
            ] = {

                "total_questions":
                    len(classifications),

                "model":
                    MODEL
            }

            classified_jobs.append(
                classified_job
            )

            print(
                "\n✓ JOB CLASSIFIED SUCCESSFULLY"
            )

        except Exception as e:

            print(
                "\n✗ CLASSIFIER FAILED"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            failed_job = (
                job.copy()
            )

            failed_job[
                "classifier_error"
            ] = (
                f"{type(e).__name__}: {e}"
            )

            classified_jobs.append(
                failed_job
            )

        # --------------------------------------------------------
        # Wait before next job
        # --------------------------------------------------------

        if index < total_jobs:

            print(
                "\n" + "-" * 80
            )

            print(
                f"Waiting {JOB_DELAY_SECONDS} seconds "
                f"before next job..."
            )

            print(
                "-" * 80
            )

            time.sleep(
                JOB_DELAY_SECONDS
            )

    return classified_jobs

# ================================================================
# ASSIGN CLASSIFICATIONS TO QUESTIONS
# ================================================================

def assign_classification(
    assignment_file,
    classifications
):
    """
    Attach classifier results directly to questions.json.

    For every question:

    - question_type.solving_strategy is updated
    - passage_id is updated for passage questions

    Example:

    {
        "question_type": {
            "answer_format": "single_correct",
            "solving_strategy": "passage"
        },
        "passage_id": "p1"
    }
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

        assignment = json.load(
            file
        )

    elements = assignment.get(
        "elements"
    )

    if not isinstance(
        elements,
        list
    ):

        raise ValueError(
            "Assignment JSON must contain an 'elements' list."
        )

    # ------------------------------------------------------------
    # Attach classifications
    # ------------------------------------------------------------

    for element in elements:

        if element.get(
            "element"
        ) != "question":

            continue

        question_id = element.get(
            "id"
        )

        if question_id not in classifications:

            raise ValueError(
                f"No classification found for {question_id}."
            )

        classification = classifications[
            question_id
        ]

        strategy = classification[
            "solving_strategy"
        ]

        # --------------------------------------------------------
        # Update solving strategy
        # --------------------------------------------------------

        if "question_type" not in element:

            element["question_type"] = {}

        element[
            "question_type"
        ][
            "solving_strategy"
        ] = strategy

        # --------------------------------------------------------
        # Update passage association
        # --------------------------------------------------------

        if strategy == "passage":

            element[
                "passage_id"
            ] = classification[
                "passage_id"
            ]

        else:

            element[
                "passage_id"
            ] = ""

    # ------------------------------------------------------------
    # Save updated questions.json
    # ------------------------------------------------------------

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            assignment,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\n✓ Classifications attached to questions.json:"
    )

    print(
        path
    )

    return assignment
# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    TEST_JOB = {

        "course":
            "Advanced Algorithmic Trading and Portfolio Management",

        "week":
            5,

        "assignment_file":
            "data/assignments/"
            "advanced_algorithmic_trading_and_portfolio_management/"
            "week_5/"
            "questions.json"
    }

    classify_jobs(
        [TEST_JOB]
    )

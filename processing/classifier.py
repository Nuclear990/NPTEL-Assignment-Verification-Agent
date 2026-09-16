import os
import json
import time
from pathlib import Path

from groq import Groq, RateLimitError

from ingestion.assignment import build_questions_file_path


# ================================================================
# CONFIG
# ================================================================

MODEL = "openai/gpt-oss-20b"

# In testing, gpt-oss-20b in JSON mode failed every call on an
# assignment with passages at default/high reasoning effort (400
# json_validate_failed, empty output); "low" classified it correctly.
REASONING_EFFORT = "low"

# A rate limit asking to retry within this many seconds is the
# per-minute token limit — wait it out. A longer one is the daily
# quota: block classification for the rest of this run.
MAX_RATE_LIMIT_WAIT_SECONDS = 90
MAX_RATE_LIMIT_WAITS = 5


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
# CROSS-JOB CLASSIFICATION BLOCKING
# ================================================================
#
# A rate limit that can't be waited out (the daily quota) is
# systemic for this run, the same way it is for cleaning (see
# rag/indexing/index.py's is_cleaning_blocked): every job after
# this one would hit it again. Separate from the solver's flag —
# the classifier runs on gpt-oss-20b and the solver on
# gpt-oss-120b, and Groq rate limits each model separately.
#

_classification_blocked = False


def is_classification_blocked() -> bool:

    return _classification_blocked


def _block_classification(reason: str) -> None:

    global _classification_blocked

    if not _classification_blocked:

        print(
            f"\n🚫 Blocking further classification for the rest of "
            f"this run: {reason}"
        )

    _classification_blocked = True


def _retry_after_seconds(error: RateLimitError) -> float:

    try:

        return float(error.response.headers.get("retry-after"))

    except (TypeError, ValueError):

        return 60.0


def _create_completion(client, prompt):
    """
    The classification call, waiting out short (per-minute) rate
    limits. A long (daily) rate limit, or too many waits, blocks
    classification for the rest of this run and is re-raised.
    """

    waits = 0

    while True:

        try:

            return client.chat.completions.create(

                model=MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0,

                reasoning_effort=REASONING_EFFORT,

                response_format={
                    "type": "json_object"
                }
            )

        except RateLimitError as e:

            retry_after = _retry_after_seconds(e)

            if (
                retry_after > MAX_RATE_LIMIT_WAIT_SECONDS
                or waits >= MAX_RATE_LIMIT_WAITS
            ):

                _block_classification(
                    f"{type(e).__name__}: {e}"
                )

                raise

            waits += 1

            print(
                f"Rate limited — waiting {retry_after:.0f}s "
                f"({waits}/{MAX_RATE_LIMIT_WAITS})"
            )

            time.sleep(retry_after)


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
                "options": [...]
            }
        ]
    }

    Questions carry no passage_id until classification adds one to
    those classified as passage questions.
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
        "\nAssignment file:"
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

    response = _create_completion(
        client,
        prompt
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
    - passage_id is set only on passage questions (and removed
      from any other question)

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

            element.pop(
                "passage_id",
                None
            )

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
# ENSURE ONE ASSIGNMENT IS CLASSIFIED (SKIP IF ALREADY DONE)
# ================================================================

def ensure_assignment_classified(
    course: str,
    week
):
    """
    Classify course + week's questions.json unless every question
    in it already has a solving_strategy. Returns the
    classifications, or None when the assignment isn't scraped yet,
    was already classified, or classification is blocked for this
    run by a rate limit.
    """

    if is_classification_blocked():

        print(
            "\nClassification is blocked for the rest of this run — "
            "skipping."
        )

        return None

    assignment_file = build_questions_file_path(
        course,
        week
    )

    if not os.path.exists(assignment_file):

        print(
            "\nAssignment not scraped yet — skipping classification."
        )

        return None

    elements = load_assignment(
        assignment_file
    )["elements"]

    already_classified = all(
        element.get("question_type", {}).get("solving_strategy")
        for element in elements
        if element.get("element") == "question"
    )

    if already_classified:

        print(
            f"Assignment already classified -> {assignment_file}"
        )

        return None

    try:

        return classify_assignment(
            assignment_file
        )

    except RateLimitError:

        # Classification is now blocked for the rest of this run
        # (see _create_completion) — retried on a later run.
        return None

"""
Solve one classified assignment (course + week) into answers.json.

- conceptual / numerical question: retrieve the relevant lecture
  chunks, then one LLM call with question + options + chunks (no
  Python calculation tool for numerical questions yet)
- passage question: one LLM call with question + options + passage
- output: answers.json next to questions.json, one record per
  question: {question_id, correct_options, reasoning, confidence}
- every question gets an answer, even a low-confidence one
"""

import json
import os
import time
from pathlib import Path

from groq import BadRequestError, Groq, RateLimitError

from ingestion.assignment import build_questions_file_path
from rag.retrieval.retrieval import (
    load_retrieval_session,
    retrieve_context_for_question
)


# ================================================================
# CONFIG
# ================================================================

MODEL = "openai/gpt-oss-120b"

MAX_ATTEMPTS = 3            # per question, when the reply is empty/truncated/invalid
SLEEP_BETWEEN_CALLS = 2     # seconds, be gentle on TPM rate limits

# A rate limit asking to retry within this many seconds is the
# per-minute token limit — wait it out. A longer one is the daily
# request quota: give up for this run (answers so far are saved).
MAX_RATE_LIMIT_WAIT_SECONDS = 90
MAX_RATE_LIMIT_WAITS = 5

_client = None


# ================================================================
# CROSS-JOB SOLVING BLOCKING
# ================================================================
#
# A rate limit that can't be waited out (the daily quota) is
# systemic for this run, the same way it is for cleaning (see
# rag/indexing/index.py's is_cleaning_blocked): every later
# question, and every job after this one, would hit it again.
# Separate from the classifier's flag — the solver runs on
# gpt-oss-120b and the classifier on gpt-oss-20b, and Groq rate
# limits each model separately.
#

_solving_blocked = False


def is_solving_blocked() -> bool:

    return _solving_blocked


def _block_solving(reason: str) -> None:

    global _solving_blocked

    if not _solving_blocked:

        print(
            f"\n🚫 Blocking further solving for the rest of "
            f"this run: {reason}"
        )

    _solving_blocked = True


def get_client():
    """Create (and cache) the Groq client from GROQ_API_KEY."""

    global _client

    if _client is None:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable not found."
            )

        _client = Groq(api_key=api_key)

    return _client


# ================================================================
# PROMPT
# ================================================================

SYSTEM_PROMPT = """
You are answering one multiple-choice question from an NPTEL course
assignment.

You are given the question, its options (each with an id), and
supporting material: either excerpts from the course's lecture
transcripts, or a passage from the assignment itself.

Rules:
1. Base the answer on the supporting material first. If it does not
   settle the question, use your own knowledge of the subject and
   lower your confidence accordingly.
2. You MUST answer. Never refuse or reply that the material is
   insufficient — pick the best option(s) and put any doubt into the
   confidence instead.
3. "single_correct": select exactly one option id.
   "multiple_correct": select every correct option id (one or more).
4. Only use option ids that appear in the question.
5. For numerical questions, work the calculation step by step and
   check the arithmetic before choosing.

Return VALID JSON only, in exactly this format:

{
  "correct_options": ["q1_o2"],
  "reasoning": "why these options are correct, citing the material",
  "confidence": 0.85
}

confidence is a number from 0 to 1.
""".strip()


def build_user_prompt(
    question: dict,
    material_label: str,
    material: str
) -> str:

    options_text = "\n".join(
        f"{option['id']}: {option['text']}"
        for option in question["options"]
    )

    return (
        f"Answer format: {question['question_type']['answer_format']}\n"
        f"Question type: {question['question_type']['solving_strategy']}\n\n"
        f"QUESTION ({question['id']}):\n{question['question']}\n\n"
        f"OPTIONS:\n{options_text}\n\n"
        f"{material_label}:\n{material or '(none found)'}"
    )


# ================================================================
# LLM CALL
# ================================================================

def _retry_after_seconds(error: RateLimitError) -> float:

    try:

        return float(error.response.headers.get("retry-after"))

    except (TypeError, ValueError):

        return 60.0


def _create_completion(user_prompt: str):
    """
    One chat completion, waiting out short (per-minute) rate limits.
    A long (daily) rate limit, or too many waits, blocks solving for
    the rest of this run and is re-raised.
    """

    waits = 0

    while True:

        try:

            return get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                reasoning_effort="medium",
                response_format={"type": "json_object"},
            )

        except RateLimitError as e:

            retry_after = _retry_after_seconds(e)

            if (
                retry_after > MAX_RATE_LIMIT_WAIT_SECONDS
                or waits >= MAX_RATE_LIMIT_WAITS
            ):
                _block_solving(f"{type(e).__name__}: {e}")
                raise

            waits += 1

            print(
                f"  rate limited — waiting {retry_after:.0f}s "
                f"({waits}/{MAX_RATE_LIMIT_WAITS})"
            )

            time.sleep(retry_after)


def parse_answer(question: dict, response_text: str) -> dict:
    """
    Parse and validate one model reply into an answer record.
    Raises ValueError if the reply is unusable.
    """

    try:

        data = json.loads(response_text)

    except json.JSONDecodeError as e:

        raise ValueError(f"invalid JSON: {e}")

    if not isinstance(data, dict):
        raise ValueError("reply must be a JSON object")

    option_ids = [option["id"] for option in question["options"]]

    selected = data.get("correct_options")

    if not isinstance(selected, list) or not selected:
        raise ValueError("correct_options must be a non-empty list")

    unknown = [
        option_id
        for option_id in selected
        if option_id not in option_ids
    ]

    if unknown:
        raise ValueError(f"unknown option ids: {unknown}")

    # Dedup, in the question's own option order.
    selected = [
        option_id
        for option_id in option_ids
        if option_id in selected
    ]

    if (
        question["question_type"]["answer_format"] == "single_correct"
        and len(selected) != 1
    ):
        raise ValueError(
            f"single_correct question got {len(selected)} options"
        )

    confidence = data.get("confidence")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError(f"invalid confidence: {confidence!r}")

    reasoning = data.get("reasoning")

    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be a non-empty string")

    return {
        "question_id": question["id"],
        "correct_options": selected,
        "reasoning": reasoning.strip(),
        "confidence": float(confidence)
    }


def solve_question(
    question: dict,
    material_label: str,
    material: str
) -> dict:
    """
    Answer one question, retrying an empty/truncated/invalid reply
    MAX_ATTEMPTS times before raising.
    """

    if not question.get("options"):
        raise ValueError(f"{question['id']}: question has no options")

    user_prompt = build_user_prompt(
        question,
        material_label,
        material
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:

            response = _create_completion(user_prompt)

            choice = response.choices[0]

            if choice.finish_reason == "length":
                raise ValueError("reply was truncated")

            return parse_answer(
                question,
                choice.message.content or ""
            )

        except BadRequestError as e:

            # JSON mode rejects a reply that isn't valid JSON with a
            # 400 — retryable, unlike other bad requests.
            if "json_validate_failed" not in str(e):
                raise

            problem = "reply was not valid JSON"

        except ValueError as e:

            problem = str(e)

        print(f"  attempt {attempt}/{MAX_ATTEMPTS}: {problem}")

        if attempt < MAX_ATTEMPTS:
            time.sleep(SLEEP_BETWEEN_CALLS)

    raise RuntimeError(
        f"{question['id']}: no valid answer after {MAX_ATTEMPTS} attempts"
    )


# ================================================================
# ANSWERS FILE
# ================================================================

def get_answers_file_path(course: str, week) -> Path:

    return (
        Path(build_questions_file_path(course, week)).parent
        / "answers.json"
    )


def load_answers(answers_file: Path) -> list:

    if not answers_file.exists():
        return []

    return json.loads(
        answers_file.read_text(encoding="utf-8")
    )


def save_answers(answers_file: Path, answers: list) -> None:
    """
    Written to a temp file first, then renamed over answers.json
    (same as jobs.save_jobs), so a crash never truncates it.
    """

    tmp_file = answers_file.with_name(answers_file.name + ".tmp")

    tmp_file.write_text(
        json.dumps(answers, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    tmp_file.replace(answers_file)


# ================================================================
# ENTRY POINT
# ================================================================

def solve_assignment(course: str, week) -> bool:
    """
    Answer every question in course + week's classified
    questions.json into answers.json, next to it.

    Resumable: questions already in answers.json are skipped, and the
    file is saved after every answer, so a failure partway (e.g. the
    daily rate limit) loses nothing. A question that fails is left
    for the next run; a rate limit that can't be waited out blocks
    solving for the rest of this run (this job and every later one).

    Returns True once every question has an answer.
    """

    if is_solving_blocked():

        print(
            "\nSolving is blocked for the rest of this run — skipping."
        )

        return False

    questions_file = Path(build_questions_file_path(course, week))

    if not questions_file.exists():
        raise FileNotFoundError(
            f"Assignment not scraped yet: {questions_file}"
        )

    elements = json.loads(
        questions_file.read_text(encoding="utf-8")
    )["elements"]

    questions = [
        element
        for element in elements
        if element.get("element") == "question"
    ]

    passages = {
        element["id"]: element.get("passage") or ""
        for element in elements
        if element.get("element") == "passage"
    }

    unclassified = [
        question["id"]
        for question in questions
        if not question.get("question_type", {}).get("solving_strategy")
    ]

    if unclassified:
        raise ValueError(
            f"Questions not classified yet: {unclassified}"
        )

    answers_file = get_answers_file_path(course, week)

    answers = load_answers(answers_file)

    answered_ids = {answer["question_id"] for answer in answers}

    remaining = [
        question
        for question in questions
        if question["id"] not in answered_ids
    ]

    question_order = {
        question["id"]: index
        for index, question in enumerate(questions)
    }

    print("\n" + "=" * 80)
    print("SOLVING ASSIGNMENT")
    print("=" * 80)

    print(f"Questions: {len(questions)} ({len(remaining)} unanswered)")

    session = None

    for index, question in enumerate(remaining, start=1):

        strategy = question["question_type"]["solving_strategy"]

        print(f"\n[{index}/{len(remaining)}] {question['id']} ({strategy})")

        try:

            if strategy == "passage":

                passage = passages.get(question.get("passage_id"))

                if not passage:
                    raise ValueError(
                        f"passage '{question.get('passage_id')}' "
                        f"not found in questions.json"
                    )

                material_label, material = "PASSAGE", passage

            else:

                # Loaded once, only if some question needs retrieval.
                if session is None:
                    session = load_retrieval_session(course, week)

                material_label = "LECTURE EXCERPTS"

                material = retrieve_context_for_question(
                    session,
                    question
                )

            answer = solve_question(
                question,
                material_label,
                material
            )

        except RateLimitError:

            # Solving is now blocked for the rest of this run (see
            # _create_completion). Stop; answers so far are saved.
            break

        except Exception as e:

            print(f"  ✗ FAILED: {type(e).__name__}: {e}")

            continue

        answers.append(answer)

        answers.sort(
            key=lambda record: question_order.get(
                record["question_id"],
                len(questions)
            )
        )

        save_answers(answers_file, answers)

        print(
            f"  ✓ {answer['correct_options']} "
            f"(confidence {answer['confidence']:.2f})"
        )

        if index < len(remaining):
            time.sleep(SLEEP_BETWEEN_CALLS)

    answered_ids = {answer["question_id"] for answer in answers}

    unanswered = [
        question["id"]
        for question in questions
        if question["id"] not in answered_ids
    ]

    print(f"\nAnswers saved -> {answers_file}")

    if unanswered:
        print(f"Still unanswered: {unanswered}")

    return not unanswered

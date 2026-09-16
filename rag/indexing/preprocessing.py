import os
import re
import time
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_CHARS = 15000        # max raw chars per window sent to the model, overlap included
OVERLAP_CHARS = 2000     # raw chars repeated from the previous window, as context only
SLEEP_BETWEEN_CALLS = 2  # seconds, be gentle on TPM rate limits
MAX_ATTEMPTS = 3         # per chunk, when the model returns empty or truncated output

_client = None


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

SYSTEM_PROMPT = """
You are cleaning an ASR-generated lecture transcript for a RAG knowledge base.
The cleaned output will later be split into retrieval chunks of roughly 180 words
each (hard cap 260 words), so the header structure you produce directly controls
chunk quality. Follow the rules below exactly.

Rules:
1. Fix disfluencies, filler words, false starts, and run-on sentences. Restore punctuation.
2. Fix garbled numbers/technical notation (e.g. "10 to the ^ of 26" → "10^26 FLOPs").
3. Do NOT summarize, condense, or drop any substantive claim, example, name, or number.
   Every fact in the raw transcript must still be present somewhere in the output.
4. Use two levels of Markdown header:
   - "## [topic label]" for a broad topic/theme — insert one whenever the lecture
     shifts to a genuinely new subject, sub-argument, paper, framework, or example.
   - "### [sub-topic label]" for a chunk boundary *within* that topic. Insert a new
     "### " subheader at least every ~150-200 words of body text, even if the topic
     hasn't changed, at the nearest natural break (end of an example, end of an
     argument, a new name/date/case being introduced). NEVER let more than ~220
     words of prose accumulate under one header without a "### " break.
5. Each "### " subsection should be self-contained enough to be read on its own:
   if it relies on a pronoun or "this"/"that" referring to something introduced more
   than a paragraph earlier, briefly restate what it refers to instead of assuming
   the reader has that context.
6. Keep each Markdown table or bullet/numbered list intact under a single header —
   never split one across a "### " boundary. If a list is long, give it its own
   "### " subheader rather than folding it into surrounding prose.
7. You will sometimes be shown the previous chunk's most recent cleaned subsection
   for context. If this new chunk continues that same topic, keep using "### "
   subheaders under it per rule 4 (do not repeat the "## " header or its content) —
   only start a new "## " header once the topic has genuinely moved on.
Output only the cleaned, headered Markdown. No preamble, no commentary, no annotations in headers.
REMEMBER: do not drop any facts, names, numbers, information.
"""


# ---------------------------------------------------------------------------
# Character chunk splitter
# ---------------------------------------------------------------------------

def split_into_chunks(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS
) -> list[tuple[str, str]]:
    """Splits text into windows of at most max_chars, each starting with the
    previous window's last overlap_chars. Returns (overlap, new_text) pairs so
    the overlap is shown to the model as context but never cleaned twice."""

    chunks: list[tuple[str, str]] = []
    new_start = 0

    while new_start < len(text):
        window_start = max(new_start - overlap_chars, 0)
        window_end = min(window_start + max_chars, len(text))
        chunks.append((text[window_start:new_start], text[new_start:window_end]))
        new_start = window_end

    return chunks


# ---------------------------------------------------------------------------
# Last-section extraction (for continuation context)
# ---------------------------------------------------------------------------

def get_last_section(cleaned_text: str) -> str:
    """Returns the most recent '## ...' or '### ...' section (header + body)
    from cleaned markdown, so continuation context stays chunk-sized instead of
    dragging along an entire topic's worth of prior subsections. If there's no
    header at all, returns the whole text."""
    matches = list(re.finditer(r'^#{2,3} .*$', cleaned_text, flags=re.MULTILINE))
    if not matches:
        return cleaned_text
    return cleaned_text[matches[-1].start():]


# ---------------------------------------------------------------------------
# LLM cleaning call
# ---------------------------------------------------------------------------

def clean_chunk(chunk: str, last_section: str = "", overlap: str = "") -> tuple[str, dict]:
    overlap_note = ""
    if overlap:
        overlap_note = (
            f"The text below overlaps with the end of the previous chunk and has "
            f"already been cleaned. Use it ONLY as context for where this chunk "
            f"starts — do NOT clean it or include it in your output:\n"
            f"\"\"\"\n{overlap}\n\"\"\"\n\n"
        )

    context_note = ""
    if last_section:
        context_note = (
            f"\n\nThe previous chunk's final section (already cleaned) was:\n"
            f"\"\"\"\n{last_section}\n\"\"\"\n"
            f"If this new chunk continues that same topic, continue writing "
            f"under that section — do NOT repeat its header or its content, "
            f"just continue the prose as if it flows on. Only start a new "
            f"'## ' header once the topic genuinely shifts."
        )

    user_content = f"{overlap_note}Transcript chunk:\n{chunk}{context_note}"

    for attempt in range(1, MAX_ATTEMPTS + 1):

        response = get_client().chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            reasoning_effort="medium",
            temperature=0,
        )

        choice = response.choices[0]
        cleaned = choice.message.content or ""

        # Reasoning models can burn the whole completion budget thinking and
        # return empty/cut-off content; accepting it silently drops the chunk.
        if cleaned.strip() and choice.finish_reason != "length":
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            }
            return cleaned, usage

        problem = "truncated" if choice.finish_reason == "length" else "empty"
        print(f"  attempt {attempt}/{MAX_ATTEMPTS}: model returned {problem} output")

        if attempt < MAX_ATTEMPTS:
            time.sleep(SLEEP_BETWEEN_CALLS)

    raise RuntimeError(
        f"Model returned empty/truncated output after {MAX_ATTEMPTS} attempts"
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def clean_transcript(text: str, max_chars: int = MAX_CHARS) -> str:
    chunks = split_into_chunks(text, max_chars=max_chars)
    print(f"Split into {len(chunks)} chunk(s)\n")

    cleaned_pieces: list[str] = []
    last_section = ""
    total_tokens = 0

    for i, (overlap, chunk) in enumerate(chunks):
        print(f"Cleaning chunk {i+1}/{len(chunks)} "
              f"({len(chunk):,} new chars + {len(overlap):,} overlap, "
              f"context: {len(last_section)} chars)...")

        cleaned, usage = clean_chunk(chunk, last_section, overlap)

        if usage["total_tokens"]:
            total_tokens += usage["total_tokens"]
            print(f"  tokens used: {usage['total_tokens']:,} "
                  f"(prompt {usage['prompt_tokens']:,}, completion {usage['completion_tokens']:,})")

        if cleaned.lstrip().startswith("##") or not cleaned_pieces:
            # model started a fresh section (or this is the very first chunk)
            cleaned_pieces.append(cleaned)
        else:
            # model continued the previous section — merge without a header repeat
            cleaned_pieces[-1] = cleaned_pieces[-1].rstrip() + "\n" + cleaned.lstrip()

        last_section = get_last_section(cleaned_pieces[-1])

        if i < len(chunks) - 1:
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nTotal tokens used across all chunks: {total_tokens:,}")
    return "\n\n".join(cleaned_pieces)

import os
import re
import time
from groq import Groq

from rag.config import BASE_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_FILE = BASE_DIR / "sample.txt"
OUTPUT_FILE = BASE_DIR / "sample_cleaned.txt"

MAX_CHARS = 15000        # max raw chars per chunk sent to the model
OVERLAP_SENTENCES = 0    # not used for raw overlap anymore — continuation is handled via last_section
SLEEP_BETWEEN_CALLS = 2  # seconds, be gentle on TPM rate limits

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
     argument, a new name/date/case being introduced). Never let more than ~220
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
REMEMBER: hard cap 260 words
"""


# ---------------------------------------------------------------------------
# Sentence-aware chunk splitter
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — splits on '.', '?', '!' followed by
    whitespace and a capital letter/quote, while trying not to break on
    common abbreviations. Good enough for cleaning purposes, not
    linguistically perfect."""
    protected = re.sub(
        r'\b(Mr|Mrs|Ms|Dr|Prof|vs|etc|e\.g|i\.e|U\.S|U\.K)\.',
        lambda m: m.group(0).replace('.', '<DOT>'),
        text,
    )
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', protected)
    return [s.replace('<DOT>', '.').strip() for s in raw_sentences if s.strip()]


def split_into_chunks(text: str, max_chars: int = 15000, overlap_sentences: int = 0) -> list[str]:
    """Splits text into chunks of up to max_chars, breaking only at sentence
    boundaries. overlap_sentences repeats the last N sentences of a chunk at
    the start of the next one (set to 0 if using last-section continuation
    instead of raw overlap)."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_len = len(sentence) + 1

        if current_len + sentence_len > max_chars and current:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(overlap)
            current_len = sum(len(s) + 1 for s in current)
            continue

        current.append(sentence)
        current_len += sentence_len
        i += 1

    if current:
        chunks.append(" ".join(current))

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

def clean_chunk(chunk: str, last_section: str = "") -> tuple[str, dict]:
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

    user_content = f"Transcript chunk:\n{chunk}{context_note}"

    response = get_client().chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        reasoning_effort="medium",
        temperature=0,
    )

    cleaned = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
        "total_tokens": response.usage.total_tokens if response.usage else None,
    }
    return cleaned, usage


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def clean_transcript(text: str, max_chars: int = MAX_CHARS) -> str:
    chunks = split_into_chunks(text, max_chars=max_chars, overlap_sentences=OVERLAP_SENTENCES)
    print(f"Split into {len(chunks)} chunk(s)\n")

    cleaned_pieces: list[str] = []
    last_section = ""
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        print(f"Cleaning chunk {i+1}/{len(chunks)} "
              f"({len(chunk):,} chars, context: {len(last_section)} chars)...")

        cleaned, usage = clean_chunk(chunk, last_section)

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

        # basic sanity check: warn if a chunk's cleaned output looks over-summarized
        ratio = len(cleaned) / max(len(chunk), 1)
        if ratio < 0.5:
            print(f"  ⚠️  cleaned output is only {ratio:.0%} of input length — check for over-summarization")

        if i < len(chunks) - 1:
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nTotal tokens used across all chunks: {total_tokens:,}")
    return "\n\n".join(cleaned_pieces)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    cleaned_text = clean_transcript(raw_text)

    OUTPUT_FILE.write_text(cleaned_text, encoding="utf-8")

    print(f"\nSaved cleaned transcript to: {OUTPUT_FILE}")
    print(f"Input characters:  {len(raw_text):,}")
    print(f"Output characters: {len(cleaned_text):,}")

import re
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

from rag.config import MAX_CHUNK_WORDS, OVERLAP_WORDS


# ================================================================
# CONFIG
# ================================================================

# Matches the "## topic" / "### subtopic" structure that the LLM
# cleaning step (rag/indexing/preprocessing.py) writes into every
# transcript.
HEADERS_TO_SPLIT_ON = [
    ("##", "topic"),
    ("###", "subtopic"),
]

# The run of "## " / "### " lines a header-split section starts with.
_LEADING_HEADINGS = re.compile(r"\A(?:#{2,3} [^\n]*(?:\n|\Z))+")


# ================================================================
# HELPERS
# ================================================================

def _word_count(text: str) -> int:

    return len(text.split())


def _leading_heading(section_text: str) -> str:

    match = _LEADING_HEADINGS.match(section_text)

    if not match:
        return ""

    return "\n".join(
        line.strip()
        for line in match.group().splitlines()
    )


# ================================================================
# CHUNK A CLEANED TRANSCRIPT
# ================================================================

def chunk_cleaned_transcript(cleaned_text: str) -> list[dict]:
    """
    Split an LLM-cleaned, Markdown-headered transcript into retrieval
    chunks.

    The cleaning step already breaks the transcript into "## topic" /
    "### subtopic" sections sized to roughly TARGET_CHUNK_WORDS words
    each, so splitting on those headers gives chunks that are already
    close to the right size and topically self-contained. Any section
    that still comes out longer than MAX_CHUNK_WORDS (the cleaning
    prompt's hard cap) gets a secondary word-based split with overlap,
    so nothing downstream ever receives an oversized chunk.

    When a section is split, its heading line(s) are repeated at the
    top of every later piece, so each chunk is embedded with its topic
    — the word cap is reduced by the heading's length to leave room.
    """

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False
    )

    sections = header_splitter.split_text(cleaned_text)

    chunks = []

    for section in sections:

        heading = _leading_heading(section.page_content)

        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_WORDS - _word_count(heading),
            chunk_overlap=OVERLAP_WORDS,
            length_function=_word_count,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        pieces = [
            piece.strip()
            for piece in size_splitter.split_text(section.page_content)
        ]

        pieces = [piece for piece in pieces if piece]

        for index, piece in enumerate(pieces):

            # Overlap from a very short first piece can already carry
            # the heading into the next one — don't add it twice.
            if (
                heading
                and index > 0
                and not piece.startswith(heading.splitlines()[0])
            ):

                piece = f"{heading}\n{piece}"

            chunks.append({
                "text": piece,
                "topic": section.metadata.get("topic"),
                "subtopic": section.metadata.get("subtopic"),
                "word_count": _word_count(piece)
            })

    return chunks


# ================================================================
# CHUNK A CLEANED TRANSCRIPT FILE
# ================================================================

def chunk_cleaned_transcript_file(cleaned_transcript_file) -> list[dict]:

    cleaned_text = Path(
        cleaned_transcript_file
    ).read_text(encoding="utf-8")

    return chunk_cleaned_transcript(cleaned_text)

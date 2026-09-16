import io
import os
import base64
import math
import mimetypes

from groq import Groq
from PIL import Image, UnidentifiedImageError


# ================================================================
# CONFIG
# ================================================================

MODEL = "qwen/qwen3.8-27b"

MAX_COMPLETION_TOKENS = 1000

TEMPERATURE = 0


# ================================================================
# IMAGE
# ================================================================

MAX_ASPECT_RATIO = 2.0


def _pad_wide_image(image_bytes):
    """
    Return PNG bytes of the image padded with white (top and bottom,
    original centred) to MAX_ASPECT_RATIO, or None if the image is
    not wider than that or can't be decoded (sent as-is then).

    Very wide, short images — e.g. one line of question text — made
    qwen/qwen3.8-27b invent "enlarged crops" and misread symbols (α as
    "a"); padding to 2:1 fixed both in testing and halves image tokens.
    """

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size

            if width <= height * MAX_ASPECT_RATIO:
                return None

            rgba = image.convert("RGBA")

    except (UnidentifiedImageError, OSError):
        return None

    canvas = Image.new(
        "RGB",
        (width, math.ceil(width / MAX_ASPECT_RATIO)),
        "white"
    )

    canvas.paste(
        rgba,
        (0, (canvas.height - height) // 2),
        rgba
    )

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")

    return buffer.getvalue()


def encode_image(image_path):
    """
    Read the complete image from disk and return:

        (mime_type, base64_data)

    Images wider than MAX_ASPECT_RATIO are padded first and sent as PNG.
    """

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(
            f"Could not determine a valid image MIME type "
            f"for: {image_path}"
        )

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_bytes = image_file.read()

    if not image_bytes:
        raise ValueError(
            f"Image file is empty: {image_path}"
        )

    padded_bytes = _pad_wide_image(image_bytes)

    if padded_bytes is not None:

        image_bytes = padded_bytes
        mime_type = "image/png"

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return mime_type, encoded_image


# ================================================================
# PROMPT
# ================================================================

def build_prompt():
    """
    Prompt specifically designed for faithful image extraction.

    The model should describe only what is actually inside the
    supplied image and should not infer content from the UI in
    which the image may be displayed.
    """

    return """
You are a visual transcription system for a blind person.

INPUT
- One image
- Surrounding question/context, if provided

TASK
Describe the supplied image so that a blind person can understand all
information in it that is relevant to the question.

The image is the ONLY visual source of truth.

OUTPUT
Return ONLY the final transcription/description of the image.

- For text-based images, transcribe the visible text directly.
- For questions, give the complete visible question/problem.
- For graphs, tables, diagrams, flowcharts, or figures, describe all
  visible information needed to understand them, including labels,
  values, axes, legends, connections, and relationships.
- Preserve numbers, symbols, formulas, and mathematical notation exactly.
- Be detailed where the visual contains information that cannot be
  represented by ordinary text, but do not add unnecessary commentary.

STRICT ACCURACY RULES
- Describe ONLY what is visibly present in the image.
- Do NOT infer, calculate, estimate, or assume anything that is not visible.
- Do NOT use outside knowledge to fill gaps.
- Do NOT guess unclear or cropped information.
- If something cannot be determined from the image, say so.
- Treat the input as ONE image. Do not describe multiple images, crops,
  panels, or duplicate sections unless they are actually visible as
  separate parts of the supplied image.
- Do NOT describe the surrounding UI, browser, viewer, desktop, or interface.
- Do NOT explain your reasoning or thought process.
- NEVER output <think>...</think>.
- Do NOT solve the question.

Return only the completed description.
""".strip()


# ================================================================
# GROQ
# ================================================================

def get_client():
    """Create the Groq client from GROQ_API_KEY."""

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


def describe_image(
    client,
    image_path
):
    """
    Send the complete image to the vision model and return
    the response object.
    """

    mime_type, base64_image = encode_image(
        image_path
    )

    image_data_url = (
        f"data:{mime_type};base64,{base64_image}"
    )

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "user",

                "content": [

                    {
                        "type": "text",

                        "text": build_prompt()
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],

        temperature=TEMPERATURE,

        max_completion_tokens=MAX_COMPLETION_TOKENS,
        
        reasoning_effort="none",

        stream=False
    )

    return response


# ================================================================
# MAIN
# ================================================================

def image_to_text(image_path):
    """
    Return the vision model's text description of the image at
    image_path.
    """

    response = describe_image(
        client=get_client(),
        image_path=image_path
    )

    return response.choices[0].message.content


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import sys
    print(image_to_text(sys.argv[1]))

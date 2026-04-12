"""
extractors/claude_extractor.py — Claude Vision API wrapper.

Sends the form image to claude-sonnet-4 with a structured extraction prompt
and parses the JSON response into an ExtractionResult.

All HTTP communication goes through ``_call_claude_api()`` — a single
choke-point that can be mocked in tests or replaced with another provider
without touching any other code.
"""

import base64
import json
import logging
import re
from pathlib import Path

import anthropic

from config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL, REQUEST_TIMEOUT_SECONDS
from extractors.base_extractor import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

_BLANK = "[BLANK]"
_UNREADABLE = "[UNREADABLE]"
_NEEDS_REVIEW = "[NEEDS_REVIEW]"

# ── Validation helpers (same as Tesseract extractor) ─────────────────────────

_MOBILE_RE = re.compile(r"^[6-9]\d{9}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are an expert data-entry assistant. Carefully read the handwritten application form in the image and extract the following fields exactly as written by the applicant.  # noqa: E501

Return ONLY a valid JSON object with these exact keys (no markdown, no extra text):

{
  "app_no": "...",
  "name": "...",
  "address": "...",
  "mob_no": "...",
  "email": "...",
  "q1_chanted_before": "Yes | No | NEEDS_REVIEW",
  "q2_familiarity_level": (
    "Can chant confidently without guidance | Can chant but need occasional guidance"
    " | Know partially and need training | Beginner-need full training | NEEDS_REVIEW"
  ),
  "q3_formal_training": "Yes | No | NEEDS_REVIEW",
  "q4_training_mode": "In-person | Online | Either | NEEDS_REVIEW",
  "q5_preferred_language": "Kannada | Sanskrit | Other Language | NEEDS_REVIEW",
  "q6_comments": "..."
}

Rules:
- If a field is empty or not filled in, use "[BLANK]".
- If a field is filled but completely illegible, use "[UNREADABLE]".
- For checkbox questions, use the exact option label for whichever box is ticked.
- For mob_no, return digits only (no spaces, dashes, country code).
- For address, collapse multi-line text into a single string separated by
  commas.
- Do NOT add any explanation outside the JSON object.
"""


# ── Internal API wrapper ──────────────────────────────────────────────────────


def _image_to_base64(image_path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for *image_path*.

    Raises:
        OSError: if the file cannot be read.
        ValueError: if the file extension is not a supported image type.
    """
    ext = image_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext)
    if media_type is None:
        raise ValueError(f"Unsupported image extension: {ext!r}")

    with open(image_path, "rb") as fh:
        encoded = base64.standard_b64encode(fh.read()).decode("utf-8")
    return encoded, media_type


def _call_claude_api(api_key: str, image_path: Path) -> str:
    """Send image + prompt to Claude and return the raw text response.

    This is the ONLY function that touches the Anthropic SDK.
    Swap or mock this function to change the underlying AI provider.

    Raises:
        anthropic.APIError: on any API-level error.
        ValueError: for unsupported image types.
        OSError: if the image file cannot be read.
    """
    img_b64, media_type = _image_to_base64(image_path)

    client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": _EXTRACTION_PROMPT},
                ],
            }
        ],
    )

    # response.content is a list of ContentBlock objects; take the first text block.
    text_blocks = [blk for blk in response.content if blk.type == "text"]
    if not text_blocks:
        raise ValueError("Claude returned no text content blocks.")
    return text_blocks[0].text


# ── JSON parsing ──────────────────────────────────────────────────────────────


def _parse_claude_response(raw_text: str) -> dict:
    """Parse Claude's JSON response into a fields dict.

    Claude sometimes wraps JSON in markdown fences despite instructions.
    Strip them before parsing.
    """
    # Strip markdown code fences.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE).strip()
    # Sometimes there's trailing ``` on its own line.
    cleaned = cleaned.rstrip("`").strip()
    return json.loads(cleaned)


# ── Confidence scoring ────────────────────────────────────────────────────────


def _compute_claude_confidences(fields: dict) -> dict:
    """Heuristic confidence scores for Claude-extracted fields.

    Claude does not emit numeric confidence values, so we derive them from
    format validation and field completeness.
    """
    confs: dict[str, float] = {}

    for field, value in fields.items():
        if value in (_BLANK, _UNREADABLE, _NEEDS_REVIEW, "", None):
            confs[field] = 0.0 if value == _UNREADABLE else 50.0
            continue

        if field == "app_no":
            confs[field] = 95.0 if str(value).isdigit() else 75.0

        elif field == "mob_no":
            digits = re.sub(r"\D", "", str(value))
            confs[field] = 90.0 if _MOBILE_RE.match(digits) else 60.0

        elif field == "email":
            confs[field] = 88.0 if _EMAIL_RE.match(str(value)) else 55.0

        elif field == "name":
            confs[field] = 85.0 if len(str(value)) >= 3 else 50.0

        elif field == "address":
            confs[field] = 80.0 if len(str(value)) >= 8 else 50.0

        elif field.startswith("q") and value not in (_NEEDS_REVIEW,):
            # Claude is much better at checkboxes — give high confidence.
            confs[field] = 88.0

        else:
            confs[field] = 70.0

    return {k: round(v, 1) for k, v in confs.items()}


# ── Extractor class ───────────────────────────────────────────────────────────


class ClaudeExtractor(BaseExtractor):
    """Claude Vision extraction engine.

    Args:
        api_key: Anthropic API key (passed from the UI; never stored on disk).
    """

    def __init__(self, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("ClaudeExtractor requires a non-empty API key.")
        self._api_key = api_key.strip()

    @property
    def engine_name(self) -> str:
        return "claude"

    def health_check(self) -> bool:
        """Return True when the Anthropic SDK is importable and a key is set."""
        try:
            import anthropic as _  # noqa: F401
            return bool(self._api_key)
        except ImportError:
            logger.error("anthropic SDK not installed.  Run: pip install anthropic")
            return False

    def extract(self, image_path: Path) -> ExtractionResult:
        """Call Claude Vision and return a structured ExtractionResult."""
        logger.info("[Claude] Processing: %s", image_path.name)
        try:
            raw_response = _call_claude_api(self._api_key, image_path)
            logger.debug("[Claude] Raw response:\n%s", raw_response[:600])
        except (anthropic.APIError, OSError, ValueError) as exc:
            logger.error("[Claude] API call failed for %s: %s", image_path.name, exc)
            return ExtractionResult(
                fields={}, confidences={}, engine=self.engine_name,
                error=f"API error: {exc}"
            )

        try:
            fields = _parse_claude_response(raw_response)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("[Claude] JSON parse failed for %s: %s", image_path.name, exc)
            return ExtractionResult(
                fields={}, confidences={}, engine=self.engine_name,
                raw_text=raw_response,
                error=f"JSON parse error: {exc}"
            )

        # Sanitise mob_no: strip non-digits after Claude might include spaces.
        if "mob_no" in fields and fields["mob_no"] not in (_BLANK, _UNREADABLE):
            fields["mob_no"] = re.sub(r"\D", "", str(fields["mob_no"]))

        confidences = _compute_claude_confidences(fields)

        result = ExtractionResult(
            fields=fields,
            confidences=confidences,
            engine=self.engine_name,
            raw_text=raw_response,
        )
        logger.info(
            "[Claude] Done: %s | avg_text_conf=%.1f",
            image_path.name,
            result.avg_text_field_confidence,
        )
        return result

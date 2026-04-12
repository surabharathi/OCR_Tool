"""
extractors/tesseract_extractor.py — Local Tesseract OCR engine wrapper.

Preprocessing pipeline:
  1. Convert to grayscale.
  2. Boost contrast (Pillow ImageEnhance).
  3. Upscale if the long edge is below IMAGE_MIN_LONG_EDGE (helps Tesseract).
  4. Run pytesseract twice:
       a. image_to_data()  → per-word confidence scores.
       b. image_to_string() → clean text for regex parsing.

Field extraction uses regex anchored to the printed label text.
Checkbox extraction is best-effort; Claude handles corrections.
"""

import logging
import re
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from config import IMAGE_CONTRAST_FACTOR, IMAGE_MIN_LONG_EDGE, TESSERACT_LANG, TESSERACT_PSM
from extractors.base_extractor import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_BLANK = "[BLANK]"
_UNREADABLE = "[UNREADABLE]"
_NEEDS_REVIEW = "[NEEDS_REVIEW]"

# Fields considered "text fields" for confidence gating.
_TEXT_FIELDS = ("app_no", "name", "address", "mob_no", "email")

# Validation: Indian mobile numbers start with 6-9 and have 10 digits.
_MOBILE_RE = re.compile(r"^[6-9]\d{9}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ── Image preprocessing ───────────────────────────────────────────────────────


def _preprocess(image_path: Path) -> Image.Image:
    """Return a pre-processed PIL image optimised for Tesseract."""
    img = Image.open(image_path).convert("RGB")

    # Upscale if too small — Tesseract performs poorly below ~150 DPI.
    w, h = img.size
    long_edge = max(w, h)
    if long_edge < IMAGE_MIN_LONG_EDGE:
        scale = IMAGE_MIN_LONG_EDGE / long_edge
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug("Upscaled image from %dx%d to %dx%d.", w, h, new_w, new_h)

    img = img.convert("L")                          # grayscale
    img = ImageEnhance.Contrast(img).enhance(IMAGE_CONTRAST_FACTOR)
    img = img.filter(ImageFilter.SHARPEN)
    return img


# ── Field parsing ─────────────────────────────────────────────────────────────


def _clean(text: str) -> str:
    """Collapse whitespace; strip leading/trailing noise."""
    return re.sub(r"\s+", " ", text).strip()


def _parse_fields(raw_text: str) -> dict:
    """Extract structured fields from Tesseract's raw text using regex."""
    # Normalise line-internal whitespace while preserving newlines.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw_text.splitlines()]
    text = "\n".join(lines)

    fields: dict[str, str] = {}

    # ── Application Number ────────────────────────────────────────────────────
    m = re.search(r"Application\s+No\.?\s*[:\.]?\s*(\S+)", text, re.IGNORECASE)
    fields["app_no"] = m.group(1).strip() if m else _UNREADABLE

    # ── Name ─────────────────────────────────────────────────────────────────
    # Everything after "Name :" up to the next "Address" label.
    m = re.search(
        r"Name\s*[:\.]?\s*(.+?)(?=\n\s*Address|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        name_val = _clean(m.group(1))
        fields["name"] = name_val if len(name_val) >= 2 else _UNREADABLE
    else:
        fields["name"] = _UNREADABLE

    # ── Address (multi-line) ──────────────────────────────────────────────────
    m = re.search(
        r"Address\s*[:\.]?\s*(.+?)(?=\n\s*Mob\.|\n\s*Mobile|\n\s*Phone|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        addr_val = _clean(m.group(1))
        fields["address"] = addr_val if len(addr_val) >= 5 else _UNREADABLE
    else:
        fields["address"] = _UNREADABLE

    # ── Mobile Number ─────────────────────────────────────────────────────────
    m = re.search(
        r"Mob\.?\s*No\.?\s*[:\.]?\s*([\d\s\+\-]{8,18})",
        text, re.IGNORECASE
    )
    if m:
        mob_raw = re.sub(r"[\s\-\+]", "", m.group(1))
        # Strip leading country code if present (e.g. 91 prefix).
        if len(mob_raw) == 12 and mob_raw.startswith("91"):
            mob_raw = mob_raw[2:]
        fields["mob_no"] = mob_raw if mob_raw else _UNREADABLE
    else:
        fields["mob_no"] = _UNREADABLE

    # ── Email ─────────────────────────────────────────────────────────────────
    m = re.search(r"[Ee]mail\s*[:\.]?\s*(\S+@\S+)", text)
    if m:
        fields["email"] = m.group(1).strip()
    else:
        # Secondary pass: look for any email-like token in the full text.
        m2 = re.search(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", text)
        fields["email"] = m2.group(0).strip() if m2 else _BLANK

    # ── Checkbox Q1: Have you chanted Lalitha Sahasranamam before? ─────────────
    fields["q1_chanted_before"] = _extract_yes_no(text, r"chanted.*?earlier", after_offset=200)

    # ── Checkbox Q2: Familiarity level ───────────────────────────────────────
    familiarity_options = [
        "Can chant confidently without guidance",
        "Can chant but need occasional guidance",
        "Know partially and need training",
        "Beginner-need full training",
    ]
    fields["q2_familiarity_level"] = _extract_checked_option(text, familiarity_options)

    # ── Checkbox Q3: Formal training required? ────────────────────────────────
    fields["q3_formal_training"] = _extract_yes_no(
        text, r"require formal training", after_offset=100
    )

    # ── Checkbox Q4: Training mode ────────────────────────────────────────────
    fields["q4_training_mode"] = _extract_checked_option(
        text, ["In-person", "Online", "Either"]
    )

    # ── Checkbox Q5: Preferred language ──────────────────────────────────────
    fields["q5_preferred_language"] = _extract_checked_option(
        text, ["Kannada", "Sanskrit", "Other Language"]
    )

    # ── Q6: Comments (optional free text) ────────────────────────────────────
    m = re.search(
        r"Comments\s*/\s*Expectations.*?[:\-]?\s*(.+?)(?=\n\n|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        comment_val = _clean(m.group(1))
        fields["q6_comments"] = comment_val if len(comment_val) > 2 else _BLANK
    else:
        fields["q6_comments"] = _BLANK

    return fields


def _extract_yes_no(text: str, question_pattern: str, after_offset: int = 150) -> str:
    """Locate 'Yes'/'No' near a question and guess which is ticked."""
    m = re.search(question_pattern, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return _NEEDS_REVIEW
    segment = text[m.end(): m.end() + after_offset]
    # Tesseract may render a tick as V, v, \/, ✓, ☑
    tick = re.compile(r"[Vv✓☑✔]")
    yes_pos = _find_word_pos(segment, "yes")
    no_pos = _find_word_pos(segment, "no")
    if yes_pos is None and no_pos is None:
        return _NEEDS_REVIEW
    # Find closest tick to each option.
    ticks = [m2.start() for m2 in tick.finditer(segment)]
    if not ticks:
        return _NEEDS_REVIEW

    def _closest_tick(pos: int | None) -> int:
        if pos is None:
            return 9999
        return min(abs(t - pos) for t in ticks)
    if _closest_tick(yes_pos) < _closest_tick(no_pos):
        return "Yes"
    if _closest_tick(no_pos) < _closest_tick(yes_pos):
        return "No"
    return _NEEDS_REVIEW


def _find_word_pos(text: str, word: str) -> int | None:
    m = re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)
    return m.start() if m else None


def _extract_checked_option(text: str, options: list[str]) -> str:
    """Return the option whose label appears closest to a tick character."""
    tick = re.compile(r"[Vv✓☑✔]")
    ticks = [m.start() for m in tick.finditer(text)]
    if not ticks:
        return _NEEDS_REVIEW
    best_option: str | None = None
    best_distance = 9999
    for option in options:
        m = re.search(re.escape(option[:10]), text, re.IGNORECASE)
        if not m:
            continue
        dist = min(abs(t - m.start()) for t in ticks)
        if dist < best_distance:
            best_distance = dist
            best_option = option
    # Only trust the match if the tick is within 80 characters.
    return best_option if (best_option and best_distance <= 80) else _NEEDS_REVIEW


# ── Confidence scoring ────────────────────────────────────────────────────────


def _compute_confidences(fields: dict, word_confs: list[float]) -> dict:
    """Compute per-field confidence scores.

    Strategy:
    • Overall Tesseract average is the baseline.
    • Format-validation bonus/penalty is applied per text field.
    • Checkbox fields get a fixed low score to signal they need review.
    """
    overall = sum(word_confs) / len(word_confs) if word_confs else 0.0
    confs: dict[str, float] = {}

    for field, value in fields.items():
        if value in (_BLANK, _UNREADABLE, _NEEDS_REVIEW):
            confs[field] = 0.0 if value == _UNREADABLE else 50.0
            continue

        if field == "app_no":
            # Application numbers are printed — should be very high confidence.
            confs[field] = 95.0 if value.isdigit() else overall * 0.8

        elif field == "mob_no":
            confs[field] = 90.0 if _MOBILE_RE.match(value) else min(overall, 60.0)

        elif field == "email":
            confs[field] = 85.0 if _EMAIL_RE.match(value) else min(overall, 55.0)

        elif field in ("name", "address"):
            # Length heuristic: very short values are likely garbled.
            length_ok = len(value) >= (3 if field == "name" else 8)
            confs[field] = overall if length_ok else overall * 0.6

        else:
            # Checkbox fields — Tesseract is unreliable here.
            confs[field] = 40.0

    return {k: round(v, 1) for k, v in confs.items()}


# ── Extractor class ───────────────────────────────────────────────────────────


class TesseractExtractor(BaseExtractor):
    """Tesseract 5 OCR engine with structured-form field parsing."""

    @property
    def engine_name(self) -> str:
        return "tesseract"

    def health_check(self) -> bool:
        """Return True when Tesseract binary is available."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract binary not found.  Install tesseract-ocr.")
            return False

    def extract(self, image_path: Path) -> ExtractionResult:
        """Run Tesseract on *image_path* and return structured extraction result."""
        logger.info("[Tesseract] Processing: %s", image_path.name)
        try:
            img = _preprocess(image_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("[Tesseract] Preprocessing failed for %s: %s", image_path.name, exc)
            return ExtractionResult(
                fields={}, confidences={}, engine=self.engine_name,
                error=f"Preprocessing failed: {exc}"
            )

        config_str = f"--psm {TESSERACT_PSM} --oem 3 -l {TESSERACT_LANG}"

        try:
            # Full text for regex parsing.
            raw_text: str = pytesseract.image_to_string(img, config=config_str)
            logger.debug("[Tesseract] Raw text (%d chars):\n%s", len(raw_text), raw_text[:400])

            # Word-level data for confidence scores.
            data = pytesseract.image_to_data(
                img, config=config_str, output_type=pytesseract.Output.DICT
            )
            word_confs: list[float] = [
                float(c) for c in data["conf"]
                if isinstance(c, (int, float)) and c >= 0
            ]
        except pytesseract.TesseractError as exc:
            logger.error("[Tesseract] OCR failed for %s: %s", image_path.name, exc)
            return ExtractionResult(
                fields={}, confidences={}, engine=self.engine_name,
                error=f"Tesseract error: {exc}"
            )

        fields = _parse_fields(raw_text)
        confidences = _compute_confidences(fields, word_confs)

        result = ExtractionResult(
            fields=fields,
            confidences=confidences,
            engine=self.engine_name,
            raw_text=raw_text,
        )
        logger.info(
            "[Tesseract] Done: %s | avg_text_conf=%.1f",
            image_path.name,
            result.avg_text_field_confidence,
        )
        return result

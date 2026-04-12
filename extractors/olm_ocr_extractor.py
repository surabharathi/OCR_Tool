"""
extractors/olm_ocr_extractor.py — Local VLM extractor via Ollama.

Supports any vision-capable Ollama model.  Recommended models for this
project (pull before first use):

    ollama pull reducto/rolmocr      # OCR-specialised, best for handwriting
    ollama pull allenai/olmocr2      # Checkbox-aware semantic tagging

Both are fine-tunes of Qwen2.5-VL-7B and run comfortably on the RTX 4090's
16 GB VRAM at full precision — no quantisation needed.

This extractor is the SOLE engine in the simplified single-phase pipeline.
All API communication is isolated in ``_call_ollama()`` so the underlying
model or runtime can be swapped without touching any other module.
"""

import base64
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract
import ollama

from config import (
    BLANK_CONFIDENCE,
    ENABLE_IMAGE_ENHANCEMENT,
    ENABLE_ORIENTATION_CORRECTION,
    FIELD_VALIDATION_CONFIDENCE,
    NEEDS_REVIEW_CONFIDENCE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    TESSERACT_CMD,
    UNREADABLE_CONFIDENCE,
)

# Configure pytesseract
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
from extractors.base_extractor import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# ── Sentinels ─────────────────────────────────────────────────────────────────
_BLANK = "[BLANK]"
_UNREADABLE = "[UNREADABLE]"
_NEEDS_REVIEW = "[NEEDS_REVIEW]"

# ── Validation patterns ───────────────────────────────────────────────────────
_MOBILE_RE = re.compile(r"^[6-9]\d{9}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ── Image preprocessing ──────────────────────────────────────────────────────


def _preprocess_image(image_path: Path) -> Image.Image:
    """Load and preprocess image: detect/correct orientation, enhance for OCR.

    Returns:
        PIL Image ready for base64 encoding and VLM processing.
    """
    logger.debug("Preprocessing image: %s", image_path.name)

    try:
        img = Image.open(image_path)
        logger.debug("Opened image: %s (%dx%d, mode: %s)", image_path.name, img.width, img.height, img.mode)
    except Exception as exc:
        logger.error("Failed to open image %s: %s", image_path.name, exc)
        raise

    # ── Step 1: EXIF orientation correction ────────────────────────────────────
    if ENABLE_ORIENTATION_CORRECTION:
        try:
            original_size = (img.width, img.height)
            img = _apply_exif_orientation(img)
            if (img.width, img.height) != original_size:
                logger.info("EXIF orientation corrected for %s: %s -> %s",
                          image_path.name, original_size, (img.width, img.height))
        except Exception as exc:
            logger.warning("EXIF orientation correction failed for %s: %s", image_path.name, exc)

    # ── Step 2: OCR-based orientation detection ───────────────────────────────
    if ENABLE_ORIENTATION_CORRECTION:
        try:
            corrected_img = _detect_and_correct_orientation(img)
            if corrected_img is not img:  # Only log if rotation was applied
                logger.info("Auto-corrected orientation for %s using OCR", image_path.name)
            img = corrected_img
        except Exception as exc:
            logger.warning("OCR orientation detection failed for %s: %s", image_path.name, exc)

    # ── Step 3: Basic enhancement for OCR readability ────────────────────────
    if ENABLE_IMAGE_ENHANCEMENT:
        try:
            img = _enhance_for_ocr(img)
            logger.debug("Enhanced image for OCR: %s", image_path.name)
        except Exception as exc:
            logger.warning("Image enhancement failed for %s: %s", image_path.name, exc)

    logger.debug("Preprocessing complete for %s", image_path.name)
    return img


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation tag to correct image rotation."""
    if not hasattr(img, '_getexif') or img._getexif() is None:
        return img

    exif = img._getexif()
    orientation = exif.get(274)  # EXIF Orientation tag

    if orientation == 1:
        return img  # Normal
    elif orientation == 2:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        return img.rotate(180)
    elif orientation == 4:
        return img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 5:
        return img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        return img.rotate(-90)
    elif orientation == 7:
        return img.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 8:
        return img.rotate(90)
    else:
        return img


def _detect_and_correct_orientation(img: Image.Image) -> Image.Image:
    """Use Tesseract OSD to detect text orientation and correct if needed."""
    # Convert PIL to numpy array for OpenCV
    img_array = np.array(img.convert('RGB'))

    # Convert RGB to BGR for OpenCV
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Get orientation data from Tesseract
    try:
        osd_data = pytesseract.image_to_osd(img_cv)
        logger.debug("OSD data for image: %s", osd_data)

        # Extract rotation angle from OSD output
        rotation_match = re.search(r'Rotate:\s*(\d+)', osd_data)
        if rotation_match:
            rotation = int(rotation_match.group(1))
            if rotation == 90:
                return img.rotate(-90, expand=True)
            elif rotation == 180:
                return img.rotate(180, expand=True)
            elif rotation == 270:
                return img.rotate(90, expand=True)

    except pytesseract.TesseractError as exc:
        logger.debug("Tesseract OSD failed: %s", exc)

    return img


def _enhance_for_ocr(img: Image.Image) -> Image.Image:
    """Apply basic enhancements to improve OCR readability."""
    # Convert to grayscale if not already
    if img.mode != 'L':
        img = img.convert('L')

    # Deskew the image
    img = _deskew_image(img)

    # Enhance contrast using PIL
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast by 2x

    # Sharpen slightly
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)

    return img.convert('RGB')  # Convert back to RGB for base64 encoding


def _deskew_image(img: Image.Image) -> Image.Image:
    """Deskew the image to correct slight rotations."""
    import numpy as np
    from scipy.ndimage import interpolation as inter

    # Convert to numpy array
    img_array = np.array(img)

    # Threshold to binary
    _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]

        # Correct the angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # Rotate the image
        if abs(angle) > 0.5:  # Only rotate if angle is significant
            img = img.rotate(angle, expand=True)

    return img


# ── Ollama wrapper (single choke-point for all model calls) ───────────────────
_EXTRACTION_PROMPT = (
    "You are an expert data-entry assistant specialising in handwritten Indian"
    " application forms. Carefully read every handwritten entry in the image"
    " and extract the fields listed below.\n\n"
    "Return ONLY a valid JSON object — no markdown fences, no explanation:\n\n"
    "{\n"
    '  "app_no": "...",\n'
    '  "name": "...",\n'
    '  "address_line_1": "...",\n'
    '  "address_line_2": "...",\n'
    '  "address_line_3": "...",\n'
    '  "mob_no": "...",\n'
    '  "email": "...",\n'
    '  "q1_chanted_before": "Yes | No | [NEEDS_REVIEW]",\n'
    '  "q2_familiarity_level": '
    '"Can chant confidently without guidance |'
    " Can chant but need occasional guidance |"
    " Know partially and need training |"
    ' Beginner-need full training | [NEEDS_REVIEW]",\n'
    '  "q3_formal_training": "Yes | No | [NEEDS_REVIEW]",\n'
    '  "q4_training_mode": "In-person | Online | Either | [NEEDS_REVIEW]",\n'
    '  "q5_preferred_language": '
    '"Kannada | Sanskrit | Other Language | [NEEDS_REVIEW]",\n'
    '  "q6_comments": "..."\n'
    "}\n\n"
    "Rules:\n"
    '- Empty or unfilled field → "[BLANK]"\n'
    '- Filled but completely illegible → "[UNREADABLE]"\n'
    "- Checkbox questions: use the exact label of the ticked option.\n"
    "- mob_no: digits only, no spaces, dashes or country code.\n"
    "- address_line_1, address_line_2, address_line_3: Extract each line of the address separately. If the address has fewer than 3 lines, leave the extra lines as '[BLANK]'. Do not combine lines.\n"
    "- app_no: Extract the application number as digits only.\n"
    "- name: Extract the full name.\n"
    "- Return ONLY the JSON object, nothing else."
)


def _call_ollama(model: str, image_path: Path, base_url: str) -> str:
    """Send *image_path* + extraction prompt to the Ollama model.

    This is the ONLY function that touches the Ollama SDK / HTTP layer.
    To swap the local runtime (e.g. to vLLM or llama.cpp), change only here.

    Args:
        model:      Ollama model name, e.g. ``"reducto/rolmocr"``.
        image_path: Path to the JPEG/PNG form image.
        base_url:   Ollama server base URL.

    Returns:
        Raw text response from the model.

    Raises:
        ollama.ResponseError: on API-level failures.
        FileNotFoundError:    if *image_path* does not exist.
        ValueError:           if the model returns no content.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Preprocess image: orientation correction + enhancement ───────────────
    processed_img = _preprocess_image(image_path)

    # Convert to base64 for Ollama API
    import io
    buffer = io.BytesIO()
    processed_img.save(buffer, format='PNG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    client = ollama.Client(host=base_url, timeout=OLLAMA_TIMEOUT_SECONDS)

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": _EXTRACTION_PROMPT,
                "images": [image_b64],
            }
        ],
        options={
            "temperature": 0.0,      # Deterministic output for data extraction.
            "num_predict": 512,      # Sufficient for the JSON response.
        },
    )

    text = response.get("message", {}).get("content", "").strip()
    if not text:
        raise ValueError("Ollama returned an empty response.")
    return text


# ── JSON parsing ──────────────────────────────────────────────────────────────


def _parse_response(raw: str) -> dict:
    """Parse the model's JSON response, stripping any accidental markdown fences.

    Raises:
        json.JSONDecodeError: if the cleaned text is not valid JSON.
    """
    # Strip ``` / ```json fences that sometimes appear despite instructions.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Extract the first {...} block in case the model added any preamble.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    return json.loads(cleaned)


# ── Confidence scoring ────────────────────────────────────────────────────────


def _compute_confidences(fields: dict) -> dict:
    """Derive per-field confidence scores from format validation.

    The local VLM does not expose raw logit/probability scores, so confidence
    is inferred from:
      • Whether the value passes domain format checks (mobile, email).
      • Whether the value is a sentinel ([BLANK], [UNREADABLE], [NEEDS_REVIEW]).
      • Heuristic length checks for name and address.
    """
    confs: dict[str, float] = {}

    for field, value in fields.items():
        val = str(value).strip() if value is not None else ""

        if val == _UNREADABLE:
            confs[field] = UNREADABLE_CONFIDENCE
        elif val == _BLANK:
            confs[field] = BLANK_CONFIDENCE
        elif val == _NEEDS_REVIEW:
            confs[field] = NEEDS_REVIEW_CONFIDENCE
        elif field == "app_no":
            confs[field] = 95.0 if val.isdigit() and len(val) >= 3 else 50.0
        elif field == "mob_no":
            digits = re.sub(r"\D", "", val)
            confs[field] = 92.0 if _MOBILE_RE.match(digits) else 55.0
        elif field == "email":
            confs[field] = 90.0 if _EMAIL_RE.match(val) else 55.0
        elif field == "name":
            confs[field] = FIELD_VALIDATION_CONFIDENCE if len(val) >= 3 else 45.0
        elif field in ("address_line_1", "address_line_2", "address_line_3"):
            confs[field] = FIELD_VALIDATION_CONFIDENCE if len(val) >= 3 else 45.0
        elif field.startswith("q"):
            # VLMs are highly reliable at checkbox reading — high base score.
            confs[field] = 88.0
        else:
            confs[field] = 70.0

    return {k: round(v, 1) for k, v in confs.items()}


# ── Post-processing ───────────────────────────────────────────────────────────


def _sanitise_fields(fields: dict) -> dict:
    """Normalise field values in-place after parsing."""
    # Strip country code from mobile number if present.
    mob = str(fields.get("mob_no") or "").strip()
    if mob and mob not in (_BLANK, _UNREADABLE, _NEEDS_REVIEW):
        digits = re.sub(r"\D", "", mob)
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        fields["mob_no"] = digits if digits else _UNREADABLE

    # Clean app_no to digits only
    app_no = str(fields.get("app_no") or "").strip()
    if app_no and app_no not in (_BLANK, _UNREADABLE, _NEEDS_REVIEW):
        digits = re.sub(r"\D", "", app_no)
        fields["app_no"] = digits if digits else _UNREADABLE

    # Normalise None → sentinel.
    for key, val in fields.items():
        if val is None or str(val).strip() == "":
            fields[key] = _BLANK

    return fields


# ── Extractor class ───────────────────────────────────────────────────────────


class OlmOCRExtractor(BaseExtractor):
    """Local VLM OCR engine backed by Ollama (olmOCR-2 / RolmOCR).

    Args:
        model:    Ollama model name.  Defaults to ``config.OLLAMA_MODEL``.
        base_url: Ollama server URL.  Defaults to ``config.OLLAMA_BASE_URL``.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self._model = model
        self._base_url = base_url

    @property
    def engine_name(self) -> str:
        return f"olmocr:{self._model}"

    def health_check(self) -> bool:
        """Return True when Ollama is running and the configured model is available."""
        try:
            client = ollama.Client(host=self._base_url, timeout=10)
            models_response = client.list()
            
            # Extract model names; try different response formats
            available = []
            for m in models_response.get("models", []):
                # Try 'name' first, then 'model'
                model_name = m.get("name") or m.get("model")
                if model_name:
                    available.append(model_name)
            
            if not available:
                logger.warning(
                    "Ollama is reachable but has no models pulled. "
                    "Run: ollama pull %s", self._model
                )
                return False
            # Check for an exact or prefix match (e.g. "reducto/rolmocr:latest").
            matched = any(
                m == self._model or m.startswith(self._model)
                for m in available
            )
            if not matched:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s. "
                    "Run: ollama pull %s",
                    self._model, available, self._model,
                )
            return matched
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Ollama health check failed — is Ollama running? Error: %s", exc
            )
            return False

    def extract(self, image_path: Path) -> ExtractionResult:
        """Run the local VLM on *image_path* and return a structured result."""
        logger.info("[OlmOCR] Processing: %s (model=%s)", image_path.name, self._model)

        try:
            raw_response = _call_ollama(self._model, image_path, self._base_url)
            logger.debug("[OlmOCR] Raw response:\n%s", raw_response[:600])
        except (FileNotFoundError, ValueError, Exception) as exc:  # noqa: BLE001
            logger.error("[OlmOCR] Inference failed for %s: %s", image_path.name, exc)
            return ExtractionResult(
                fields={}, confidences={}, engine=self.engine_name,
                error=f"Ollama inference error: {exc}",
            )

        try:
            fields = _parse_response(raw_response)
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.error("[OlmOCR] JSON parse failed for %s: %s", image_path.name, exc)
            return ExtractionResult(
                fields={}, confidences={}, engine=self.engine_name,
                raw_text=raw_response,
                error=f"JSON parse error: {exc}",
            )

        fields = _sanitise_fields(fields)
        confidences = _compute_confidences(fields)

        result = ExtractionResult(
            fields=fields,
            confidences=confidences,
            engine=self.engine_name,
            raw_text=raw_response,
        )
        logger.info(
            "[OlmOCR] Done: %s | avg_conf=%.1f",
            image_path.name,
            result.avg_text_field_confidence,
        )
        return result

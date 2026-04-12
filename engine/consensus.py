"""
engine/consensus.py — Compare Tesseract and Claude results and produce a
                       final merged ExtractionResult.

Logic:
  • Both engines produced a result → merge field-by-field, prefer the higher-
    confidence value; flag fields where the two diverge by more than
    CONFIDENCE_DIVERGENCE_THRESHOLD.
  • Only Tesseract → use Tesseract result as-is.
  • Only Claude     → use Claude result as-is.
  • Neither          → return error result.
"""

import logging
from typing import Optional

from config import CONFIDENCE_DIVERGENCE_THRESHOLD
from extractors.base_extractor import ExtractionResult

logger = logging.getLogger(__name__)

_ALL_FIELDS = (
    "app_no", "name", "address", "mob_no", "email",
    "q1_chanted_before", "q2_familiarity_level", "q3_formal_training",
    "q4_training_mode", "q5_preferred_language", "q6_comments",
)


def merge(
    tess_result: Optional[ExtractionResult],
    claude_result: Optional[ExtractionResult],
) -> ExtractionResult:
    """Merge Tesseract and Claude results into a single authoritative record.

    Args:
        tess_result:   Result from TesseractExtractor (may be None).
        claude_result: Result from ClaudeExtractor    (may be None).

    Returns:
        A new ExtractionResult with:
          • ``engine``: "tesseract", "claude", or "consensus".
          • ``flagged``:  True when any field diverges beyond the threshold.
          • ``flag_reason``: Human-readable description of divergent fields.
    """
    if tess_result is None and claude_result is None:
        logger.error("Both extractors returned None — cannot merge.")
        return ExtractionResult(
            fields={}, confidences={}, engine="none",
            error="No extraction result available from either engine."
        )

    if tess_result is None or tess_result.is_failed():
        logger.info("Using Claude result only (Tesseract unavailable/failed).")
        return claude_result  # type: ignore[return-value]

    if claude_result is None or claude_result.is_failed():
        logger.info("Using Tesseract result only (Claude unavailable/not run).")
        return tess_result

    # ── Both available — merge ────────────────────────────────────────────────
    merged_fields: dict[str, str] = {}
    merged_confs: dict[str, float] = {}
    divergent_fields: list[str] = []

    for field in _ALL_FIELDS:
        t_val = tess_result.fields.get(field, "[UNREADABLE]")
        c_val = claude_result.fields.get(field, "[UNREADABLE]")
        t_conf = tess_result.confidences.get(field, 0.0)
        c_conf = claude_result.confidences.get(field, 0.0)

        # Prefer the higher-confidence value.
        if c_conf >= t_conf:
            merged_fields[field] = c_val
            merged_confs[field] = c_conf
        else:
            merged_fields[field] = t_val
            merged_confs[field] = t_conf

        # Flag significant textual divergence on text fields.
        if field in ("app_no", "name", "address", "mob_no", "email"):
            conf_diff = abs(t_conf - c_conf)
            values_differ = _values_differ(t_val, c_val)
            if values_differ and conf_diff > CONFIDENCE_DIVERGENCE_THRESHOLD:
                divergent_fields.append(
                    f"{field}(tess={t_val!r}@{t_conf:.0f} vs "
                    f"claude={c_val!r}@{c_conf:.0f})"
                )

    flagged = len(divergent_fields) > 0
    flag_reason = "; ".join(divergent_fields) if flagged else ""

    if flagged:
        logger.warning(
            "Consensus divergence for image: %s", flag_reason
        )

    result = ExtractionResult(
        fields=merged_fields,
        confidences=merged_confs,
        engine="consensus",
        raw_text=(
            f"[TESSERACT]\n{tess_result.raw_text}\n\n"
            f"[CLAUDE]\n{claude_result.raw_text}"
        ),
    )
    # Attach flag metadata as extra attributes (not part of the dataclass).
    result.flagged = flagged
    result.flag_reason = flag_reason
    return result


def _values_differ(val_a: str, val_b: str) -> bool:
    """Return True when two field values are meaningfully different."""
    sentinel = {"[BLANK]", "[UNREADABLE]", "[NEEDS_REVIEW]", ""}
    # If either is a sentinel, don't count as a real content divergence.
    if val_a in sentinel or val_b in sentinel:
        return False
    # Normalise case and whitespace for comparison.
    return val_a.strip().lower() != val_b.strip().lower()

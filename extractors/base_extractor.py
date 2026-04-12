"""
extractors/base_extractor.py — Abstract interface every OCR engine must implement.

To plug in a new engine (Google Vision, AWS Textract, Azure Form Recognizer…):
  1. Subclass BaseExtractor.
  2. Implement ``extract()`` and ``health_check()``.
  3. Register the new class in extractors/__init__.py.
  4. The rest of the pipeline picks it up automatically.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class ExtractionResult:
    """Value object returned by every extractor.

    Attributes:
        fields:       Extracted field values keyed by field name.
        confidences:  Per-field confidence score in [0, 100].
        engine:       Human-readable engine identifier, e.g. "tesseract".
        raw_text:     Full raw OCR text for debugging.
        error:        Non-None when extraction failed entirely.
    """

    __slots__ = ("fields", "confidences", "engine", "raw_text", "error", "flagged", "flag_reason")

    def __init__(
        self,
        fields: dict,
        confidences: dict,
        engine: str,
        raw_text: str = "",
        error: str | None = None,
    ) -> None:
        self.fields = fields
        self.confidences = confidences
        self.engine = engine
        self.raw_text = raw_text
        self.error = error
        self.flagged: bool = False
        self.flag_reason: str = ""

    @property
    def avg_text_field_confidence(self) -> float:
        """Average confidence across the primary text fields only.

        Checkbox fields are intentionally excluded so that their typically
        lower Tesseract confidence does not unfairly trigger Claude review
        for forms where the text fields were read cleanly.
        """
        text_fields = ("app_no", "name", "address_line_1", "address_line_2", "address_line_3", "mob_no", "email")
        values = [
            v for k, v in self.confidences.items()
            if k in text_fields and isinstance(v, (int, float))
        ]
        return round(sum(values) / len(values), 1) if values else 0.0

    def is_failed(self) -> bool:
        return self.error is not None

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(engine={self.engine!r}, "
            f"avg_conf={self.avg_text_field_confidence}, "
            f"error={self.error!r})"
        )


class BaseExtractor(ABC):
    """Abstract base class for OCR/extraction engines."""

    @abstractmethod
    def extract(self, image_path: Path) -> ExtractionResult:
        """Extract all form fields from *image_path*.

        Must never raise — return an ExtractionResult with ``error`` set
        instead so the pipeline can handle failures gracefully.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the engine is installed and reachable."""
        raise NotImplementedError

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Short identifier string, e.g. ``"tesseract"`` or ``"claude"``."""
        raise NotImplementedError

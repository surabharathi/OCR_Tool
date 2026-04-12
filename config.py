"""
config.py — Central configuration for the Form Extractor application.
All tuneable constants live here; nothing is hard-coded elsewhere.

Architecture: Single-engine pipeline using a local VLM (olmOCR-2 / RolmOCR)
running on the NVIDIA RTX 4090 via Ollama.  No cloud API costs.
"""

# ── Local VLM (Ollama) ────────────────────────────────────────────────────────

# Primary recommended model — Qwen2.5-VL-7B
# Pull with:  ollama pull qwen2.5vl:7b
# Alternatives:  reducto/rolmocr, allenai/olmocr2
OLLAMA_MODEL: str = "qwen2.5vl:7b"

# Ollama REST endpoint — default when Ollama runs locally.
OLLAMA_BASE_URL: str = "http://localhost:11434"

# Seconds before a single-image inference call times out.
OLLAMA_TIMEOUT_SECONDS: int = 120

# ── Confidence thresholds ─────────────────────────────────────────────────────
# Confidence is derived from format-validation rules (no raw logit scores).
# Fields that pass format checks receive high scores; sentinels get lower ones.
FIELD_VALIDATION_CONFIDENCE: float = 85.0
UNREADABLE_CONFIDENCE: float = 0.0
BLANK_CONFIDENCE: float = 50.0
NEEDS_REVIEW_CONFIDENCE: float = 30.0

# ── CSV Output ────────────────────────────────────────────────────────────────
CSV_MAX_ROWS: int = 50_000
CSV_BASE_NAME: str = "extracted_data"

# ── Processing ────────────────────────────────────────────────────────────────
# RTX 4090 comfortably handles 2 concurrent inference calls.
# Raise to 3-4 if VRAM stays below 14 GB during your workload.
MAX_WORKERS: int = 2

# ── Image ─────────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".pdf"}
)

# Image preprocessing options
ENABLE_ORIENTATION_CORRECTION: bool = True  # Auto-detect and correct image orientation using EXIF + OCR
ENABLE_IMAGE_ENHANCEMENT: bool = True      # Apply contrast/sharpness enhancement for better OCR

# Tesseract OCR path (for orientation detection) — set to None to auto-detect
# Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
TESSERACT_CMD: str | None = None

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR: str = "logs"
LOG_MAX_BYTES: int = 10 * 1024 * 1024   # 10 MB per file
LOG_BACKUP_COUNT: int = 5

# ── State / Progress files ────────────────────────────────────────────────────
PROGRESS_FILE: str = "progress.json"
HASH_STORE_FILE: str = "processed_hashes.json"

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST: str = "127.0.0.1"
FLASK_PORT: int = 5000
FLASK_DEBUG: bool = False
SSE_HEARTBEAT_INTERVAL: int = 15  # seconds

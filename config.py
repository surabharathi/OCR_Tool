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

# How many times to retry a failed Ollama inference before giving up.
OLLAMA_MAX_RETRIES: int = 3

# Seconds to wait between retry attempts.
OLLAMA_RETRY_DELAY_SECONDS: float = 2.0

# Number of concurrent inference requests Ollama will serve simultaneously.
# Matches OLLAMA_NUM_PARALLEL env var — set that before starting Ollama:
#   $env:OLLAMA_NUM_PARALLEL = "1"
#   $env:OLLAMA_FLASH_ATTENTION = "1"
# With RTX 4090 (24 GB VRAM) and qwen2.5vl:7b (~4.5 GB weights), the vision
# encoder peak activation memory (not just KV cache) can be 8–10 GB per slot
# for 1500+ px images.  Running 2 slots simultaneously risks a Windows TDR
# (GPU hard crash).  1 slot is the safe ceiling; set to 2 only after measuring
# peak VRAM empirically with nvidia-smi dmon.
OLLAMA_NUM_PARALLEL: int = 1

# Maximum number of models Ollama keeps loaded in VRAM simultaneously.
# Keeping this at 1 prevents VRAM fragmentation across model swaps.
OLLAMA_MAX_LOADED_MODELS: int = 1

# How long Ollama keeps the model warm after the last request.
# Shorter values free VRAM sooner between runs.
OLLAMA_KEEP_ALIVE: str = "5m"

# ── Confidence thresholds ─────────────────────────────────────────────────────
# Confidence is derived from format-validation rules (no raw logit scores).
# Fields that pass format checks receive high scores; sentinels get lower ones.
FIELD_VALIDATION_CONFIDENCE: float = 85.0
UNREADABLE_CONFIDENCE: float = 0.0
BLANK_CONFIDENCE: float = 50.0
NEEDS_REVIEW_CONFIDENCE: float = 30.0

# When two engines agree on a field value but their confidence scores differ by
# more than this threshold, the field is flagged for manual review.
# Used by engine/consensus.py when running in multi-engine mode.
CONFIDENCE_DIVERGENCE_THRESHOLD: float = 20.0

# ── CSV Output ────────────────────────────────────────────────────────────────
CSV_MAX_ROWS: int = 50_000
CSV_BASE_NAME: str = "extracted_data"

# ── Processing ────────────────────────────────────────────────────────────────
# Number of Python worker threads submitting requests to Ollama concurrently.
# Must match OLLAMA_NUM_PARALLEL — having more workers than Ollama slots causes
# concurrent vision encoder passes that exhaust VRAM and trigger a Windows TDR.
MAX_WORKERS: int = 1

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

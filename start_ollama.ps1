# start_ollama.ps1 — Launch Ollama with RTX 4090 optimisations (Phase 1).
#
# Run this script INSTEAD of launching Ollama manually.  It sets the
# environment variables that Ollama reads at startup before serving any model.
#
# Usage:
#   .\start_ollama.ps1
#
# Then in a separate terminal run:
#   .\venv\Scripts\python.exe app.py

# ── Phase 1: Ollama performance env vars ─────────────────────────────────────

# Allow up to 2 concurrent inference requests.
# With qwen2.5vl:7b (~4.5 GB weights) on RTX 4090 (24 GB VRAM):
#   weights:          ~4.5 GB
#   KV cache × 2:    ~10.0 GB  (vision encoder needs 4–6 GB per slot
#                               for large scanned images at 1500+ px,
#                               NOT the 2 GB estimate for text-only inputs)
#   overhead:         ~1.5 GB
#   total:           ~16.0 GB  — safely within 24 GB budget.
# Running 6 slots with large images caused hard OOM crashes on previous runs.
$env:OLLAMA_NUM_PARALLEL = "2"

# Flash Attention: mathematically identical to standard attention but uses
# significantly less VRAM per KV cache slot, allowing more parallel slots.
# Zero accuracy impact — used in production everywhere.
$env:OLLAMA_FLASH_ATTENTION = "1"

# Keep the model loaded between requests (default is already "5m"; explicit here).
$env:OLLAMA_KEEP_ALIVE = "10m"

Write-Host "Starting Ollama with RTX 4090 optimisations..."
Write-Host "  OLLAMA_NUM_PARALLEL  = $env:OLLAMA_NUM_PARALLEL"
Write-Host "  OLLAMA_FLASH_ATTENTION = $env:OLLAMA_FLASH_ATTENTION"
Write-Host "  OLLAMA_KEEP_ALIVE    = $env:OLLAMA_KEEP_ALIVE"
Write-Host ""

ollama serve

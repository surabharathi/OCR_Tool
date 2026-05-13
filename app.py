"""
app.py — Flask web application for the Form Extractor (single-engine edition).

Endpoints:
  GET  /                          → Serve the single-page UI.
  POST /api/scan/start            → Begin the VLM scan.
  GET  /api/scan/stream/<scan_id> → SSE stream of live progress events.
  POST /api/scan/stop/<scan_id>   → Abort current scan.
  GET  /api/health                → Ollama + model availability check.
"""

import json
import logging
import queue
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

import logging_setup
from config import (
    FLASK_DEBUG, FLASK_HOST, FLASK_PORT,
    OLLAMA_BASE_URL, OLLAMA_MODEL, SSE_HEARTBEAT_INTERVAL,
)
from core.ollama_launcher import ensure_ollama_running, start_watchdog
from core.orchestrator import Orchestrator
from extractors.olm_ocr_extractor import OlmOCRExtractor

logging_setup.configure_logging()
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="ui/templates",
    static_folder="ui/static",
)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = uuid.uuid4().hex

_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _get_session(scan_id: str) -> dict | None:
    with _sessions_lock:
        return _sessions.get(scan_id)


def _set_session(scan_id: str, data: dict) -> None:
    with _sessions_lock:
        _sessions[scan_id] = data


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Check Ollama availability and whether the requested model is pulled."""
    model = request.args.get("model", OLLAMA_MODEL)
    base_url = request.args.get("base_url", OLLAMA_BASE_URL)
    extractor = OlmOCRExtractor(model=model, base_url=base_url)
    ok = extractor.health_check()
    return jsonify({
        "ollama_ok": ok,
        "model": model,
        "base_url": base_url,
        "status": "ready" if ok else "model_not_found",
    })


@app.route("/api/scan/start", methods=["POST"])
def start_scan():
    """
    Body (JSON):
      image_folder : str  — absolute path to the folder of form images.
      model        : str  — Ollama model name (optional, defaults to config).
      base_url     : str  — Ollama server URL (optional).
      fresh_start  : bool — wipe progress.json before starting (default false).
    """
    body = request.get_json(silent=True) or {}
    image_folder = (body.get("image_folder") or "").strip()
    model = (body.get("model") or OLLAMA_MODEL).strip()
    base_url = (body.get("base_url") or OLLAMA_BASE_URL).strip()
    fresh_start = bool(body.get("fresh_start", False))

    if not image_folder:
        return jsonify({"error": "image_folder is required."}), 400

    folder_path = Path(image_folder)
    if not folder_path.exists() or not folder_path.is_dir():
        return jsonify({"error": f"Folder does not exist: {image_folder}"}), 400

    scan_id = uuid.uuid4().hex
    progress_q: queue.Queue = queue.Queue(maxsize=1000)

    def _safe_put(event: dict) -> None:
        """Non-blocking enqueue — drops the event if the queue is full.

        A disconnected SSE client stops draining the queue.  Using the default
        blocking put() would cause the scan thread to hang permanently once the
        1 000-slot buffer is exhausted.  Dropping a progress event is far
        preferable to a deadlocked scan.
        """
        try:
            progress_q.put_nowait(event)
        except queue.Full:
            logger.warning("Progress queue full — dropping event: %s", event.get("type"))

    orchestrator = Orchestrator(
        image_folder=folder_path,
        model=model,
        base_url=base_url,
        progress_callback=_safe_put,
        fresh_start=fresh_start,
    )

    # Run asynchronously in a background thread so the UI can stream progress.
    def _run_scan() -> None:
        logger.info("Starting asynchronous scan %s for folder: %s", scan_id[:8], image_folder)
        try:
            orchestrator.run()
            # orchestrator.run() already emits scan_complete via the callback;
            # do NOT put another one here — that would deliver the event twice.
            logger.info("Scan %s completed successfully.", scan_id[:8])
        except Exception as exc:
            logger.exception("Scan %s crashed: %s", scan_id[:8], exc)
            progress_q.put_nowait({"type": "error", "message": str(exc)})
        finally:
            progress_q.put_nowait(None)  # Signal SSE generator to close.
            session = _get_session(scan_id)
            if session is not None:
                session["status"] = "completed"

    session_data = {
        "orchestrator": orchestrator,
        "queue": progress_q,
        "thread": None,
        "status": "running",
    }
    _set_session(scan_id, session_data)

    thread = threading.Thread(target=_run_scan, daemon=True)
    session_data["thread"] = thread
    thread.start()

    return jsonify({"scan_id": scan_id, "status": "running"})


@app.route("/api/scan/stream/<scan_id>")
def stream(scan_id: str):
    """Server-Sent Events endpoint for live progress."""
    session = _get_session(scan_id)
    if session is None:
        return jsonify({"error": "Unknown scan_id."}), 404

    def _generate():
        q: queue.Queue = session["queue"]
        while True:
            try:
                event = q.get(timeout=SSE_HEARTBEAT_INTERVAL)
                if event is None:
                    yield 'data: {"type": "stream_end"}\n\n'
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield 'data: {"type": "heartbeat"}\n\n'

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/scan/stop/<scan_id>", methods=["POST"])
def stop_scan(scan_id: str):
    """Signal the orchestrator to stop after the current image."""
    session = _get_session(scan_id)
    if session is None:
        return jsonify({"error": "Unknown scan_id."}), 404
    session["orchestrator"].stop()
    session["status"] = "stopping"
    logger.info("Stop requested for scan %s.", scan_id[:8])
    return jsonify({"status": "stop_requested"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _ollama_proc = ensure_ollama_running(OLLAMA_BASE_URL)
    if _ollama_proc is not None:
        start_watchdog(_ollama_proc, OLLAMA_BASE_URL)
    logger.info("Starting Form Extractor on http://%s:%d", FLASK_HOST, FLASK_PORT)
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True,
        use_reloader=False,
    )

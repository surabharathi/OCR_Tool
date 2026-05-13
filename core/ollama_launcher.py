"""
core/ollama_launcher.py — Manages the Ollama process lifecycle at app startup.

Called once when app.py starts:
  1. Kills any existing ollama process (so stale instances without the
     performance env vars are replaced).
  2. Starts a fresh ``ollama serve`` with OLLAMA_NUM_PARALLEL and
     OLLAMA_FLASH_ATTENTION set for the RTX 4090.
  3. Waits up to 30 s for the REST API to become ready before returning.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Optional

import psutil
import requests

from config import OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE, OLLAMA_MAX_LOADED_MODELS, OLLAMA_NUM_PARALLEL

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT = 30   # seconds to wait for ollama REST API to respond
_POLL_INTERVAL   = 0.5  # seconds between readiness polls
_GRACEFUL_WAIT   = 2    # seconds after SIGTERM before force-killing


# ── Internal helpers ──────────────────────────────────────────────────────────

def _kill_existing_ollama() -> int:
    """Terminate all running ollama processes.  Returns the count killed."""
    targets = []
    for proc in psutil.process_iter(["name", "pid"]):
        try:
            name = proc.info.get("name") or ""
            if name.lower().startswith("ollama"):
                targets.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    for proc in targets:
        try:
            logger.info("Terminating existing Ollama process (PID %d) …", proc.pid)
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if targets:
        time.sleep(_GRACEFUL_WAIT)   # give processes time to exit cleanly

    # Force-kill anything still alive
    for proc in targets:
        try:
            if proc.is_running():
                logger.warning("Force-killing Ollama process (PID %d).", proc.pid)
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return len(targets)


def _wait_for_ready(base_url: str, timeout: float) -> bool:
    """Poll the Ollama REST API until it responds 200 or *timeout* expires."""
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(tags_url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(_POLL_INTERVAL)
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_ollama_running(base_url: str = OLLAMA_BASE_URL) -> Optional[subprocess.Popen]:
    """Stop any existing Ollama, then start a fresh one with RTX 4090 optimisations.

    Args:
        base_url: Ollama REST base URL (default: ``config.OLLAMA_BASE_URL``).

    Returns:
        ``subprocess.Popen`` handle of the new Ollama process, or *None* on failure.
    """
    # ── 1. Kill stale instances ───────────────────────────────────────────────
    killed = _kill_existing_ollama()
    if killed:
        logger.info("Stopped %d existing Ollama process(es).", killed)
    else:
        logger.info("No existing Ollama process found — starting fresh.")

    # ── 2. Build environment with performance settings ────────────────────────
    env = os.environ.copy()
    env["OLLAMA_NUM_PARALLEL"]      = str(OLLAMA_NUM_PARALLEL)
    env["OLLAMA_MAX_LOADED_MODELS"] = str(OLLAMA_MAX_LOADED_MODELS)
    env["OLLAMA_FLASH_ATTENTION"]   = "1"
    env["OLLAMA_KEEP_ALIVE"]        = OLLAMA_KEEP_ALIVE

    logger.info(
        "Launching Ollama (NUM_PARALLEL=%s, MAX_LOADED_MODELS=%s, "
        "FLASH_ATTENTION=1, KEEP_ALIVE=%s)\u2026",
        env["OLLAMA_NUM_PARALLEL"],
        env["OLLAMA_MAX_LOADED_MODELS"],
        env["OLLAMA_KEEP_ALIVE"],
    )

    # ── 3. Launch `ollama serve` ──────────────────────────────────────────────
    kwargs: dict = {
        "env": env,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    # On Windows suppress the console popup window
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        proc = subprocess.Popen(["ollama", "serve"], **kwargs)
    except FileNotFoundError:
        logger.error(
            "'ollama' not found in PATH.  Install Ollama from https://ollama.com "
            "and ensure it is on your system PATH."
        )
        return None

    # ── 4. Wait for the REST API to be ready ─────────────────────────────────
    logger.info("Waiting up to %d s for Ollama to be ready…", _STARTUP_TIMEOUT)
    if _wait_for_ready(base_url, _STARTUP_TIMEOUT):
        logger.info("Ollama is ready at %s  (PID %d).", base_url, proc.pid)
        return proc

    logger.error(
        "Ollama did not become ready within %d s.  "
        "Check that port 11434 is free and Ollama is installed correctly.",
        _STARTUP_TIMEOUT,
    )
    proc.terminate()
    return None


def start_watchdog(proc: subprocess.Popen, base_url: str = OLLAMA_BASE_URL) -> threading.Thread:
    """Monitor the Ollama process and restart it automatically if it crashes.

    Runs as a daemon thread so it does not prevent the app from exiting.
    When Ollama exits unexpectedly (e.g. after a Windows GPU TDR event),
    the watchdog calls ``ensure_ollama_running`` to bring it back up before
    the next scan attempt.

    Args:
        proc:     The ``subprocess.Popen`` handle returned by
                  ``ensure_ollama_running``.
        base_url: Ollama REST base URL — forwarded to ``ensure_ollama_running``
                  on each restart.

    Returns:
        The started daemon ``threading.Thread`` (can be ignored by caller).
    """
    def _watch() -> None:
        watched = proc
        while True:
            watched.wait()   # Block until the Ollama process exits.
            exit_code = watched.returncode
            logger.warning(
                "Ollama process (PID %d) exited unexpectedly (code %s) \u2014 restarting\u2026",
                watched.pid, exit_code,
            )
            restarted = ensure_ollama_running(base_url)
            if restarted is None:
                logger.error("Watchdog: failed to restart Ollama after crash. Giving up.")
                break
            logger.info("Watchdog: Ollama restarted successfully (new PID %d).", restarted.pid)
            watched = restarted  # Watch the new process on the next iteration.

    t = threading.Thread(target=_watch, daemon=True, name="ollama-watchdog")
    t.start()
    logger.info("Ollama watchdog started.")
    return t

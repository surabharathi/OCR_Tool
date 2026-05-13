"""
core/orchestrator.py — Single-phase pipeline coordinator.

Architecture (simplified for RTX 4090 / local VLM):

    For each image in the folder
        1. Skip only if already processed via resume tracker.
        2. Run OlmOCRExtractor  →  fields + confidence scores.
        3. Write record to CSV.

No confidence-gating, no cloud API, no approval dialogs, no phases.
Everything runs on the local GPU — zero cost per image.
"""

import concurrent.futures
import logging
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional

import fitz
import ollama

from config import CSV_BASE_NAME, MAX_WORKERS, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS, SUPPORTED_EXTENSIONS
from core.csv_writer import CSVWriter
from core.resume_tracker import ResumeTracker
from extractors.base_extractor import ExtractionResult
from extractors.olm_ocr_extractor import OlmOCRExtractor

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], None]


def _result_to_record(result: ExtractionResult, filename: str) -> dict:
    """Flatten an ExtractionResult into a flat dict ready for CSVWriter."""
    record: dict = {"filename": filename}
    record.update(result.fields)

    for field, conf in result.confidences.items():
        record[f"{field}_confidence"] = conf

    record["overall_confidence"] = result.avg_text_field_confidence
    record["ocr_engine_used"] = result.engine

    # Flag for review if critical fields have low confidence
    critical_fields = ["app_no", "name", "mob_no"]
    low_conf_threshold = 70.0
    if any(result.confidences.get(field, 0) < low_conf_threshold for field in critical_fields):
        record["flagged"] = "Yes"
        record["flag_reason"] = "Low confidence in critical fields"
    else:
        record["flagged"] = "No"
        record["flag_reason"] = ""

    return record


class Orchestrator:
    """Coordinates the full image → CSV pipeline (single engine, single phase).

    Args:
        image_folder:      Folder containing JPEG/PNG form images.
        model:             Ollama model name (overrides config.OLLAMA_MODEL).
        base_url:          Ollama server URL (overrides config.OLLAMA_BASE_URL).
        progress_callback: Called with a progress dict after each image.
        fresh_start:       Wipe all previous CSVs and reset the progress tracker
                           before initialising the CSV writer.  Must be set here
                           (not via a separate cleanup_artifacts call) so that
                           CSVWriter creates its file *after* the old ones are
                           removed — preventing the header from being deleted.
    """

    def __init__(
        self,
        image_folder: str | Path,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        progress_callback: Optional[ProgressCallback] = None,
        fresh_start: bool = False,
    ) -> None:
        self.image_folder = Path(image_folder)
        self.model = model
        self.base_url = base_url
        self.progress_callback = progress_callback or (lambda _: None)

        self._stop_event = threading.Event()
        self._counter_lock = threading.Lock()  # Guards _done/_skipped/_failed across worker threads.

        self.tracker = ResumeTracker(self.image_folder)

        # Cleanup *before* CSVWriter so that writer.writeheader() is not undone
        # by a subsequent cleanup_artifacts() call.
        if fresh_start:
            self.cleanup_artifacts()
            self.tracker.reset()
            logger.info("Fresh start — previous artifacts deleted and progress tracker reset.")

        self.csv_writer = CSVWriter(self.image_folder)

        # Single shared Ollama client — httpx connection pool is thread-safe;
        # reusing it across MAX_WORKERS threads avoids per-image TCP handshakes.
        self._ollama_client = ollama.Client(host=base_url, timeout=OLLAMA_TIMEOUT_SECONDS)
        self.extractor = OlmOCRExtractor(model=model, base_url=base_url, client=self._ollama_client)

        self._total = 0
        self._done = 0
        self._skipped = 0
        self._failed = 0

    # ── Public control ────────────────────────────────────────────────────────

    def cleanup_artifacts(self) -> None:
        """Delete all auto-generated files from previous runs in *image_folder*.

        Removes:
        * All ``extracted_data_*.csv`` files written by :class:`CSVWriter`.
        * The ``.pdf_pages`` cache directory (rendered PDF page PNGs).

        Input images and original PDF source files are never touched.
        """
        # ── CSV files ─────────────────────────────────────────────────────────
        csv_pattern = f"{CSV_BASE_NAME}_*.csv"
        deleted_csv = 0
        for csv_file in self.image_folder.glob(csv_pattern):
            try:
                csv_file.unlink()
                logger.info("Deleted previous CSV: %s", csv_file.name)
                deleted_csv += 1
            except OSError as exc:
                logger.warning("Could not delete %s: %s", csv_file.name, exc)

        # ── PDF page cache directory ──────────────────────────────────────────
        pdf_cache = self.image_folder / ".pdf_pages"
        if pdf_cache.exists():
            try:
                shutil.rmtree(pdf_cache)
                logger.info("Deleted PDF page cache: %s", pdf_cache)
            except OSError as exc:
                logger.warning("Could not delete PDF cache %s: %s", pdf_cache, exc)

        logger.info(
            "Artifact cleanup complete — removed %d CSV file(s) and PDF page cache.",
            deleted_csv,
        )

    def stop(self) -> None:
        """Signal the pipeline to stop after the current image finishes."""
        self._stop_event.set()
        logger.info("Stop signal received.")

    # ── Image discovery ───────────────────────────────────────────────────────

    def get_image_files(self) -> list[Path]:
        """Return sorted list of supported image files in *image_folder*.

        Multi-page PDFs are converted into PNG pages and returned as image inputs.
        """
        files: list[Path] = []
        pdf_count = 0
        image_count = 0

        logger.info("Scanning folder %s for supported files...", self.image_folder)

        for p in self.image_folder.iterdir():
            if not p.is_file():
                continue

            suffix = p.suffix.lower()
            logger.debug("Found file: %s (suffix: %s)", p.name, suffix)

            if suffix == ".pdf":
                logger.info("Processing PDF: %s", p.name)
                pdf_pages = self._expand_pdf(p)
                files.extend(pdf_pages)
                pdf_count += 1
                logger.info("PDF %s expanded to %d pages", p.name, len(pdf_pages))
            elif suffix in (SUPPORTED_EXTENSIONS - {".pdf"}):  # Exclude PDF from image check
                files.append(p)
                image_count += 1
                logger.debug("Added image file: %s", p.name)

        files.sort(key=lambda p: p.name)
        logger.info("Found %d supported inputs (%d PDFs, %d images) in %s.",
                   len(files), pdf_count, image_count, self.image_folder)

        if not files:
            logger.warning("No supported files found in %s", self.image_folder)

        return files

    def _pdf_cache_dir(self) -> Path:
        cache_dir = self.image_folder / ".pdf_pages"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    def _expand_pdf(self, pdf_path: Path) -> list[Path]:
        """Render each page of a PDF into a PNG image in a hidden cache directory."""
        cache_dir = self._pdf_cache_dir()
        pages: list[Path] = []
        pdf_mtime = pdf_path.stat().st_mtime

        logger.info("Opening PDF: %s (modified: %s)", pdf_path.name, pdf_mtime)

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", pdf_path.name, exc)
            return pages

        # Use a context manager so the document is always closed — even if a
        # page render raises an unexpected exception mid-loop.
        with doc:
            logger.info("PDF opened successfully: %d pages", doc.page_count)
            for page_index in range(doc.page_count):
                output_path = cache_dir / f"{pdf_path.stem}_page_{page_index+1}.png"
                logger.debug("Processing page %d/%d -> %s", page_index+1, doc.page_count, output_path.name)

                # Check if page already exists and is up to date
                if output_path.exists():
                    png_mtime = output_path.stat().st_mtime
                    if png_mtime >= pdf_mtime:
                        logger.debug("Page %d already rendered (up to date)", page_index+1)
                        pages.append(output_path)
                        continue

                try:
                    page = doc.load_page(page_index)
                    logger.debug("Loaded page %d, rendering...", page_index+1)

                    # Render at 1.5× — sufficient for OCR, ~44% fewer pixels than
                    # 2.0× which cuts the vision encoder's peak VRAM significantly.
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    logger.debug("Rendered page %d (%dx%d)", page_index+1, pix.width, pix.height)

                    pix.save(output_path)
                    del pix  # Explicitly free the C-level pixel buffer immediately;
                              # do not wait for GC — 100-page PDFs accumulate ~440 MB.
                    logger.info("Saved page %d to %s", page_index+1, output_path.name)

                    pages.append(output_path)
                except Exception as exc:
                    logger.error(
                        "Failed to render page %d of %s: %s",
                        page_index + 1, pdf_path.name, exc,
                    )
                    continue

        logger.info(
            "Expanded PDF %s into %d page image(s) in %s",
            pdf_path.name, len(pages), cache_dir
        )
        return pages

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Process all images with the local VLM.

        Returns:
            Summary dict with processed / skipped / failed counts and CSV path.

        Raises:
            RuntimeError: if Ollama is unreachable or the model is not pulled.
        """
        if not self.extractor.health_check():
            raise RuntimeError(
                f"Ollama health check failed for model '{self.model}'. "
                f"Ensure Ollama is running and the model is pulled:\n"
                f"    ollama pull {self.model}"
            )

        images = self.get_image_files()
        self._total = len(images)
        self._done = self._skipped = self._failed = 0

        logger.info("Total images queued for processing: %d", self._total)
        logger.debug("Processing queue: %s", [p.name for p in images])

        self._emit({
            "type": "scan_start",
            "total": self._total,
            "model": self.model,
            "message": (
                f"Scanning {self._total} image(s) with {self.model} "
                f"on local GPU — no API costs."
            ),
        })

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            thread_name_prefix="olmocr",
        ) as pool:
            futures = {
                pool.submit(self._process_single, img): img
                for img in images
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    if self._stop_event.is_set():
                        logger.info("Scan stopped by user after current batch.")
                        break
                    img = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Unexpected error processing %s: %s", img.name, exc, exc_info=True)
                        with self._counter_lock:
                            self._failed += 1
                        self.tracker.mark_failed(img.name, str(exc))
                        self._emit_progress(img.name, "failed")
            except Exception:
                # Unexpected crash (e.g. health-check RuntimeError bubbling up,
                # or an unhandled OS error).  Notify the UI before re-raising so
                # the SSE stream gets a terminal event instead of silently closing.
                logger.exception("Scan terminated unexpectedly inside executor.")
                self._emit({
                    "type": "scan_interrupted",
                    "message": "Scan terminated unexpectedly. Check logs for details.",
                })
                raise

        summary = {
            "type": "scan_complete",
            "total": self._total,
            "processed": self._done,
            "skipped_duplicates": self._skipped,
            "failed": self._failed,
            "csv_file": str(self.csv_writer.current_file),
            "stats": self.tracker.get_stats(),
        }
        self._emit(summary)
        logger.info("Scan complete: %s", summary)
        return summary

    def _process_single(self, image_path: Path) -> None:
        """Process one image — called from the thread pool."""
        # Check stop signal first — lets queued workers exit without doing any work.
        if self._stop_event.is_set():
            return

        filename = image_path.name
        logger.info("Starting processing of: %s", filename)

        # ── Resume checks only; deduplication is disabled for full scan coverage. ──
        if self.tracker.is_already_handled(filename):
            logger.info("Skipping (resume): %s", filename)
            with self._counter_lock:
                self._skipped += 1
            self._emit_progress(filename, "skipped_resume")
            return

        logger.info("Deduplication disabled; processing %s", filename)

        # ── VLM extraction ─────────────────────────────────────────────────────
        logger.info("Starting VLM extraction for: %s", filename)
        result: ExtractionResult = self.extractor.extract(image_path)

        if result.is_failed():
            logger.error("Extraction failed for %s: %s", filename, result.error)
            self.tracker.mark_failed(filename, result.error or "Unknown error")
            with self._counter_lock:
                self._failed += 1
            self._emit_progress(filename, "failed", result=result)
            return

        # ── Write to CSV ───────────────────────────────────────────────────────
        logger.debug("Writing CSV record for: %s", filename)
        record = _result_to_record(result, filename)
        try:
            self.csv_writer.write_record(record)
            self.tracker.mark_processed(filename)
            with self._counter_lock:
                self._done += 1
            logger.info("Successfully processed: %s", filename)
            self._emit_progress(filename, "processed", result=result)
        except OSError as exc:
            logger.error("CSV write failed for %s: %s", filename, exc)
            self.tracker.mark_failed(filename, f"CSV write error: {exc}")
            with self._counter_lock:
                self._failed += 1
            self._emit_progress(filename, "failed")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _progress_pct(self) -> float:
        if self._total == 0:
            return 0.0
        done = self._done + self._skipped + self._failed
        return round(done / self._total * 100, 1)

    def _emit(self, payload: dict) -> None:
        try:
            self.progress_callback(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Progress callback raised: %s", exc)

    def _emit_progress(
        self,
        filename: str,
        status: str,
        result: Optional[ExtractionResult] = None,
    ) -> None:
        payload: dict = {
            "type": "image_done",
            "filename": filename,
            "status": status,
            "progress": self._progress_pct(),
            "counts": {
                "done": self._done,
                "skipped": self._skipped,
                "failed": self._failed,
                "total": self._total,
            },
        }
        if result is not None:
            payload["avg_confidence"] = result.avg_text_field_confidence
            payload["fields"] = result.fields
            payload["engine"] = result.engine

        logger.info(
            "Progress: %s -> %s | %s%% complete | done=%d skipped=%d failed=%d",
            filename,
            status,
            payload["progress"],
            self._done,
            self._skipped,
            self._failed,
        )
        self._emit(payload)

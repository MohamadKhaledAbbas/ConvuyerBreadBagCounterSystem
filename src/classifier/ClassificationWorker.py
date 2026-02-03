"""
Threaded Classification Worker.

Handles classification in background thread to prevent blocking main detection/tracking loop.

Architecture:
- Main thread: Detection + Tracking + ROI collection
- Classification thread: Classify completed tracks asynchronously
- Queue-based communication between threads

Benefits:
- Main loop can run at full speed (limited only by detection/tracking)
- Classification happens in parallel
- No frame drops due to slow classification
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
import numpy as np

from src.classifier.BaseClassifier import BaseClassifier, ClassificationResult
from src.utils.AppLogging import logger
from src.classifier.IClassificationComponents import IClassificationWorker


@dataclass
class ClassificationJob:
    """Job for classification worker."""
    track_id: int
    roi: np.ndarray  # Best quality ROI
    bbox_history: List[Tuple[int, int, int, int]]  # Full history for context
    callback: Optional[Callable] = None  # Callback(track_id, class_name, confidence)


class ClassificationWorker(IClassificationWorker):
    """
    Background worker for async classification.

    Runs in separate thread to avoid blocking main detection/tracking loop.
    Processes completed tracks from queue.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        max_queue_size: int = 100,
        name: str = "ClassificationWorker"
    ):
        """
        Initialize classification worker.

        Args:
            classifier: Classifier instance
            max_queue_size: Maximum jobs in queue
            name: Thread name for debugging
        """
        self.classifier = classifier
        self.name = name

        # Job queue
        self.job_queue: queue.Queue[ClassificationJob] = queue.Queue(maxsize=max_queue_size)

        # Worker thread
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Statistics
        self.total_jobs_processed = 0
        self.total_jobs_dropped = 0
        self.average_processing_time_ms = 0.0

        logger.info(f"[{self.name}] Initialized with queue size {max_queue_size}")

    def start(self):
        """Start background worker thread."""
        if self._running:
            logger.warning(f"[{self.name}] Already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=self.name,
            daemon=True
        )
        self._thread.start()
        logger.info(f"[{self.name}] Started")

    def stop(self, timeout: float = 5.0):
        """
        Stop background worker thread.

        Args:
            timeout: Max time to wait for thread to finish
        """
        if not self._running:
            return

        logger.info(f"[{self.name}] Stopping...")
        self._running = False
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

        logger.info(
            f"[{self.name}] Stopped. Processed: {self.total_jobs_processed}, "
            f"Dropped: {self.total_jobs_dropped}"
        )

    def submit_job(
        self,
        track_id: int,
        roi: np.ndarray,
        bbox_history: List[Tuple[int, int, int, int]],
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Submit classification job to queue.

        Args:
            track_id: Track identifier
            roi: Best quality ROI to classify
            bbox_history: Full bbox history for context
            callback: Optional callback(track_id, class_name, confidence)

        Returns:
            True if job queued, False if queue full (job dropped)
        """
        try:
            job = ClassificationJob(
                track_id=track_id,
                roi=roi.copy(),  # Copy to avoid race conditions
                bbox_history=bbox_history.copy(),
                callback=callback
            )

            # Non-blocking put with immediate failure if full
            self.job_queue.put_nowait(job)
            return True

        except queue.Full:
            self.total_jobs_dropped += 1
            logger.warning(
                f"[{self.name}] Queue full! Dropped track {track_id}. "
                f"Total dropped: {self.total_jobs_dropped}"
            )
            return False

    def _worker_loop(self):
        """Main worker loop - runs in background thread."""
        logger.info(f"[{self.name}] Worker loop started")

        while self._running:
            try:
                # Wait for job with timeout to allow clean shutdown
                job = self.job_queue.get(timeout=0.5)

                # Process job
                self._process_job(job)

            except queue.Empty:
                # No jobs available, continue loop
                continue

            except Exception as e:
                logger.error(f"[{self.name}] Error in worker loop: {e}", exc_info=True)

        # Process remaining jobs before exit
        remaining = self.job_queue.qsize()
        if remaining > 0:
            logger.info(f"[{self.name}] Processing {remaining} remaining jobs...")
            while not self.job_queue.empty():
                try:
                    job = self.job_queue.get_nowait()
                    self._process_job(job)
                except queue.Empty:
                    break

        logger.info(f"[{self.name}] Worker loop exited")

    def _process_job(self, job: ClassificationJob):
        """
        Process a single classification job.

        Args:
            job: ClassificationJob to process
        """
        start_time = time.perf_counter()

        try:
            # Classify the ROI
            result: ClassificationResult = self.classifier.classify(job.roi)

            processing_time = (time.perf_counter() - start_time) * 1000

            # Update statistics
            self.total_jobs_processed += 1
            self.average_processing_time_ms = (
                (self.average_processing_time_ms * (self.total_jobs_processed - 1) + processing_time)
                / self.total_jobs_processed
            )

            logger.debug(
                f"[{self.name}] Track {job.track_id}: {result.class_name} "
                f"({result.confidence:.2f}) in {processing_time:.1f}ms"
            )

            # Call callback if provided
            if job.callback:
                try:
                    job.callback(job.track_id, result.class_name, result.confidence)
                except Exception as e:
                    logger.error(
                        f"[{self.name}] Callback error for track {job.track_id}: {e}",
                        exc_info=True
                    )

        except Exception as e:
            logger.error(
                f"[{self.name}] Classification error for track {job.track_id}: {e}",
                exc_info=True
            )

    def get_queue_size(self) -> int:
        """Get current number of jobs in queue."""
        return self.job_queue.qsize()

    def get_statistics(self) -> dict:
        """Get worker statistics."""
        return {
            'running': self._running,
            'queue_size': self.job_queue.qsize(),
            'total_processed': self.total_jobs_processed,
            'total_dropped': self.total_jobs_dropped,
            'avg_processing_time_ms': self.average_processing_time_ms
        }

    def __del__(self):
        """Cleanup on deletion."""
        if self._running:
            self.stop(timeout=1.0)

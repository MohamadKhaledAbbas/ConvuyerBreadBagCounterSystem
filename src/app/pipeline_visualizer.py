"""
Pipeline visualizer for drawing annotations and display.

Separates visualization concerns from core pipeline logic.
"""

from typing import List, Dict
import cv2
import numpy as np

from src.detection.BaseDetection import Detection
from src.tracking.ITracker import TrackedObject
from src.config.tracking_config import TrackingConfig


class PipelineVisualizer:
    """
    Handles visualization and display of pipeline results.

    Responsibilities:
    - Draw detection boxes
    - Draw tracking boxes and IDs
    - Draw status overlay
    - Manage display window

    Does NOT handle:
    - Processing logic
    - Recording
    - Database
    """

    def __init__(
        self,
        tracking_config: TrackingConfig,
        window_name: str = "Conveyor Counter",
        display_size: tuple = (960, 540)
    ):
        """
        Initialize visualizer.

        Args:
            tracking_config: Tracking configuration for thresholds
            window_name: OpenCV window name
            display_size: Display resolution (width, height)
        """
        self.tracking_config = tracking_config
        self.window_name = window_name
        self.display_size = display_size
        self._window_created = False

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracks: List[TrackedObject],
        fps: float,
        active_tracks: int,
        total_counted: int,
        counts_by_class: Dict[str, int]
    ) -> np.ndarray:
        """
        Draw all annotations on frame.

        Args:
            frame: Input frame
            detections: Detection list
            tracks: Active tracks
            fps: Current FPS
            active_tracks: Number of active tracks
            total_counted: Total bags counted
            counts_by_class: Count per class

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw detections (blue boxes)
        self._draw_detections(annotated, detections)

        # Draw tracks (green boxes with IDs)
        self._draw_tracks(annotated, tracks)

        # Draw status overlay
        self._draw_status(annotated, fps, active_tracks, total_counted, counts_by_class)

        return annotated

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]):
        """Draw detection bounding boxes."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue

    def _draw_tracks(self, frame: np.ndarray, tracks: List[TrackedObject]):
        """Draw tracking boxes, IDs, and trajectories."""
        for track in tracks:
            x1, y1, x2, y2 = track.bbox

            # Color: green if confirmed, yellow if tentative
            confirmed = track.hits >= self.tracking_config.min_track_duration_frames
            color = (0, 255, 0) if confirmed else (0, 255, 255)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"ID:{track.track_id} ({track.confidence:.2f})"
            cv2.putText(
                frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )

            # Draw trajectory
            if len(track.position_history) > 1:
                pts = np.array(list(track.position_history), dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, 1)

    def _draw_status(
        self,
        frame: np.ndarray,
        fps: float,
        active_tracks: int,
        total_counted: int,
        counts_by_class: Dict[str, int]
    ):
        """Draw status overlay."""
        status_lines = [
            f"FPS: {fps:.1f}",
            f"Tracks: {active_tracks}",
            f"Counted: {total_counted}",
        ]

        # Add class counts
        for cls, count in sorted(counts_by_class.items()):
            status_lines.append(f"  {cls}: {count}")

        # Draw text with background
        y = 30
        for line in status_lines:
            # Text size for background
            (text_width, text_height), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw background rectangle
            cv2.rectangle(
                frame,
                (5, y - text_height - 5),
                (15 + text_width, y + 5),
                (0, 0, 0),
                -1
            )

            # Draw text
            cv2.putText(
                frame, line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )
            y += 25

    def show(self, frame: np.ndarray) -> bool:
        """
        Display frame in window.

        Args:
            frame: Frame to display

        Returns:
            False if user pressed 'q' to quit
        """
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

        # Resize for faster display
        display_frame = cv2.resize(frame, self.display_size)
        cv2.imshow(self.window_name, display_frame)

        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')

    def cleanup(self):
        """Close display window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

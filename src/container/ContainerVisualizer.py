"""
Container Pipeline Visualizer.

Handles visualization and display of container tracking results.
Draws QR detection boxes, track trajectories, direction arrows,
exit zones, and status panels on the frame.

Used in two modes:
- Display mode: cv2.imshow() with real-time updates
- Headless mode: annotation only for on-demand snapshots
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.container.qr.QRCodeDetector import QRDetection
from src.container.tracking.ContainerTracker import (
    TrackedContainer,
    Direction,
    TrackState,
)
from src.utils.AppLogging import logger


class ContainerVisualizer:
    """
    Visualization for the container QR tracking pipeline.

    Features:
    - QR detection bounding box with value label
    - Active track trajectories with direction arrows
    - Exit zone indicator lines (top/bottom)
    - Status panel (FPS, counts, mismatch, per-QR breakdown)
    - Event log (recent events)
    - Display window management (cv2.imshow)
    """

    # Color scheme
    COLORS = {
        'qr_detected': (0, 255, 120),         # Green for QR detection (active detect)
        'qr_predicted': (0, 165, 255),         # Orange for tracker-predicted bbox
        'track_positive': (0, 200, 0),         # Green for positive direction
        'track_negative': (0, 120, 255),       # Orange for negative direction
        'track_unknown': (200, 200, 200),      # Gray for unknown
        'track_lost': (80, 80, 255),           # Red for lost
        'exit_zone': (0, 255, 255),            # Yellow for exit zone lines
        'panel_bg': (30, 30, 35),              # Dark panel
        'panel_border': (70, 70, 80),          # Panel border
        'text_primary': (255, 255, 255),       # White
        'text_secondary': (180, 180, 190),     # Grey
        'text_header': (255, 200, 100),        # Gold
        'text_success': (100, 230, 120),       # Green
        'text_warning': (100, 180, 255),       # Orange
        'text_error': (120, 120, 255),         # Red
        'text_info': (255, 200, 100),          # Gold
        'trajectory': (255, 255, 0),           # Cyan for trajectory
    }

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE = 0.6
    FONT_SCALE_MEDIUM = 0.5
    FONT_SCALE_SMALL = 0.45

    def __init__(
        self,
        window_name: str = "Container Tracker",
        display_size: Tuple[int, int] = (1280, 720),
    ):
        self.window_name = window_name
        self.display_size = display_size
        self._window_created = False

    def annotate_frame(
        self,
        frame: np.ndarray,
        detection: Optional[QRDetection],
        active_tracks: Dict[int, TrackedContainer],
        fps: float = 0.0,
        avg_slack_ms: float = 0.0,
        estimated_max_fps: float = 0.0,
        total_positive: int = 0,
        total_negative: int = 0,
        total_lost: int = 0,
        qr_positive: Optional[Dict[int, int]] = None,
        qr_negative: Optional[Dict[int, int]] = None,
        recent_events: Optional[List[Dict]] = None,
        exit_zone_ratio: float = 0.15,
        frame_detections: Optional[List] = None,
        frame_mode: str = "detect",
    ) -> np.ndarray:
        """
        Draw all annotations on a frame.

        Args:
            frame: BGR frame to annotate (modified in place)
            detection: Current QR detection (if any) — legacy, single detection
            active_tracks: Currently active tracks
            fps: Current processing FPS
            total_positive: Total positive count
            total_negative: Total negative count
            total_lost: Total lost tracks
            qr_positive: Per-QR positive counts
            qr_negative: Per-QR negative counts
            recent_events: Recent event list for log display
            exit_zone_ratio: Exit zone ratio for line drawing
            frame_detections: List of (detection, is_predicted) tuples
                              Green box for detected, orange for predicted

        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]

        # Draw exit zone lines
        self._draw_exit_zones(frame, h, w, exit_zone_ratio)

        # Draw active tracks (trajectories + boxes)
        for track in active_tracks.values():
            self._draw_track(frame, track)

        # Draw detection / prediction boxes with colour coding
        if frame_detections:
            for det_obj, is_predicted in frame_detections:
                if is_predicted:
                    self._draw_qr_detection(frame, det_obj,
                                            color=self.COLORS['qr_predicted'],
                                            label_prefix="PRED")
                else:
                    self._draw_qr_detection(frame, det_obj,
                                            color=self.COLORS['qr_detected'],
                                            label_prefix="QR")
        elif detection:
            # Legacy fallback (single detection, no frame_detections list)
            self._draw_qr_detection(frame, detection)

        # Draw status panel
        self._draw_status_panel(
            frame, fps, avg_slack_ms, estimated_max_fps,
            total_positive, total_negative, total_lost,
            len(active_tracks), qr_positive, qr_negative,
            frame_mode=frame_mode,
        )

        # Draw event log
        if recent_events:
            self._draw_event_log(frame, recent_events, h)

        return frame

    def _draw_exit_zones(
        self, frame: np.ndarray, h: int, w: int, ratio: float
    ) -> None:
        """Draw exit zone indicator lines (vertical, left/right)."""
        left_x = int(w * ratio)
        right_x = int(w * (1 - ratio))
        color = self.COLORS['exit_zone']

        # Dashed vertical lines
        dash_len = 20
        gap_len = 10
        for y in range(0, h, dash_len + gap_len):
            end_y = min(y + dash_len, h)
            cv2.line(frame, (left_x, y), (left_x, end_y), color, 1)
            cv2.line(frame, (right_x, y), (right_x, end_y), color, 1)

        # Labels
        cv2.putText(
            frame, "EXIT (left/filled)", (left_x + 5, 20),
            self.FONT, self.FONT_SCALE_SMALL, color, 1
        )
        cv2.putText(
            frame, "EXIT (right/return)", (right_x - 160, 20),
            self.FONT, self.FONT_SCALE_SMALL, color, 1
        )

    def _draw_qr_detection(
        self, frame: np.ndarray, detection,
        color: tuple = None,
        label_prefix: str = "QR",
    ) -> None:
        """Draw QR code detection / prediction box and label."""
        if color is None:
            color = self.COLORS['qr_detected']

        # Bounding polygon
        cv2.polylines(frame, [detection.bbox], True, color, 2)

        # Center dot
        cv2.circle(frame, detection.center, 6, color, -1)
        cv2.circle(frame, detection.center, 8, color, 2)

        # Label with background
        qr_num = getattr(detection, 'qr_number', detection.value)
        label = f"{label_prefix} #{qr_num}"
        (tw, th), _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE_LARGE, 2)
        lx = detection.center[0] - tw // 2
        ly = detection.center[1] - 25

        cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), color, -1)
        cv2.putText(
            frame, label, (lx, ly),
            self.FONT, self.FONT_SCALE_LARGE, (0, 0, 0), 2
        )

    def _draw_track(
        self, frame: np.ndarray, track: TrackedContainer
    ) -> None:
        """Draw a tracked container with trajectory and direction arrow."""
        # Color by direction
        if track.direction == Direction.POSITIVE:
            color = self.COLORS['track_positive']
        elif track.direction == Direction.NEGATIVE:
            color = self.COLORS['track_negative']
        elif track.state == TrackState.LOST:
            color = self.COLORS['track_lost']
        else:
            color = self.COLORS['track_unknown']

        # Draw trajectory polyline
        if len(track.positions) >= 2:
            pts = [(p[0], p[1]) for p in track.positions]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], self.COLORS['trajectory'], 2)

        # Draw current position marker
        if track.positions:
            cx, cy = track.positions[-1][0], track.positions[-1][1]
            cv2.circle(frame, (cx, cy), 12, color, 3)

            # Direction arrow (horizontal)
            if len(track.positions) >= 3:
                prev_x = track.positions[-3][0]
                dx = cx - prev_x
                arrow_len = 30
                if dx < 0:  # Moving left (positive / filled)
                    cv2.arrowedLine(
                        frame, (cx + arrow_len, cy), (cx - arrow_len, cy),
                        color, 2, tipLength=0.35
                    )
                elif dx > 0:  # Moving right (negative / returning)
                    cv2.arrowedLine(
                        frame, (cx - arrow_len, cy), (cx + arrow_len, cy),
                        color, 2, tipLength=0.35
                    )

            # Track label
            label = f"T{track.track_id} QR#{track.qr_value}"
            (tw, th), _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE_MEDIUM, 1)
            cv2.rectangle(
                frame,
                (cx - tw // 2 - 3, cy - 30 - th - 3),
                (cx + tw // 2 + 3, cy - 27),
                self.COLORS['panel_bg'], -1
            )
            cv2.putText(
                frame, label, (cx - tw // 2, cy - 30),
                self.FONT, self.FONT_SCALE_MEDIUM, color, 1
            )

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        fps: float,
        avg_slack_ms: float,
        estimated_max_fps: float,
        total_positive: int,
        total_negative: int,
        total_lost: int,
        active_tracks: int,
        qr_positive: Optional[Dict[int, int]],
        qr_negative: Optional[Dict[int, int]],
        frame_mode: str = "detect",
    ) -> None:
        """Draw the status/stats panel on top-left."""
        x, y = 10, 10
        w, h_panel = 260, 280

        if qr_positive:
            h_panel += 25 * len(qr_positive)

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h_panel), self.COLORS['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h_panel), self.COLORS['panel_border'], 1)

        # Header
        cy = y + 25
        cv2.putText(
            frame, "Container Tracker", (x + 10, cy),
            self.FONT, self.FONT_SCALE_LARGE, self.COLORS['text_header'], 2
        )

        # FPS
        cy += 30
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (x + 10, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, self.COLORS['text_primary'], 1
        )
        cv2.putText(
            frame, f"Active: {active_tracks}", (x + 140, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, self.COLORS['text_info'], 1
        )

        # Rolling pacing / headroom estimate
        cy += 25
        slack_color = (
            self.COLORS['text_success'] if avg_slack_ms >= 15.0 else
            self.COLORS['text_warning'] if avg_slack_ms >= 5.0 else
            self.COLORS['text_error']
        )
        cv2.putText(
            frame, f"Slack~: {avg_slack_ms:.0f}ms", (x + 10, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, slack_color, 1
        )
        cv2.putText(
            frame, f"Est: {estimated_max_fps:.0f}", (x + 140, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, self.COLORS['text_info'], 1
        )

        # Counts
        cy += 28
        cv2.putText(
            frame, f"Positive:", (x + 10, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, self.COLORS['text_secondary'], 1
        )
        cv2.putText(
            frame, f"{total_positive}", (x + 120, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, self.COLORS['text_success'], 2
        )

        cy += 25
        cv2.putText(
            frame, f"Negative:", (x + 10, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, self.COLORS['text_secondary'], 1
        )
        cv2.putText(
            frame, f"{total_negative}", (x + 120, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, self.COLORS['text_warning'], 2
        )

        # Mismatch
        mismatch = total_positive - total_negative
        cy += 25
        m_color = self.COLORS['text_error'] if abs(mismatch) > 5 else self.COLORS['text_secondary']
        cv2.putText(
            frame, f"Mismatch:", (x + 10, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, self.COLORS['text_secondary'], 1
        )
        sign = "+" if mismatch > 0 else ""
        cv2.putText(
            frame, f"{sign}{mismatch}", (x + 120, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, m_color, 2
        )

        cy += 25
        cv2.putText(
            frame, f"Lost: {total_lost}", (x + 10, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, self.COLORS['text_error'], 1
        )

        # Frame processing mode
        cy += 25
        _mode_colors = {
            'detect': self.COLORS['text_success'],    # green
            'predict': self.COLORS['text_warning'],   # orange/yellow
            'gate': self.COLORS['text_info'],          # blue/cyan
        }
        _mode_labels = {
            'detect': 'QR Detect',
            'predict': 'Track Predict',
            'gate': 'Motion Gate',
        }
        mode_color = _mode_colors.get(frame_mode, self.COLORS['text_primary'])
        mode_label = _mode_labels.get(frame_mode, frame_mode)
        cv2.putText(
            frame, f"Mode: {mode_label}", (x + 10, cy),
            self.FONT, self.FONT_SCALE_MEDIUM, mode_color, 1
        )

        # Per-QR breakdown
        if qr_positive:
            cy += 30
            cv2.putText(
                frame, "Per-QR Breakdown:", (x + 10, cy),
                self.FONT, self.FONT_SCALE_SMALL, self.COLORS['text_header'], 1
            )
            for qr_val in sorted(qr_positive.keys()):
                cy += 22
                pos = qr_positive.get(qr_val, 0)
                neg = (qr_negative or {}).get(qr_val, 0)
                cv2.putText(
                    frame, f"  QR#{qr_val}: +{pos} / -{neg}",
                    (x + 10, cy),
                    self.FONT, self.FONT_SCALE_SMALL,
                    self.COLORS['text_primary'], 1
                )

    def _draw_event_log(
        self, frame: np.ndarray, events: List[Dict], frame_h: int
    ) -> None:
        """Draw recent events log in bottom-left corner."""
        if not events:
            return

        x = 10
        w = 400
        line_h = 22
        # Show last 5 events; lost events get an extra reason line
        display_events = events[-5:]

        total_lines = sum(
            2 if ev.get('is_lost') and ev.get('lost_reason') else 1
            for ev in display_events
        )
        h_panel = 30 + line_h * total_lines
        y = frame_h - h_panel - 10

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h_panel), self.COLORS['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h_panel), self.COLORS['panel_border'], 1)

        cy = y + 20
        cv2.putText(
            frame, "Recent Events", (x + 10, cy),
            self.FONT, self.FONT_SCALE_SMALL, self.COLORS['text_header'], 1
        )

        for ev in reversed(display_events):
            cy += line_h
            qr = ev.get('qr_value', '?')
            direction = ev.get('direction', 'unknown')
            lost = ev.get('is_lost', False)
            lost_reason = ev.get('lost_reason', '')

            if direction == 'positive':
                d_icon = "+"
                color = self.COLORS['text_success']
            elif direction == 'negative':
                d_icon = "-"
                color = self.COLORS['text_warning']
            else:
                d_icon = "?"
                color = self.COLORS['text_secondary']

            suffix = " [LOST]" if lost else ""
            label = f"{d_icon} QR#{qr} {direction}{suffix}"
            cv2.putText(
                frame, label, (x + 10, cy),
                self.FONT, self.FONT_SCALE_SMALL, color, 1
            )

            if lost and lost_reason:
                cy += line_h
                # Truncate reason to fit panel width
                reason_text = lost_reason[:52] + "…" if len(lost_reason) > 52 else lost_reason
                cv2.putText(
                    frame, f"  {reason_text}", (x + 10, cy),
                    self.FONT, self.FONT_SCALE_SMALL, self.COLORS['text_error'], 1
                )

    def show(self, frame: np.ndarray, delay_ms: int = 1) -> bool:
        """
        Display frame in OpenCV window.

        Args:
            frame: Frame to display
            delay_ms: waitKey delay in milliseconds (caller controls pacing)

        Returns:
            False if user pressed 'q' to quit
        """
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.window_name, 100, 100)
            self._window_created = True

        h, w = frame.shape[:2]
        if (w, h) == self.display_size:
            display_frame = frame
        else:
            display_frame = cv2.resize(frame, self.display_size)
        cv2.imshow(self.window_name, display_frame)

        key = cv2.waitKey(delay_ms) & 0xFF
        return key != ord('q')

    def cleanup(self) -> None:
        """Close display window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

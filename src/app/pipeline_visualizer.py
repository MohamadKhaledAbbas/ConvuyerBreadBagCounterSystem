"""
Pipeline visualizer for drawing annotations and display.

Separates visualization concerns from core pipeline logic.
Enhanced with comprehensive debug visualization for pipeline debugging.
Modern UI with improved UX, alignment, and visual appearance.
"""

import time
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection
from src.tracking.ITracker import TrackedObject


class PipelineVisualizer:
    """
    Handles visualization and display of pipeline results.

    Responsibilities:
    - Draw detection boxes
    - Draw tracking boxes and IDs
    - Draw status overlay
    - Draw debug information panel
    - Manage display window

    Does NOT handle:
    - Processing logic
    - Recording
    - Database
    """

    # Modern color scheme with better contrast and aesthetics
    COLORS = {
        # Detection & Tracking
        'detection': (255, 140, 40),       # Soft blue for detections
        'track_tentative': (80, 200, 255), # Warm orange for tentative tracks
        'track_confirmed': (80, 220, 80),  # Soft green for confirmed tracks
        'track_lost': (80, 80, 255),       # Soft red for lost tracks

        # Panel backgrounds (with transparency support)
        'panel_bg': (30, 30, 35),          # Dark charcoal
        'panel_border': (70, 70, 80),      # Subtle gray border
        'panel_header_bg': (45, 45, 55),   # Slightly lighter header

        # Text colors
        'text_primary': (255, 255, 255),   # White for primary text
        'text_secondary': (180, 180, 190), # Gray for secondary text
        'text_header': (255, 200, 100),    # Warm gold for headers
        'text_success': (100, 230, 120),   # Green for success/counts
        'text_warning': (100, 180, 255),   # Orange for warnings
        'text_error': (120, 120, 255),     # Red for errors
        'text_info': (255, 200, 100),      # Gold for info

        # Pipeline stages
        'stage_bg': (50, 50, 60),          # Dark stage background
        'stage_border': (100, 180, 255),   # Accent blue border
        'stage_text': (255, 255, 255),     # White stage text
        'stage_arrow': (120, 120, 130),    # Gray arrows

        # Event log
        'event_count': (100, 230, 120),    # Green
        'event_reject': (100, 180, 255),   # Orange
        'event_batch': (180, 140, 255),    # Purple
        'event_classify': (255, 200, 100), # Gold
    }

    # Font settings - more consistent sizing
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_TITLE = 0.7
    FONT_SCALE_LARGE = 0.6
    FONT_SCALE_MEDIUM = 0.5
    FONT_SCALE_SMALL = 0.45

    FONT_WEIGHT_BOLD = 2
    FONT_WEIGHT_NORMAL = 1

    # Box and line settings
    BOX_THICKNESS_THICK = 3
    BOX_THICKNESS_NORMAL = 2
    BOX_THICKNESS_THIN = 1

    # Panel settings
    PANEL_PADDING = 15
    PANEL_RADIUS = 8
    LINE_HEIGHT_LARGE = 32
    LINE_HEIGHT_MEDIUM = 26
    LINE_HEIGHT_SMALL = 22

    def __init__(
        self,
        tracking_config: TrackingConfig,
        window_name: str = "Conveyor Counter",
        display_size: tuple = (1280, 720)
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

    def _draw_rounded_rect(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = -1,
        radius: int = 8
    ):
        """Draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2

        if thickness == -1:
            # Filled rounded rectangle
            # Draw main rectangle
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)

            # Draw corners
            cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            # Outlined rounded rectangle
            cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

            cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def _draw_panel_background(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        alpha: float = 0.85
    ):
        """Draw a semi-transparent panel background with subtle border."""
        overlay = frame.copy()

        # Draw filled background
        self._draw_rounded_rect(
            overlay,
            (x, y),
            (x + w, y + h),
            self.COLORS['panel_bg'],
            -1,
            self.PANEL_RADIUS
        )

        # Blend with original
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw border
        self._draw_rounded_rect(
            frame,
            (x, y),
            (x + w, y + h),
            self.COLORS['panel_border'],
            1,
            self.PANEL_RADIUS
        )

    def _draw_header(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        w: int
    ) -> int:
        """Draw a styled header and return the new y position."""
        # Header background
        header_h = 35
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + header_h),
            self.COLORS['panel_header_bg'],
            -1
        )

        # Header text (centered)
        text_size = cv2.getTextSize(text, self.FONT, self.FONT_SCALE_TITLE, self.FONT_WEIGHT_BOLD)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (header_h + text_size[1]) // 2

        cv2.putText(
            frame, text, (text_x, text_y),
            self.FONT, self.FONT_SCALE_TITLE,
            self.COLORS['text_header'], self.FONT_WEIGHT_BOLD
        )

        # Subtle underline
        cv2.line(
            frame,
            (x + 10, y + header_h),
            (x + w - 10, y + header_h),
            self.COLORS['panel_border'],
            1
        )

        return y + header_h + 10

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracks: List[TrackedObject],
        fps: float,
        active_tracks: int,
        total_counted: int,
        counts_by_class: Dict[str, int],
        debug_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Draw all annotations on frame including debug panel.

        Args:
            frame: Input frame
            detections: Detection list
            tracks: Active tracks
            fps: Current FPS
            active_tracks: Number of active tracks
            total_counted: Total bags counted (CONFIRMED)
            counts_by_class: Count per class (CONFIRMED)
            debug_info: Optional debug information dict including tentative counts

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw detections (blue boxes)
        self._draw_detections(annotated, detections)

        # Draw tracks (colored boxes with IDs and info)
        self._draw_tracks(annotated, tracks)

        # Draw main status overlay (top-left)
        tentative_total = debug_info.get('tentative_total', 0) if debug_info else 0
        tentative_counts = debug_info.get('tentative_counts', {}) if debug_info else {}
        self._draw_status(
            annotated, fps, active_tracks, total_counted, counts_by_class,
            tentative_total, tentative_counts
        )

        # Draw pipeline debug panel (right side)
        if debug_info:
            self._draw_debug_panel(annotated, debug_info, tracks)

        # Draw recent events log (bottom)
        if debug_info and 'recent_events' in debug_info:
            self._draw_event_log(annotated, debug_info['recent_events'])

        return annotated

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]):
        """Draw detection bounding boxes with clean styling."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw detection box with rounded corners effect (using lines)
            color = self.COLORS['detection']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.BOX_THICKNESS_THIN)

            # Confidence badge
            conf_text = f"{det.confidence:.0%}"
            text_size = cv2.getTextSize(conf_text, self.FONT, self.FONT_SCALE_SMALL, self.FONT_WEIGHT_NORMAL)[0]

            badge_w = text_size[0] + 10
            badge_h = text_size[1] + 8

            # Badge background
            cv2.rectangle(
                frame,
                (x1, y2 + 2),
                (x1 + badge_w, y2 + badge_h + 2),
                color,
                -1
            )

            # Badge text
            cv2.putText(
                frame, conf_text,
                (x1 + 5, y2 + badge_h - 2),
                self.FONT, self.FONT_SCALE_SMALL,
                (255, 255, 255), self.FONT_WEIGHT_NORMAL
            )

    def _draw_tracks(self, frame: np.ndarray, tracks: List[TrackedObject]):
        """Draw tracking boxes with modern styling."""
        for track in tracks:
            x1, y1, x2, y2 = track.bbox

            # Determine track state and color
            confirmed = track.hits >= self.tracking_config.min_track_duration_frames
            is_stale = track.time_since_update > 0

            if is_stale:
                color = self.COLORS['track_lost']
                state_icon = "⏸"
                state_str = f"LOST +{track.time_since_update}"
                thickness = self.BOX_THICKNESS_NORMAL
            elif confirmed:
                color = self.COLORS['track_confirmed']
                state_icon = "✓"
                state_str = "CONFIRMED"
                thickness = self.BOX_THICKNESS_THICK
            else:
                color = self.COLORS['track_tentative']
                state_icon = "○"
                state_str = f"PENDING {track.hits}/{self.tracking_config.min_track_duration_frames}"
                thickness = self.BOX_THICKNESS_NORMAL

            # Draw main box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw corner accents for confirmed tracks
            if confirmed:
                corner_len = 15
                # Top-left
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
                # Top-right
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
                # Bottom-left
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
                # Bottom-right
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

            # Label with ID and state
            label = f"T{track.track_id}  {state_str}"
            text_size = cv2.getTextSize(label, self.FONT, self.FONT_SCALE_MEDIUM, self.FONT_WEIGHT_BOLD)[0]

            label_h = text_size[1] + 12
            label_w = text_size[0] + 16

            # Position label above or below box
            label_y = y1 - label_h - 4
            if label_y < 10:
                label_y = y2 + 4

            # Label background with color accent
            cv2.rectangle(
                frame,
                (x1, label_y),
                (x1 + label_w, label_y + label_h),
                self.COLORS['panel_bg'],
                -1
            )

            # Color accent bar
            cv2.rectangle(
                frame,
                (x1, label_y),
                (x1 + 4, label_y + label_h),
                color,
                -1
            )

            # Label text
            cv2.putText(
                frame, label,
                (x1 + 10, label_y + text_size[1] + 6),
                self.FONT, self.FONT_SCALE_MEDIUM,
                color, self.FONT_WEIGHT_BOLD
            )

            # Draw velocity arrow
            if hasattr(track, 'velocity') and track.velocity is not None:
                vx, vy = track.velocity
                cx, cy = track.center
                scale = 6
                end_x = int(cx + vx * scale)
                end_y = int(cy + vy * scale)
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)

            # Draw trajectory
            if len(track.position_history) > 1:
                pts = np.array(list(track.position_history), dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, 2)

    def _draw_status(
        self,
        frame: np.ndarray,
        fps: float,
        active_tracks: int,
        total_counted: int,
        counts_by_class: Dict[str, int],
        tentative_total: int,
        tentative_counts: Dict[str, int]
    ):
        """Draw main status panel with clean modern design."""
        # Calculate panel dimensions
        num_classes = max(len(counts_by_class), 1)
        panel_w = 280
        panel_h = 280 + (num_classes * self.LINE_HEIGHT_SMALL)
        panel_x = 15
        panel_y = 15

        # Draw panel background
        self._draw_panel_background(frame, panel_x, panel_y, panel_w, panel_h)

        # Draw header
        y = self._draw_header(frame, "COUNTER STATUS", panel_x, panel_y, panel_w)
        y += 5

        x = panel_x + self.PANEL_PADDING

        # System metrics section
        # FPS with color indicator
        fps_color = self.COLORS['text_success'] if fps >= 15 else (
            self.COLORS['text_warning'] if fps >= 10 else self.COLORS['text_error']
        )

        # FPS indicator dot
        dot_radius = 5
        cv2.circle(frame, (x + dot_radius, y + 8), dot_radius, fps_color, -1)

        cv2.putText(
            frame, f"FPS: {fps:.1f}", (x + 20, y + 12),
            self.FONT, self.FONT_SCALE_MEDIUM,
            self.COLORS['text_primary'], self.FONT_WEIGHT_NORMAL
        )
        y += self.LINE_HEIGHT_MEDIUM

        # Active tracks
        cv2.putText(
            frame, f"Active Tracks: {active_tracks}", (x, y + 12),
            self.FONT, self.FONT_SCALE_MEDIUM,
            self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL
        )
        y += self.LINE_HEIGHT_MEDIUM + 10

        # Divider
        cv2.line(frame, (x, y), (panel_x + panel_w - self.PANEL_PADDING, y),
                self.COLORS['panel_border'], 1)
        y += 15

        # Tentative counts section
        cv2.putText(
            frame, "Pending", (x, y + 10),
            self.FONT, self.FONT_SCALE_SMALL,
            self.COLORS['text_warning'], self.FONT_WEIGHT_NORMAL
        )

        # Tentative count badge
        tent_text = str(tentative_total)
        tent_size = cv2.getTextSize(tent_text, self.FONT, self.FONT_SCALE_LARGE, self.FONT_WEIGHT_BOLD)[0]
        tent_x = panel_x + panel_w - self.PANEL_PADDING - tent_size[0] - 16

        cv2.rectangle(
            frame,
            (tent_x, y - 2),
            (tent_x + tent_size[0] + 16, y + tent_size[1] + 8),
            self.COLORS['text_warning'],
            -1
        )
        cv2.putText(
            frame, tent_text, (tent_x + 8, y + tent_size[1] + 2),
            self.FONT, self.FONT_SCALE_LARGE,
            (30, 30, 35), self.FONT_WEIGHT_BOLD
        )
        y += self.LINE_HEIGHT_LARGE + 5

        # Confirmed counts section (prominent)
        cv2.putText(
            frame, "Confirmed", (x, y + 10),
            self.FONT, self.FONT_SCALE_SMALL,
            self.COLORS['text_success'], self.FONT_WEIGHT_NORMAL
        )
        y += self.LINE_HEIGHT_MEDIUM

        # Large confirmed count
        count_text = str(total_counted)
        count_size = cv2.getTextSize(count_text, self.FONT, 1.5, 3)[0]

        cv2.putText(
            frame, count_text, (x, y + count_size[1]),
            self.FONT, 1.5,
            self.COLORS['text_success'], 3
        )
        y += count_size[1] + 20

        # Divider
        cv2.line(frame, (x, y), (panel_x + panel_w - self.PANEL_PADDING, y),
                self.COLORS['panel_border'], 1)
        y += 15

        # Class breakdown
        cv2.putText(
            frame, "By Class", (x, y + 10),
            self.FONT, self.FONT_SCALE_SMALL,
            self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL
        )
        y += self.LINE_HEIGHT_MEDIUM

        if counts_by_class:
            for cls, count in sorted(counts_by_class.items()):
                # Truncate long class names
                display_cls = cls if len(cls) <= 20 else cls[:17] + "..."

                # Class name
                cv2.putText(
                    frame, display_cls, (x + 5, y + 10),
                    self.FONT, self.FONT_SCALE_SMALL,
                    self.COLORS['text_primary'], self.FONT_WEIGHT_NORMAL
                )

                # Count (right-aligned)
                count_str = str(count)
                count_w = cv2.getTextSize(count_str, self.FONT, self.FONT_SCALE_SMALL, self.FONT_WEIGHT_BOLD)[0][0]
                cv2.putText(
                    frame, count_str,
                    (panel_x + panel_w - self.PANEL_PADDING - count_w, y + 10),
                    self.FONT, self.FONT_SCALE_SMALL,
                    self.COLORS['text_success'], self.FONT_WEIGHT_BOLD
                )
                y += self.LINE_HEIGHT_SMALL
        else:
            cv2.putText(
                frame, "No items yet", (x + 5, y + 10),
                self.FONT, self.FONT_SCALE_SMALL,
                self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL
            )

    def _draw_debug_panel(
        self,
        frame: np.ndarray,
        debug_info: Dict,
        tracks: List[TrackedObject]
    ):
        """Draw pipeline debug panel with modern styling."""
        h, w = frame.shape[:2]
        panel_w = 300
        panel_h = 380
        panel_x = w - panel_w - 15
        panel_y = 15

        # Draw panel background
        self._draw_panel_background(frame, panel_x, panel_y, panel_w, panel_h)

        # Draw header
        y = self._draw_header(frame, "PIPELINE DEBUG", panel_x, panel_y, panel_w)
        y += 8

        x = panel_x + self.PANEL_PADDING
        content_w = panel_w - (self.PANEL_PADDING * 2)

        # Pipeline stages
        stages = [
            ("DETECT", f"{len(tracks)} objects", "🔍"),
            ("TRACK", f"{len(tracks)} active", "📍"),
            ("ROI COLLECT", f"+{debug_info.get('rois_collected', 0)}", "📷"),
            ("CLASSIFY", f"{debug_info.get('pending_classify', 0)} queue", "🏷"),
            ("SMOOTH", f"{debug_info.get('pending_smooth', 0)} batch", "📊"),
            ("COUNT", f"{debug_info.get('rejected', 0)} reject", "✓"),
        ]

        stage_h = 32
        stage_gap = 8

        for i, (stage, info, icon) in enumerate(stages):
            # Stage background
            cv2.rectangle(
                frame,
                (x, y),
                (x + content_w, y + stage_h),
                self.COLORS['stage_bg'],
                -1
            )

            # Left accent bar
            cv2.rectangle(
                frame,
                (x, y),
                (x + 3, y + stage_h),
                self.COLORS['stage_border'],
                -1
            )

            # Stage number and name
            stage_label = f"{i+1}. {stage}"
            cv2.putText(
                frame, stage_label, (x + 10, y + 20),
                self.FONT, self.FONT_SCALE_SMALL,
                self.COLORS['stage_text'], self.FONT_WEIGHT_NORMAL
            )

            # Stage info (right-aligned)
            info_w = cv2.getTextSize(info, self.FONT, self.FONT_SCALE_SMALL, self.FONT_WEIGHT_NORMAL)[0][0]
            cv2.putText(
                frame, info, (x + content_w - info_w - 8, y + 20),
                self.FONT, self.FONT_SCALE_SMALL,
                self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL
            )

            y += stage_h

            # Draw connector arrow between stages
            if i < len(stages) - 1:
                arrow_x = x + content_w // 2
                cv2.line(frame, (arrow_x, y), (arrow_x, y + stage_gap - 2),
                        self.COLORS['stage_arrow'], 1)
                # Arrow head
                cv2.line(frame, (arrow_x - 4, y + stage_gap - 6), (arrow_x, y + stage_gap - 2),
                        self.COLORS['stage_arrow'], 1)
                cv2.line(frame, (arrow_x + 4, y + stage_gap - 6), (arrow_x, y + stage_gap - 2),
                        self.COLORS['stage_arrow'], 1)
                y += stage_gap

        y += 15

        # Divider
        cv2.line(frame, (x, y), (x + content_w, y), self.COLORS['panel_border'], 1)
        y += 15

        # Last classification
        last_class = debug_info.get('last_class', 'None')
        if last_class is None:
            last_class = 'None'
        elif len(last_class) > 22:
            last_class = last_class[:19] + "..."

        cv2.putText(
            frame, "Last:", (x, y + 12),
            self.FONT, self.FONT_SCALE_SMALL,
            self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL
        )
        cv2.putText(
            frame, last_class, (x + 45, y + 12),
            self.FONT, self.FONT_SCALE_SMALL,
            self.COLORS['text_info'], self.FONT_WEIGHT_NORMAL
        )
        y += self.LINE_HEIGHT_MEDIUM

        # Processing time
        proc_ms = debug_info.get('processing_ms', 0)
        proc_color = self.COLORS['text_success'] if proc_ms < 50 else (
            self.COLORS['text_warning'] if proc_ms < 100 else self.COLORS['text_error']
        )

        cv2.putText(
            frame, "Process:", (x, y + 12),
            self.FONT, self.FONT_SCALE_SMALL,
            self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL
        )
        cv2.putText(
            frame, f"{proc_ms:.1f}ms", (x + 65, y + 12),
            self.FONT, self.FONT_SCALE_SMALL,
            proc_color, self.FONT_WEIGHT_NORMAL
        )

    def _draw_event_log(self, frame: np.ndarray, recent_events: List[Tuple[float, str]]):
        """Draw recent events log with modern styling."""
        h, w = frame.shape[:2]

        if not recent_events:
            return

        # Get last 5 events
        events_to_show = recent_events[-5:]

        # Calculate panel dimensions
        panel_w = 500
        panel_h = len(events_to_show) * self.LINE_HEIGHT_SMALL + 50
        panel_x = 15
        panel_y = h - panel_h - 15

        # Draw panel background
        self._draw_panel_background(frame, panel_x, panel_y, panel_w, panel_h)

        # Draw header
        y = self._draw_header(frame, "RECENT EVENTS", panel_x, panel_y, panel_w)
        y += 5

        x = panel_x + self.PANEL_PADDING

        current_time = time.time()

        for event_time, event_text in events_to_show:
            age = current_time - event_time

            # Determine color based on event type
            if 'CONFIRMED' in event_text or 'COUNT' in event_text:
                color = self.COLORS['event_count']
                indicator = "●"
            elif 'REJECTED' in event_text:
                color = self.COLORS['event_reject']
                indicator = "○"
            elif 'BATCH' in event_text or 'SMOOTH' in event_text:
                color = self.COLORS['event_batch']
                indicator = "◆"
            elif 'CLASSIFY' in event_text:
                color = self.COLORS['event_classify']
                indicator = "▸"
            else:
                color = self.COLORS['text_secondary']
                indicator = "·"

            # Fade older events
            alpha = max(0.4, 1.0 - (age / 10.0))
            faded_color = tuple(int(c * alpha) for c in color)

            # Timestamp
            time_str = f"{age:5.1f}s"
            cv2.putText(
                frame, time_str, (x, y + 12),
                self.FONT, self.FONT_SCALE_SMALL,
                self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL
            )

            # # Indicator dot
            # cv2.putText(
            #     frame, indicator, (x + 55, y + 12),
            #     self.FONT, self.FONT_SCALE_SMALL,
            #     faded_color, self.FONT_WEIGHT_NORMAL
            # )

            # Event text (truncate if needed)
            max_len = 50
            display_text = event_text if len(event_text) <= max_len else event_text[:max_len-3] + "..."
            cv2.putText(
                frame, display_text, (x + 75, y + 12),
                self.FONT, self.FONT_SCALE_SMALL,
                faded_color, self.FONT_WEIGHT_NORMAL
            )

            y += self.LINE_HEIGHT_SMALL

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
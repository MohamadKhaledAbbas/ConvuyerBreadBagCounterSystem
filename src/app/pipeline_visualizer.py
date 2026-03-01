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
        debug_info: Optional[Dict] = None,
        ghost_tracks: Optional[List[Dict]] = None
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
            ghost_tracks: Optional list of ghost track info for visualization

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw conveyor ROI zone overlay (if enabled)
        if (self.tracking_config.conveyor_roi_enabled
                and self.tracking_config.conveyor_roi_show_overlay):
            self._draw_conveyor_roi(annotated)

        # Draw detections (blue boxes)
        self._draw_detections(annotated, detections)

        # Draw tracks (colored boxes with IDs and info)
        self._draw_tracks(annotated, tracks)

        # Draw ghost tracks (predicted positions with dashed lines)
        if ghost_tracks:
            self._draw_ghost_tracks(annotated, ghost_tracks)

        # Draw main status overlay (top-left)
        tentative_total = debug_info.get('tentative_total', 0) if debug_info else 0
        tentative_counts = debug_info.get('tentative_counts', {}) if debug_info else {}
        lost_track_count = debug_info.get('lost_track_count', 0) if debug_info else 0
        tracks_created = debug_info.get('tracks_created', 0) if debug_info else 0
        duplicates_prevented = debug_info.get('duplicates_prevented', 0) if debug_info else 0
        ghost_tracks_count = debug_info.get('ghost_tracks', 0) if debug_info else 0
        self._draw_status(
            annotated, fps, active_tracks, total_counted, counts_by_class,
            tentative_total, tentative_counts, lost_track_count,
            tracks_created, duplicates_prevented, ghost_tracks_count
        )

        # Draw pipeline debug panel (right side)
        if debug_info:
            self._draw_debug_panel(annotated, debug_info, tracks)

        # Draw state machine panel (bottom-right) when RLSM is active
        sm_info = debug_info.get('sm_info') if debug_info else None
        if sm_info:
            self._draw_sm_panel(annotated, sm_info)

        # Draw recent events log (bottom)
        if debug_info and 'recent_events' in debug_info:
            self._draw_event_log(annotated, debug_info['recent_events'])

        return annotated

    def _draw_conveyor_roi(self, frame: np.ndarray):
        """
        Draw conveyor ROI zone overlay.

        Shades the excluded areas with a dark tint and draws the active
        zone boundary as a dashed cyan rectangle.
        Uses a single overlay + addWeighted for correct alpha blending.
        """
        h, w = frame.shape[:2]
        cfg = self.tracking_config
        x1 = max(0, cfg.conveyor_roi_x_min)
        x2 = min(w, cfg.conveyor_roi_x_max)
        y1 = max(0, cfg.conveyor_roi_y_min)
        y2 = min(h, cfg.conveyor_roi_y_max)

        # Create overlay — copy frame, fill excluded regions with dark color
        overlay = frame.copy()
        shade = (15, 15, 25)
        # Left
        if x1 > 0:
            cv2.rectangle(overlay, (0, 0), (x1, h), shade, -1)
        # Right
        if x2 < w:
            cv2.rectangle(overlay, (x2, 0), (w, h), shade, -1)
        # Top (between left/right)
        if y1 > 0:
            cv2.rectangle(overlay, (x1, 0), (x2, y1), shade, -1)
        # Bottom (between left/right)
        if y2 < h:
            cv2.rectangle(overlay, (x1, y2), (x2, h), shade, -1)

        # Blend once
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Draw dashed zone boundary (cyan)
        boundary_color = (200, 200, 50)  # Cyan-ish
        dash_len = 12
        gap_len = 8

        for edge in [
            ((x1, y1), (x2, y1)),  # Top
            ((x1, y2), (x2, y2)),  # Bottom
            ((x1, y1), (x1, y2)),  # Left
            ((x2, y1), (x2, y2)),  # Right
        ]:
            pt1, pt2 = edge
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            length = max(abs(dx), abs(dy))
            if length == 0:
                continue
            for i in range(0, length, dash_len + gap_len):
                s = i / length
                e = min((i + dash_len) / length, 1.0)
                sp = (int(pt1[0] + dx * s), int(pt1[1] + dy * s))
                ep = (int(pt1[0] + dx * e), int(pt1[1] + dy * e))
                cv2.line(frame, sp, ep, boundary_color, 1, cv2.LINE_AA)

        # Small label at top of zone
        cv2.putText(frame, "CONVEYOR ZONE", (x1 + 6, y1 + 14),
                    self.FONT, 0.4, boundary_color, 1, cv2.LINE_AA)

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

    def _draw_ghost_tracks(self, frame: np.ndarray, ghost_tracks: List[Dict]):
        """
        Draw ghost tracks with predicted positions.

        Shows where the tracker expects each ghost to be based on conveyor velocity.
        Uses dashed lines and semi-transparent styling to distinguish from active tracks.

        Args:
            frame: Frame to draw on
            ghost_tracks: List of ghost track info from tracker.get_ghost_tracks_for_visualization()
        """
        ghost_color = (100, 100, 255)  # Light red/pink for ghosts
        prediction_color = (255, 150, 100)  # Light blue for predicted path

        for ghost in ghost_tracks:
            track_id = ghost['track_id']
            pred_bbox = ghost['predicted_bbox']
            original_pos = ghost['original_pos']
            predicted_pos = ghost['predicted_pos']
            elapsed = ghost['elapsed_seconds']
            velocity = ghost['velocity']
            hits = ghost['hits']

            x1, y1, x2, y2 = pred_bbox

            # Skip if bbox is completely off-screen
            frame_h, frame_w = frame.shape[:2]
            if x2 < 0 or x1 > frame_w or y2 < 0 or y1 > frame_h:
                continue

            # Draw dashed rectangle for predicted position
            # Using dotted line effect
            dash_length = 8
            gap_length = 6

            # Draw dashed box edges
            def draw_dashed_line(pt1, pt2, color, thickness=2):
                dist = int(np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2))
                if dist == 0:
                    return
                dx = (pt2[0] - pt1[0]) / dist
                dy = (pt2[1] - pt1[1]) / dist
                i = 0
                while i < dist:
                    start_x = int(pt1[0] + i * dx)
                    start_y = int(pt1[1] + i * dy)
                    end_i = min(i + dash_length, dist)
                    end_x = int(pt1[0] + end_i * dx)
                    end_y = int(pt1[1] + end_i * dy)
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
                    i += dash_length + gap_length

            # Draw the dashed rectangle
            draw_dashed_line((x1, y1), (x2, y1), ghost_color)  # Top
            draw_dashed_line((x2, y1), (x2, y2), ghost_color)  # Right
            draw_dashed_line((x2, y2), (x1, y2), ghost_color)  # Bottom
            draw_dashed_line((x1, y2), (x1, y1), ghost_color)  # Left

            # Draw line from original position to predicted position
            cv2.arrowedLine(
                frame,
                (int(original_pos[0]), int(original_pos[1])),
                (int(predicted_pos[0]), int(predicted_pos[1])),
                prediction_color,
                2,
                tipLength=0.2
            )

            # Draw small circle at original position
            cv2.circle(frame, (int(original_pos[0]), int(original_pos[1])), 5, ghost_color, -1)

            # Label with ghost info
            label = f"GHOST T{track_id} ({elapsed:.1f}s)"

            # Show velocity info with source indicator
            vel_source = ghost.get('velocity_source', 'default')
            learned_vel = ghost.get('learned_velocity_px_sec')
            if vel_source == 'learned' and learned_vel:
                vel_label = f"vel={velocity[1]:.1f}px/f (learned: {learned_vel:.0f}px/s)"
            else:
                vel_label = f"vel={velocity[1]:.1f}px/f (default)"

            text_size = cv2.getTextSize(label, self.FONT, self.FONT_SCALE_SMALL, 1)[0]
            label_y = y1 - 25
            if label_y < 20:
                label_y = y2 + 20

            # Background for label - wider to fit velocity info
            vel_text_size = cv2.getTextSize(vel_label, self.FONT, self.FONT_SCALE_SMALL, 1)[0]
            bg_width = max(text_size[0], vel_text_size[0]) + 12
            cv2.rectangle(
                frame,
                (x1, label_y - text_size[1] - 4),
                (x1 + bg_width, label_y + 18),
                (40, 40, 50),
                -1
            )

            # Ghost label
            cv2.putText(
                frame, label,
                (x1 + 4, label_y),
                self.FONT, self.FONT_SCALE_SMALL,
                ghost_color, 1
            )

            # Velocity label below
            cv2.putText(
                frame, vel_label,
                (x1 + 4, label_y + 14),
                self.FONT, self.FONT_SCALE_SMALL,
                prediction_color, 1
            )

    def _draw_status(
        self,
        frame: np.ndarray,
        fps: float,
        active_tracks: int,
        total_counted: int,
        counts_by_class: Dict[str, int],
        tentative_total: int,
        tentative_counts: Dict[str, int],
        lost_track_count: int = 0,
        tracks_created: int = 0,
        duplicates_prevented: int = 0,
        ghost_tracks: int = 0
    ):
        """Draw main status panel with clean modern design."""
        # Calculate panel dimensions
        num_classes = max(len(counts_by_class), 1)
        panel_w = 280
        panel_h = 420 + (num_classes * self.LINE_HEIGHT_SMALL)  # Increased height for new info
        panel_x = 15
        panel_y = 15

        # Draw panel background
        self._draw_panel_background(frame, panel_x, panel_y, panel_w, panel_h)

        # Draw header
        y = self._draw_header(frame, "COUNTER STATUS", panel_x, panel_y, panel_w)
        y += 5

        x = panel_x + self.PANEL_PADDING

        # App version (small text, top of panel)
        from src.config.settings import config
        version_text = f"v{config.APP_VERSION}"
        cv2.putText(
            frame, version_text, (x, y + 10),
            self.FONT, self.FONT_SCALE_SMALL - 0.05,
            self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL
        )
        y += self.LINE_HEIGHT_SMALL

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
        y += self.LINE_HEIGHT_MEDIUM

        # Lost track count
        cv2.putText(
            frame, f"Lost Tracks: {lost_track_count}", (x, y + 12),
            self.FONT, self.FONT_SCALE_MEDIUM,
            self.COLORS['text_error'], self.FONT_WEIGHT_NORMAL
        )
        y += self.LINE_HEIGHT_MEDIUM

        # Tracks created (total tracks created this session)
        cv2.putText(
            frame, f"Tracks Created: {tracks_created}", (x, y + 12),
            self.FONT, self.FONT_SCALE_MEDIUM,
            self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL
        )
        y += self.LINE_HEIGHT_MEDIUM

        # Duplicates prevented (artifact tracks prevented)
        if duplicates_prevented > 0:
            cv2.putText(
                frame, f"Duplicates Prevented: {duplicates_prevented}", (x, y + 12),
                self.FONT, self.FONT_SCALE_MEDIUM,
                self.COLORS['text_success'], self.FONT_WEIGHT_NORMAL
            )
            y += self.LINE_HEIGHT_MEDIUM

        # Ghost tracks (currently in recovery buffer)
        if ghost_tracks > 0:
            cv2.putText(
                frame, f"Ghost Tracks: {ghost_tracks}", (x, y + 12),
                self.FONT, self.FONT_SCALE_MEDIUM,
                self.COLORS['text_warning'], self.FONT_WEIGHT_NORMAL
            )
            y += self.LINE_HEIGHT_MEDIUM

        y += 10

        # Divider
        cv2.line(frame, (x, y), (panel_x + panel_w - self.PANEL_PADDING, y),
                self.COLORS['panel_border'], 1)
        y += 15

        # Total count (pending + persisted) - NEW
        total_all = total_counted + tentative_total
        cv2.putText(
            frame, "Total (All)", (x, y + 10),
            self.FONT, self.FONT_SCALE_SMALL,
            self.COLORS['text_info'], self.FONT_WEIGHT_NORMAL
        )

        # Total count badge
        total_text = str(total_all)
        total_size = cv2.getTextSize(total_text, self.FONT, self.FONT_SCALE_LARGE, self.FONT_WEIGHT_BOLD)[0]
        total_x = panel_x + panel_w - self.PANEL_PADDING - total_size[0] - 16

        cv2.rectangle(
            frame,
            (total_x, y - 2),
            (total_x + total_size[0] + 16, y + total_size[1] + 8),
            self.COLORS['text_info'],
            -1
        )
        cv2.putText(
            frame, total_text, (total_x + 8, y + total_size[1] + 2),
            self.FONT, self.FONT_SCALE_LARGE,
            (30, 30, 35), self.FONT_WEIGHT_BOLD
        )
        y += self.LINE_HEIGHT_LARGE + 10

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
        panel_y = 75

        # Draw panel background
        self._draw_panel_background(frame, panel_x, panel_y, panel_w, panel_h)

        # Draw header
        y = self._draw_header(frame, "PIPELINE DEBUG", panel_x, panel_y, panel_w)
        y += 8

        x = panel_x + self.PANEL_PADDING
        content_w = panel_w - (self.PANEL_PADDING * 2)

        # Pipeline stages
        stages = [
            ("DETECT",     f"{len(tracks)} objects",                           "🔍"),
            ("TRACK",      f"{len(tracks)} active",                            "📍"),
            ("ROI COLLECT",f"+{debug_info.get('rois_collected', 0)}",          "📷"),
            ("CLASSIFY",   f"{debug_info.get('pending_classify', 0)} queue",   "🏷"),
        ]

        # SMOOTH stage — richer info when RLSM is active
        sm_info = debug_info.get('sm_info')
        if sm_info:
            sm_state_short = {
                'ACCUMULATING':    'ACCUM',
                'CONFIRMED_BATCH': 'LOCKED',
                'TRANSITION':      'TRANSIT',
            }.get(sm_info['state'], sm_info['state'])
            smooth_info = f"{sm_state_short} {sm_info['run_length']}/{sm_info['run_target']}"
        else:
            smooth_info = f"{debug_info.get('pending_smooth', 0)} batch"

        stages.append(("SMOOTH", smooth_info, "📊"))
        stages.append(("COUNT",  f"{debug_info.get('rejected', 0)} reject", "✓"))

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

    def _draw_sm_panel(self, frame: np.ndarray, sm_info: Dict):
        """
        Draw a compact Run-Length State Machine panel in the bottom-right corner.

        Shows:
        - State badge (color-coded) + confirmed batch class on same row
        - Current run: class x evidence / target (with progress bar)
        - Smoothing stats: corrected / total items
        - Last decision reason
        - Transition count
        """
        h, w = frame.shape[:2]

        # State -> display color
        STATE_COLORS = {
            'ACCUMULATING':    (30,  210, 220),   # yellow (BGR)
            'CONFIRMED_BATCH': (80,  230, 100),   # green
            'TRANSITION':      (80,  80,  250),   # red
        }
        state = sm_info.get('state', 'ACCUMULATING')
        state_color = STATE_COLORS.get(state, self.COLORS['text_secondary'])

        panel_w = 300
        panel_h = 220
        panel_x = w - panel_w - 15
        panel_y = h - panel_h - 15

        self._draw_panel_background(frame, panel_x, panel_y, panel_w, panel_h)
        y = self._draw_header(frame, "STATE MACHINE", panel_x, panel_y, panel_w)
        y += 6

        x     = panel_x + self.PANEL_PADDING
        right = panel_x + panel_w - self.PANEL_PADDING
        content_w = panel_w - self.PANEL_PADDING * 2

        # ── Row 1: State badge + batch class ─────────────────────────────────
        STATE_LABELS = {
            'ACCUMULATING':    'ACCUM',
            'CONFIRMED_BATCH': 'LOCKED',
            'TRANSITION':      'TRANSIT',
        }
        badge_text = STATE_LABELS.get(state, state[:8])
        badge_w = cv2.getTextSize(badge_text, self.FONT, self.FONT_SCALE_SMALL, self.FONT_WEIGHT_BOLD)[0][0] + 14
        cv2.rectangle(frame, (x, y), (x + badge_w, y + 20), state_color, -1)
        cv2.putText(
            frame, badge_text, (x + 7, y + 14),
            self.FONT, self.FONT_SCALE_SMALL,
            (20, 20, 25), self.FONT_WEIGHT_BOLD
        )

        # Batch class right of badge
        batch_class = sm_info.get('batch_class') or '...'
        bc_trunc = batch_class if len(batch_class) <= 16 else batch_class[:13] + ".."
        bc_color = self.COLORS['text_success'] if batch_class != '...' else self.COLORS['text_secondary']
        cv2.putText(frame, bc_trunc, (x + badge_w + 10, y + 14),
                    self.FONT, self.FONT_SCALE_SMALL,
                    bc_color, self.FONT_WEIGHT_BOLD)
        y += 28

        # ── Row 2: Run progress label ────────────────────────────────────────
        run_class  = sm_info.get('run_class')  or '...'
        run_len    = sm_info.get('run_length',  0)
        run_target = sm_info.get('run_target',  5)
        run_pct    = min(1.0, run_len / run_target) if run_target > 0 else 0.0

        if state == 'TRANSITION':
            run_label = f"Evidence: {run_class[:10]} {run_len}/{run_target}"
        elif state == 'ACCUMULATING':
            run_label = f"Run: {run_class[:10]} {run_len}/{run_target}"
        else:
            run_label = f"Run: {run_class[:10]} {run_len}/{run_target}"

        cv2.putText(frame, run_label, (x, y + 12),
                    self.FONT, self.FONT_SCALE_SMALL,
                    state_color, self.FONT_WEIGHT_NORMAL)
        y += self.LINE_HEIGHT_MEDIUM

        # ── Row 3: Progress bar ──────────────────────────────────────────────
        bar_w  = content_w
        bar_h  = 10
        bar_y  = y
        # Background track
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + bar_h),
                      (60, 60, 70), -1)
        # Filled portion
        filled = int(bar_w * run_pct)
        if filled > 0:
            cv2.rectangle(frame, (x, bar_y), (x + filled, bar_y + bar_h),
                          state_color, -1)
        # Threshold marker (subtle white tick at target)
        if run_target > 0:
            tick_x = x + int(bar_w * min(1.0, 1.0))  # always at 100%
            cv2.line(frame, (tick_x, bar_y), (tick_x, bar_y + bar_h),
                     (200, 200, 200), 1)
        y += bar_h + 8

        # ── Row 4: Smoothing stats ───────────────────────────────────────────
        total_smoothed = sm_info.get('total_smoothed', 0)
        total_records  = sm_info.get('total_records', 0)
        if total_records > 0:
            rate_pct = (total_smoothed / total_records) * 100
            stats_text = f"Smoothed: {total_smoothed}/{total_records} ({rate_pct:.0f}%)"
        else:
            stats_text = "Smoothed: 0/0"
        cv2.putText(frame, stats_text, (x, y + 12),
                    self.FONT, self.FONT_SCALE_SMALL,
                    self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL)

        # Max blip (right-aligned on same row)
        max_blip = sm_info.get('max_blip', 0)
        blip_text = f"blip:{max_blip}"
        blip_w = cv2.getTextSize(blip_text, self.FONT, self.FONT_SCALE_SMALL, self.FONT_WEIGHT_NORMAL)[0][0]
        cv2.putText(frame, blip_text, (right - blip_w, y + 12),
                    self.FONT, self.FONT_SCALE_SMALL,
                    self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL)
        y += self.LINE_HEIGHT_MEDIUM

        # ── Row 5: Last decision ─────────────────────────────────────────────
        decision = sm_info.get('last_decision') or 'N/A'
        # Humanize common decision reasons
        DECISION_DISPLAY = {
            'matches_batch':       'Batch match',
            'transition_start':    'Transition started',
            'rejected_to_batch':   'Rejected -> batch',
        }
        if decision in DECISION_DISPLAY:
            decision_display = DECISION_DISPLAY[decision]
        elif decision.startswith('batch_transition:'):
            decision_display = decision.replace('batch_transition:', 'Batch: ')
        elif decision.startswith('blip_absorbed:'):
            decision_display = 'Blip absorbed'
        elif decision.startswith('transition_resolved'):
            decision_display = 'Resolved -> new'
        else:
            decision_display = decision

        decision_short = decision_display if len(decision_display) <= 26 else decision_display[:23] + ".."
        cv2.putText(frame, "Last:", (x, y + 12),
                    self.FONT, self.FONT_SCALE_SMALL,
                    self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL)
        # Color-code decision: green for batch match, orange for smoothed, red for transition
        if 'match' in decision_display.lower():
            dec_color = self.COLORS['text_success']
        elif 'transition' in decision_display.lower() or 'Transit' in decision_display:
            dec_color = (80, 80, 250)  # red
        elif 'blip' in decision_display.lower() or 'absorbed' in decision_display.lower():
            dec_color = self.COLORS['text_warning']
        elif 'resolved' in decision_display.lower() or 'Rejected' in decision_display:
            dec_color = self.COLORS['text_info']
        else:
            dec_color = self.COLORS['text_info']

        cv2.putText(frame, decision_short, (x + 42, y + 12),
                    self.FONT, self.FONT_SCALE_SMALL,
                    dec_color, self.FONT_WEIGHT_NORMAL)
        y += self.LINE_HEIGHT_MEDIUM

        # ── Row 6: Transitions count ─────────────────────────────────────────
        t_count = sm_info.get('transition_count', 0)
        cv2.putText(frame, f"Batch changes: {t_count}", (x, y + 12),
                    self.FONT, self.FONT_SCALE_SMALL,
                    self.COLORS['text_secondary'], self.FONT_WEIGHT_NORMAL)

    def _draw_event_log(self, frame: np.ndarray, recent_events: List[Tuple[float, str]]):
        """Draw recent events log with modern styling."""
        h, w = frame.shape[:2]

        if not recent_events:
            return

        # Get last 5 events
        events_to_show = recent_events[-5:]

        # Calculate panel dimensions
        panel_w = 400
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
"""Helpers for optional dev-only event-video sync overlays."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def draw_sync_debug_overlay(
    frame: np.ndarray,
    *,
    event_id: str,
    camera: str,
    capture_monotonic: float,
    anchor_monotonic: Optional[float],
) -> np.ndarray:
    """Return a copy of ``frame`` with a compact sync-debug overlay."""
    if frame is None or frame.size == 0:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 if w < 640 else 0.62
    thickness = 1 if w < 640 else 2
    pad_x = 10 if w < 640 else 14
    pad_y = 8 if w < 640 else 10
    line_gap = 4 if w < 640 else 6
    box_x = 12 if w < 640 else 18
    box_y = 12 if h < 480 else 18

    rel_s = (
        capture_monotonic - anchor_monotonic
        if anchor_monotonic is not None
        else None
    )
    event_tail = event_id[-28:] if len(event_id) > 28 else event_id
    lines = [
        f"SYNC {camera.upper()}",
        f"event={event_tail}",
        (
            f"t={rel_s:+.3f}s  mono={capture_monotonic:.3f}"
            if rel_s is not None
            else f"mono={capture_monotonic:.3f}"
        ),
    ]

    sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0]
        for line in lines
    ]
    text_w = max((sz[0] for sz in sizes), default=0)
    text_h = sum(sz[1] for sz in sizes) + max(0, len(lines) - 1) * line_gap
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2

    overlay = out.copy()
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (16, 18, 27),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0.0, out)
    cv2.rectangle(
        out,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (56, 189, 248) if camera.lower() == "qr" else (167, 139, 250),
        1,
    )

    cursor_y = box_y + pad_y
    for idx, (line, size) in enumerate(zip(lines, sizes)):
        baseline_y = cursor_y + size[1]
        color = (255, 255, 255) if idx == 0 else (220, 226, 235)
        cv2.putText(
            out,
            line,
            (box_x + pad_x, baseline_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cursor_y = baseline_y + line_gap

    return out

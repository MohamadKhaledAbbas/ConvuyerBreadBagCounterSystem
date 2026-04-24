"""Content camera recording (3D angle view of container contents)."""

from src.container.content.ContentCameraRecorder import (
    ContentCameraRecorder,
    ContentRecorderConfig,
)
from src.container.content.EventFrameBuffer import (
    EventFrameBuffer,
    EventFrameBufferConfig,
    FrameEntry,
)
from src.container.content.EventVideoCoordinator import (
    EventVideoCoordinator,
    EventVideoResult,
)
from src.container.content.SyncDebugOverlay import (
    draw_sync_debug_overlay,
)

__all__ = [
    "ContentCameraRecorder",
    "ContentRecorderConfig",
    "EventFrameBuffer",
    "EventFrameBufferConfig",
    "FrameEntry",
    "EventVideoCoordinator",
    "EventVideoResult",
    "draw_sync_debug_overlay",
]

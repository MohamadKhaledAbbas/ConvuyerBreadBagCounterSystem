"""
Application constants for ConvuyerBreadBagCounterSystem.
"""

# Configuration keys for database - Bread Camera
is_development_key = 'is_development'
rtsp_username = "rtsp_username"
rtsp_password = "rtsp_password"
rtsp_host = "rtsp_host"
rtsp_port = "rtsp_port"
is_profiler_enabled = "is_profiler_enabled"
enable_display_key = "enable_display"
enable_recording_key = "enable_recording"
snapshot_requested_key = "snapshot_requested"  # Flag for on-demand snapshot capture

# Configuration keys for database - Container Camera (Sale Point / صالة)
container_rtsp_host = "container_rtsp_host"  # Default: 192.168.2.118
container_rtsp_port = "container_rtsp_port"  # Same as bread camera port
container_rtsp_username = "container_rtsp_username"  # Same credentials
container_rtsp_password = "container_rtsp_password"  # Same credentials
container_snapshot_requested_key = "container_snapshot_requested"
container_enable_display_key = "container_enable_display"

# Container tracking configuration keys
container_exit_zone_ratio = "container_exit_zone_ratio"  # Default: 0.15
container_lost_timeout = "container_lost_timeout"  # Default: 2.0 seconds
container_pre_event_seconds = "container_pre_event_seconds"  # Default: 5.0
container_post_event_seconds = "container_post_event_seconds"  # Default: 5.0
container_detect_interval = "container_detect_interval"  # Default: 3 (detect every N-th frame)

# Minimum number of confirmed QR detections before a track emits an event.
# Filters out single-frame decoder glitches that would otherwise count as
# containers.  Set to 1 to disable (legacy behaviour).
container_min_detections_for_event = "container_min_detections_for_event"  # Default: 3

# Event-video recording (QR-camera side):
# Independent of detect_interval \u2014 every frame is considered for the
# per-track ring buffer used to build the event clip when the content
# camera is unavailable or when event_video_source == "qr".
container_event_video_fps = "container_event_video_fps"              # Default: 20 (fps)
container_event_video_max_seconds = "container_event_video_max_seconds"  # Default: 5.0 (hard cap)
container_event_video_stationary_px = "container_event_video_stationary_px"  # Default: 5 (dedup px)

# Event video recording:
#   "qr"      -> encode from QR camera frames (always available, single camera)
#   "content" -> use content camera ring buffer (3D angle); falls back to "qr" if unavailable
container_event_video_source = "container_event_video_source"  # Default: "qr"

# Content Camera (3D angle view of container contents, 192.168.2.128)
content_rtsp_host = "content_rtsp_host"        # Default: 192.168.2.128
content_rtsp_port = "content_rtsp_port"        # Default: 554
content_rtsp_username = "content_rtsp_username"
content_rtsp_password = "content_rtsp_password"
content_recording_enabled_key = "content_recording_enabled"   # '1' to enable content recording
content_pre_event_seconds = "content_pre_event_seconds"       # Default: 3.0 (pre-roll before event)
content_post_event_seconds = "content_post_event_seconds"     # Default: 2.0 (continue recording after exit)
content_buffer_seconds = "content_buffer_seconds"             # Default: 5.0 (rolling buffer size)
content_video_fps = "content_video_fps"                       # Default: 20 (output video fps)
content_rtsp_subtype = "content_rtsp_subtype"                 # 0=main stream (720p), 1=sub-stream (360p, higher fps)
content_max_recording_seconds = "content_max_recording_seconds" # Default: 15.0 (safety cap for begin/end recordings)

# Container data retention / automatic purge
# Mirrors the pattern used by pipeline_core.py for classified ROIs.
container_snapshots_retention_hours   = "container_snapshots_retention_hours"   # Default: 72  (3 days)
container_snapshots_max_count         = "container_snapshots_max_count"         # Default: 500
container_content_videos_retention_hours = "container_content_videos_retention_hours"  # Default: 72 (3 days)
container_content_videos_max_count    = "container_content_videos_max_count"    # Default: 200
container_db_events_retention_hours   = "container_db_events_retention_hours"   # Default: 168 (7 days)
container_purge_interval_minutes      = "container_purge_interval_minutes"      # Default: 60

CONFIG_KEYS = [
    is_development_key,
    rtsp_username,
    rtsp_password,
    rtsp_host,
    rtsp_port,
    is_profiler_enabled,
    enable_display_key,
    enable_recording_key,
    snapshot_requested_key,
]

# Container configuration keys (separate list for clarity)
CONTAINER_CONFIG_KEYS = [
    container_rtsp_host,
    container_rtsp_port,
    container_rtsp_username,
    container_rtsp_password,
    container_snapshot_requested_key,
    container_enable_display_key,
    container_exit_zone_ratio,
    container_lost_timeout,
    container_pre_event_seconds,
    container_post_event_seconds,
    container_detect_interval,
    container_min_detections_for_event,
    container_event_video_source,
    container_event_video_fps,
    container_event_video_max_seconds,
    container_event_video_stationary_px,
]

# Content camera configuration keys
CONTENT_CONFIG_KEYS = [
    content_rtsp_host,
    content_rtsp_port,
    content_rtsp_username,
    content_rtsp_password,
    content_recording_enabled_key,
    content_pre_event_seconds,
    content_post_event_seconds,
    content_buffer_seconds,
    content_video_fps,
    content_rtsp_subtype,
    content_max_recording_seconds,
]

# All configuration keys
ALL_CONFIG_KEYS = CONFIG_KEYS + CONTAINER_CONFIG_KEYS + CONTENT_CONFIG_KEYS

"""
Centralized path configuration for ConvuyerBreadBagCounterSystem.

**Single place to change all storage paths** when deploying to a
portable device, different directory, or mounted volume.

Every module that needs a file/directory path MUST import from here
instead of hardcoding strings.

Environment variables override every path:
    DATA_DIR            → base for db, recordings, pipeline state, classes
    SPOOL_DIR           → video segment spool files
    LOG_DIR             → application log files
    TMP_STATUS_DIR      → cross-process IPC JSON status files
                          (default: /tmp on RDK for RAM-backed speed)
    DB_PATH             → SQLite database file (overrides DATA_DIR/db/...)
    RECORDING_DIR       → recording output directory
    PIPELINE_STATE_FILE → pipeline state JSON file
"""

import os

from src.utils.platform import IS_RDK

# ============================================================================
# Base directories — change DATA_DIR to relocate db/logs/recordings/classes
# ============================================================================

DATA_DIR: str = os.getenv(
    "DATA_DIR",
    "/media/USB_DRIVE/ConvuyerBreadCounting/data" if IS_RDK else "data",
)

# Spool: video segment files + processor state.
SPOOL_DIR: str = os.getenv(
    "SPOOL_DIR",
    os.path.join(DATA_DIR, "spool"),
)

# Logs
LOG_DIR: str = os.getenv("LOG_DIR", os.path.join(DATA_DIR, "logs"))

# Cross-process IPC status files.
# /tmp is RAM-backed tmpfs on RDK — fast and avoids eMMC wear.
TMP_STATUS_DIR: str = os.getenv("TMP_STATUS_DIR", "/tmp")

# ============================================================================
# Database
# ============================================================================

DB_DIR: str = os.path.join(DATA_DIR, "db")
DB_PATH: str = os.getenv("DB_PATH", os.path.join(DB_DIR, "bag_events.db"))

# ============================================================================
# Spool-derived paths
# ============================================================================

SPOOL_PROCESSOR_STATE_FILE: str = os.path.join(SPOOL_DIR, "processor_state.json")
SPOOL_TMP_PATTERN: str = os.path.join(SPOOL_DIR, "*.tmp")

# ============================================================================
# Cross-process status files (IPC via JSON on disk)
# ============================================================================

CODEC_HEALTH_STATUS_FILE: str = os.path.join(TMP_STATUS_DIR, "codec_health_status.json")
SPOOL_PROCESSOR_STATUS_FILE: str = os.path.join(TMP_STATUS_DIR, "spool_processor_status.json")
SPOOL_RECORDER_STATUS_FILE: str = os.path.join(TMP_STATUS_DIR, "spool_recorder_status.json")
PIPELINE_THROTTLE_STATE_FILE: str = os.path.join(TMP_STATUS_DIR, "pipeline_throttle.json")

# ============================================================================
# Other data paths
# ============================================================================

PIPELINE_STATE_FILE: str = os.getenv(
    "PIPELINE_STATE_FILE",
    os.path.join(DATA_DIR, "pipeline_state.json"),
)
RECORDING_DIR: str = os.getenv(
    "RECORDING_DIR",
    os.path.join(DATA_DIR, "recordings"),
)
KNOWN_CLASSES_DIR: str = os.path.join(DATA_DIR, "classes")
UNKNOWN_CLASSES_DIR: str = os.path.join(DATA_DIR, "unknown")
SNAPSHOT_DIR: str = os.getenv("SNAPSHOT_DIR", os.path.join(DATA_DIR, "snapshot"))
CONVEYOR_ROI_FILE: str = os.path.join(DATA_DIR, "conveyor_roi.json")
ROI_CANDIDATES_DIR: str = os.path.join(DATA_DIR, "roi_candidates")
CLASSIFIED_ROIS_DIR: str = os.path.join(DATA_DIR, "classified_rois")
LOST_SNAPSHOTS_DIR: str = os.path.join(SPOOL_DIR, "lost_snapshots")

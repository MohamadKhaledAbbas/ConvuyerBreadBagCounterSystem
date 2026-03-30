"""
Centralized path configuration for ConvuyerBreadBagCounterSystem.

**Single place to change all storage paths** when deploying to a
portable device, different directory, or mounted volume.

Every module that needs a file/directory path MUST import from here
instead of hardcoding strings.

Environment variables override every path:
    ROOT_SSD_DRIVE     → mounted SSD/USB root used to build DATA_DIR
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
import subprocess

from src.utils.platform import IS_LINUX, IS_RDK

# ============================================================================
# Base directories — change DATA_DIR to relocate db/logs/recordings/classes
# ============================================================================


_APP_DIR_NAME = "ConvuyerBreadCounting"


def _resolve_root_ssd_drive() -> str:
    """Resolve the mounted SSD/USB root that contains the application data directory.

    Resolution order:
      1. ``ROOT_SSD_DRIVE`` env var — explicit override, used as-is.
      2. Query the live mount table (``findmnt``) for all removable drives
         mounted under ``/media/``.  Among those candidates, return the first
         one whose root contains a ``ConvuyerBreadCounting`` directory —
         regardless of what the drive or mountpoint is named.
      3. If no drive has the application directory yet, return the first
         removable mount found (so the app can bootstrap itself on a fresh
         drive on first run).
      4. Return ``""`` to fall back to the local ``data/`` directory.
    """
    env_root = os.getenv("ROOT_SSD_DRIVE")
    if env_root:
        return env_root

    if not IS_LINUX:
        return ""

    try:
        result = subprocess.run(
            [
                "findmnt",
                "-rn",
                "-o",
                "TARGET,FSTYPE",
                "-t",
                "exfat,vfat,ntfs,fuseblk,ext4",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""

    if not result or result.returncode != 0:
        return ""

    candidates: list[str] = []
    # Internal RDK mount points that must never be used as data storage.
    _INTERNAL_MOUNTS = {"mass_storage", "sdcard1", "sdcard2"}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        target = parts[0]
        if not target.startswith("/media/"):
            continue
        # Exclude known internal/virtual RDK storage devices.
        mount_name = target.split("/")[-1]
        if mount_name in _INTERNAL_MOUNTS:
            continue
        candidates.append(target)

    # Prefer a drive that already has the application directory on it.
    for target in candidates:
        if os.path.isdir(os.path.join(target, _APP_DIR_NAME)):
            return target

    # No drive has the app directory yet — use the first available mount
    # so the app can create its data layout on a fresh drive.
    if candidates:
        return candidates[0]

    return ""


ROOT_SSD_DRIVE: str = _resolve_root_ssd_drive()

DEFAULT_DATA_DIR: str = (
    os.path.join(ROOT_SSD_DRIVE, "ConvuyerBreadCounting", "data")
    if ROOT_SSD_DRIVE
    else "data"
)

DATA_DIR: str = os.getenv(
    "DATA_DIR",
    DEFAULT_DATA_DIR,
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

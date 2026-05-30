"""
Centralized path configuration for ConvuyerBreadBagCounterSystem.

**Single place to change all storage paths** when deploying to a
portable device, different directory, or mounted volume.

Every module that needs a file/directory path MUST import from here
instead of hardcoding strings.

Environment variables override every path:
    ROOT_SSD_DRIVE     → mounted SSD/USB root used to build DATA_DIR
    DATA_DIR            → base for db, recordings, pipeline state, classes
    LOG_DIR             → application log files
    TMP_STATUS_DIR      → cross-process IPC JSON status files
                          (default: /tmp on RDK for RAM-backed speed)
    DB_PATH             → SQLite database file (overrides DATA_DIR/db/...)
    RECORDING_DIR       → recording output directory
    PIPELINE_STATE_FILE → pipeline state JSON file
"""

import os
import subprocess
import time

from src.utils.platform import IS_LINUX, IS_RDK

# ============================================================================
# Base directories — change DATA_DIR to relocate db/logs/recordings/classes
# ============================================================================

# Fixed project root on the RDK board. Used as the base for the local
# fallback data directory when no writable USB/SSD drive is found.
_PROJECT_ROOT: str = os.getenv("PROJECT_ROOT", "/home/sunrise/ConvuyerBreadCounting" if IS_RDK else ".")

_APP_DIR_NAME = "ConvuyerBreadCounting"

# How many times to retry detecting the SSD after running mount -a.
# Handles the case where paths.py is imported before auto-mount completes.
_MOUNT_RETRIES: int = int(os.getenv("MOUNT_RETRIES", "3"))
_MOUNT_RETRY_DELAY: float = 2.0


def _query_findmnt() -> list[str]:
    """Run findmnt and return candidate mount targets under /media/."""
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
        return []

    if not result or result.returncode != 0:
        return []

    candidates: list[str] = []
    _INTERNAL_MOUNTS = {"mass_storage", "sdcard1", "sdcard2"}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        target = parts[0]
        if not target.startswith("/media/"):
            continue
        mount_name = target.split("/")[-1]
        if mount_name in _INTERNAL_MOUNTS:
            continue
        candidates.append(target)
    return candidates


def _pick_best_candidate(candidates: list[str]) -> str:
    """Among candidates, prefer one with existing app dir, else first writable."""
    for target in candidates:
        if os.path.isdir(os.path.join(target, _APP_DIR_NAME)) and os.access(target, os.W_OK):
            return target
    for target in candidates:
        if os.access(target, os.W_OK):
            return target
    return ""


def _resolve_root_ssd_drive() -> str:
    """Resolve the mounted SSD/USB root that contains the application data directory.

    Resolution order:
      1. ``ROOT_SSD_DRIVE`` env var — explicit override, used as-is.
      2. Query the live mount table (``findmnt``) for all removable drives
         mounted under ``/media/``.
      3. If none found, run ``mount -a`` and retry (up to ``MOUNT_RETRIES``
         times, 2 s apart) to give systemd or fstab time to catch up.
      4. Return ``""`` to fall back to the local ``data/`` directory.
    """
    env_root = os.getenv("ROOT_SSD_DRIVE")
    if env_root:
        return env_root

    if not IS_LINUX:
        return ""

    # First pass — maybe it's already mounted.
    candidates = _query_findmnt()
    if candidates:
        picked = _pick_best_candidate(candidates)
        if picked:
            return picked
        # Drives exist but none are writable — mount -a won't help.
        return ""

    # Second pass — run mount -a and retry a few times (drive detected but
    # fstab mount hadn't run yet when findmnt first queried).
    for attempt in range(1, _MOUNT_RETRIES + 1):
        try:
            subprocess.run(
                ["mount", "-a"],
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(_MOUNT_RETRY_DELAY)
        candidates = _query_findmnt()
        if candidates:
            picked = _pick_best_candidate(candidates)
            if picked:
                return picked

    return ""


ROOT_SSD_DRIVE: str = _resolve_root_ssd_drive()

DEFAULT_DATA_DIR: str = (
    os.path.join(ROOT_SSD_DRIVE, "ConvuyerBreadCounting", "data")
    if ROOT_SSD_DRIVE
    else os.path.join(_PROJECT_ROOT, "data")
)

DATA_DIR: str = os.getenv(
    "DATA_DIR",
    DEFAULT_DATA_DIR,
)

# Local fallback path (used when no USB/SSD drive is mounted).
# Exported publicly so other modules can detect and merge storage splits.
FALLBACK_DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data")

# Spool directory (retained for lost_snapshots sub-directory)
SPOOL_DIR: str = os.getenv(
    "SPOOL_DIR",
    os.path.join(DATA_DIR, "spool"),
)

# Logs — always on the local SD/eMMC path so they survive SSD failures
# or read-only transitions.  Separate from DATA_DIR for this reason.
LOG_DIR: str = os.getenv("LOG_DIR", os.path.join(_PROJECT_ROOT, "data", "logs"))

# Cross-process IPC status files.
# /tmp is RAM-backed tmpfs on RDK — fast and avoids eMMC wear.
TMP_STATUS_DIR: str = os.getenv("TMP_STATUS_DIR", "/tmp")

# ============================================================================
# Database
# ============================================================================

DB_DIR: str = os.path.join(DATA_DIR, "db")
DB_PATH: str = os.getenv("DB_PATH", os.path.join(DB_DIR, "bag_events.db"))

# ============================================================================
# Cross-process status files (IPC via JSON on disk)
# ============================================================================

CODEC_HEALTH_STATUS_FILE: str = os.path.join(TMP_STATUS_DIR, "codec_health_status.json")

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
KNOWN_CLASSES_DIR: str = os.getenv(
    "KNOWN_CLASSES_DIR",
    os.path.join(_PROJECT_ROOT, "data", "classes"),
)
UNKNOWN_CLASSES_DIR: str = os.getenv(
    "UNKNOWN_CLASSES_DIR",
    os.path.join(_PROJECT_ROOT, "data", "unknown"),
)
SNAPSHOT_DIR: str = os.getenv("SNAPSHOT_DIR", os.path.join(DATA_DIR, "snapshot"))
CONVEYOR_ROI_FILE: str = os.path.join(DATA_DIR, "conveyor_roi.json")
ROI_CANDIDATES_DIR: str = os.path.join(DATA_DIR, "roi_candidates")
CLASSIFIED_ROIS_DIR: str = os.path.join(DATA_DIR, "classified_rois")
LOST_SNAPSHOTS_DIR: str = os.path.join(SPOOL_DIR, "lost_snapshots")

# Container tracking (sale point / صالة)
CONTAINER_PIPELINE_STATE_FILE: str = os.path.join(DATA_DIR, "container_pipeline_state.json")
CONTAINER_SNAPSHOT_DIR: str = os.path.join(DATA_DIR, "container_snapshots")

# Content camera recordings (3D-angle view of container contents)
CONTAINER_CONTENT_VIDEOS_DIR: str = os.path.join(DATA_DIR, "container_content_videos")

# Content camera 2 recordings (second side view)
CONTAINER_CONTENT2_VIDEOS_DIR: str = os.path.join(DATA_DIR, "container_content2_videos")

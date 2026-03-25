"""
Cross-process throttle state coordination.

The ConveyorCounterApp (breadcount-main) owns the throttle decision and
writes the current mode to a shared JSON file.  The SpoolProcessorNode
(breadcount-spool-processor), running in a **separate process**, reads
this file to decide whether to run at full speed or in power-saving
sentinel mode.

File format (``/tmp/pipeline_throttle.json``):
    {
        "mode": "full" | "degraded",
        "updated_at": 1711396800.0,
        "sentinel_interval_s": 1.0,
        "skip_n": 5
    }

Write strategy: atomic (write to .tmp, ``os.replace`` to .json).
Read  strategy: best-effort (fallback to ``"full"`` on error or stale data).

Staleness safety:
    If the file has not been refreshed within ``staleness_timeout_s``
    (default 120 s), ``read_throttle_state`` returns ``"full"`` regardless
    of the stored mode.  This guards against the main app crashing while
    the file still says ``"degraded"`` — the spool processor will revert
    to full-speed processing automatically.
"""

import json
import os
import time
from typing import Tuple

from src.utils.AppLogging import logger

# Default path for the shared throttle state file.
# /tmp is a RAM-backed tmpfs on the RDK, so reads/writes are fast and
# will not wear the eMMC.
DEFAULT_THROTTLE_STATE_PATH = "/tmp/pipeline_throttle.json"


def write_throttle_state(
    mode: str,
    sentinel_interval_s: float = 1.0,
    skip_n: int = 5,
    path: str = DEFAULT_THROTTLE_STATE_PATH,
) -> None:
    """
    Atomically write the throttle mode to the shared state file.

    Called by ConveyorCounterApp on mode transitions and periodically
    as a heartbeat (via ``_maybe_publish_state_periodic``).

    Args:
        mode:                ``"full"`` or ``"degraded"``.
        sentinel_interval_s: Seconds between sentinel probe frames
                             in degraded mode.
        skip_n:              The frame-skip factor (informational for
                             the processor to know the expected savings).
        path:                File path for the shared state.
    """
    state = {
        "mode": mode,
        "updated_at": time.time(),
        "sentinel_interval_s": sentinel_interval_s,
        "skip_n": skip_n,
    }
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.warning(f"[ThrottleState] Failed to write throttle state: {e}")


def read_throttle_state(
    path: str = DEFAULT_THROTTLE_STATE_PATH,
    staleness_timeout_s: float = 120.0,
) -> Tuple[str, float]:
    """
    Read the throttle mode from the shared state file.

    Returns ``("full", 1.0)`` if the file is missing, unreadable, or stale
    (not updated within ``staleness_timeout_s``).  This fail-safe ensures
    that a crashed main app never leaves the spool processor stuck in
    sentinel mode indefinitely.

    Args:
        path:                 File path for the shared state.
        staleness_timeout_s:  If the file's ``updated_at`` is older than
                              this many seconds, treat mode as ``"full"``.

    Returns:
        Tuple of (mode, sentinel_interval_s).
    """
    try:
        with open(path, "r") as f:
            state = json.load(f)

        mode = state.get("mode", "full")
        sentinel_interval_s = float(state.get("sentinel_interval_s", 1.0))
        updated_at = float(state.get("updated_at", 0.0))

        # Staleness check — if the main app hasn't refreshed the file
        # within the timeout, assume "full" for safety.
        age = time.time() - updated_at
        if age > staleness_timeout_s:
            logger.debug(
                f"[ThrottleState] Stale state file (age={age:.0f}s > "
                f"timeout={staleness_timeout_s:.0f}s) → defaulting to 'full'"
            )
            return "full", sentinel_interval_s

        return mode, sentinel_interval_s

    except FileNotFoundError:
        # Expected on first start before main app writes the file.
        return "full", 1.0
    except Exception as e:
        logger.debug(f"[ThrottleState] Failed to read throttle state: {e}")
        return "full", 1.0


def cleanup_throttle_state(path: str = DEFAULT_THROTTLE_STATE_PATH) -> None:
    """
    Write ``"full"`` mode to the state file on shutdown.

    Ensures the spool processor reverts to full-speed processing even
    if it reads the file before the staleness timeout kicks in.
    """
    write_throttle_state("full", path=path)


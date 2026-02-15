"""
Pipeline State - Cross-process shared state for real-time count visibility.

The main app (ConveyorCounterApp) writes pipeline state to a JSON file,
and the FastAPI server reads it to serve real-time count data.

This enables the /api/counts endpoint and SSE stream to show:
- Confirmed counts (persisted to DB after smoothing)
- Pending counts (in the smoothing window, awaiting batch validation)
- Just classified counts (tentative, before smoothing)
- Smoothing statistics and window status
"""

import json
import os
import time
from typing import Dict, Any, Optional

from src.utils.AppLogging import logger

# Default state file location (same data directory as database)
_DEFAULT_STATE_PATH = "data/pipeline_state.json"


def _get_state_path() -> str:
    """Get the state file path, checking env var at call time."""
    return os.getenv("PIPELINE_STATE_FILE", _DEFAULT_STATE_PATH)


def write_state(state: Dict[str, Any], state_file: Optional[str] = None) -> bool:
    """
    Write pipeline state to shared JSON file (called by ConveyorCounterApp).

    Args:
        state: Pipeline state dictionary with counts and smoothing info
        state_file: Optional custom path for the state file

    Returns:
        True if written successfully
    """
    filepath = state_file or _get_state_path()
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        state["_updated_at"] = time.time()

        # Write atomically via temp file to prevent partial reads
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, filepath)
        return True
    except Exception as e:
        logger.error(f"[PipelineState] Failed to write state: {e}")
        return False


def read_state(state_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Read pipeline state from shared JSON file (called by FastAPI server).

    Returns a default empty state if the file doesn't exist or is unreadable.

    Args:
        state_file: Optional custom path for the state file

    Returns:
        Pipeline state dictionary
    """
    filepath = state_file or _get_state_path()
    try:
        if not os.path.exists(filepath):
            return _empty_state()

        with open(filepath, "r") as f:
            state = json.load(f)

        return state
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[PipelineState] Failed to read state: {e}")
        return _empty_state()


def _empty_state() -> Dict[str, Any]:
    """Return default empty pipeline state."""
    return {
        "confirmed": {},
        "pending": {},
        "just_classified": {},
        "confirmed_total": 0,
        "pending_total": 0,
        "just_classified_total": 0,
        "smoothing_rate": 0.0,
        "window_status": {
            "size": 7,
            "current_items": 0,
            "next_confirmation_in": 7
        },
        "recent_events": [],
        "current_batch_type": None,
        "_updated_at": 0
    }

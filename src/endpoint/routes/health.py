"""
Health status endpoints for system diagnostics.

Provides API endpoints for checking the health of various system components
including the codec health monitor, frame processing, and ROS2 pipeline.

Architecture Note:
    main.py and the FastAPI server (run_endpoint.py) run as SEPARATE processes.
    The codec health monitor runs inside main.py and writes its status to a
    shared file (/tmp/codec_health_status.json). This endpoint reads that file
    so both processes can communicate without shared memory.
"""

import json
import os
import subprocess
import time
from typing import Dict, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.utils.AppLogging import logger
from src.utils.platform import is_rdk_platform

router = APIRouter(prefix="/api/health", tags=["health"])

# Path must match the one in codec_health_monitor.py
CODEC_HEALTH_STATUS_FILE = "/tmp/codec_health_status.json"


def _read_status_file() -> dict | None:
    """Read codec health status from the shared file written by main.py."""
    try:
        if os.path.exists(CODEC_HEALTH_STATUS_FILE):
            with open(CODEC_HEALTH_STATUS_FILE, 'r') as f:
                data = json.load(f)
            # Add freshness info
            ts = data.get("timestamp", 0)
            if ts:
                age = time.time() - ts
                data["age_seconds"] = round(age, 1)
                data["stale"] = age > 60  # Stale if older than 60s
            return data
    except Exception as e:
        logger.debug(f"[HealthAPI] Error reading codec status file: {e}")
    return None


@router.get("/codec")
async def get_codec_health() -> JSONResponse:
    """
    Get the health status of the codec (hobot_codec VPU decoder).

    Reads from a shared status file written by the codec health monitor
    running inside main.py.

    Returns:
        JSON with health status, statistics, and diagnostic information.
    """
    # Try reading from shared status file (cross-process communication)
    status = _read_status_file()

    if status is not None:
        return JSONResponse(content=status, status_code=200)

    # No status file — monitor hasn't started or not on RDK
    if not is_rdk_platform():
        return JSONResponse(
            content={
                "enabled": False,
                "reason": "Not on RDK platform — codec health monitor not applicable"
            },
            status_code=200
        )

    return JSONResponse(
        content={
            "enabled": False,
            "reason": "Codec health monitor status file not found. Is main.py running?"
        },
        status_code=200
    )


@router.post("/codec/restart")
async def restart_codec() -> JSONResponse:
    """
    Manually trigger a codec restart (for debugging/recovery).

    Directly kills hobot_codec. The ROS2 launch system should respawn it.
    Works even if the codec health monitor is not running.

    Returns:
        JSON with restart result.
    """
    if not is_rdk_platform():
        return JSONResponse(
            content={"success": False, "reason": "Not on RDK platform"},
            status_code=400
        )

    try:
        result = subprocess.run(
            ["pkill", "-9", "-f", "hobot_codec"],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            logger.warning("[HealthAPI] Manual codec restart: hobot_codec killed")
            return JSONResponse(
                content={
                    "success": True,
                    "message": "hobot_codec killed. ROS2 launch system should respawn it."
                },
                status_code=200
            )
        elif result.returncode == 1:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "No hobot_codec process found to kill"
                },
                status_code=404
            )
        else:
            return JSONResponse(
                content={
                    "success": False,
                    "message": f"pkill failed: {result.stderr}"
                },
                status_code=500
            )
    except Exception as e:
        logger.error(f"[HealthAPI] Error restarting codec: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@router.get("/pipeline")
async def get_pipeline_health() -> JSONResponse:
    """
    Get overall pipeline health status.

    Checks:
    - Codec health (reads shared status file)
    - Whether main.py is actively processing

    Returns:
        JSON with overall health status.
    """
    health: Dict[str, Any] = {
        "status": "healthy",
        "components": {}
    }

    # Check codec health from shared file
    codec_status = _read_status_file()
    if codec_status is not None:
        state = codec_status.get("state", "unknown")
        is_stale = codec_status.get("stale", False)
        codec_healthy = state == "healthy" and not is_stale

        health["components"]["codec"] = {
            "healthy": codec_healthy,
            "state": state,
            "stale": is_stale,
            "restarts_total": codec_status.get("restarts_total", 0),
            "age_seconds": codec_status.get("age_seconds"),
        }
        if not codec_healthy:
            health["status"] = "degraded"
    else:
        if is_rdk_platform():
            health["components"]["codec"] = {
                "healthy": False,
                "state": "unknown",
                "reason": "Status file not found — main.py may not be running"
            }
            health["status"] = "degraded"
        else:
            health["components"]["codec"] = {
                "healthy": True,
                "note": "Not applicable (not on RDK)"
            }

    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)

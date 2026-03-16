"""
Health status endpoints for system diagnostics.

Provides API endpoints for checking the health of various system components
including the codec health monitor, frame processing, ROS2 pipeline, and
monitoring logs.

Architecture Note:
    main.py and the FastAPI server (run_endpoint.py) run as SEPARATE processes.
    The codec health monitor runs inside main.py and writes its status to a
    shared file (/tmp/codec_health_status.json). This endpoint reads that file
    so both processes can communicate without shared memory.

    WARNING/ERROR/CRITICAL log messages are captured by a DB-backed handler
    and stored in the monitoring_logs table with 7-day retention.
"""

import json
import os
import subprocess
import time
from typing import Dict, Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.endpoint.shared import get_db
from src.utils.AppLogging import logger
from src.utils.platform import is_rdk_platform

router = APIRouter(prefix="/api/health", tags=["health"])

# Path must match the one in codec_health_monitor.py
CODEC_HEALTH_STATUS_FILE = "/tmp/codec_health_status.json"


class RecoverRequest(BaseModel):
    """Request body for manual pipeline recovery."""
    stage: int = Field(..., ge=1, le=4, description="Recovery stage (1-4)")


def _read_status_file() -> Optional[dict]:
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
    running inside main.py.  Includes recovery_events and health_checkpoints
    when available.

    Returns:
        JSON with health status, statistics, and diagnostic information.
    """
    status = _read_status_file()

    if status is not None:
        # Surface recovery events and health checkpoints at top level
        response = dict(status)
        response.setdefault("recovery_events", [])
        response.setdefault("health_checkpoints", {})
        return JSONResponse(content=response, status_code=200)

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
    - Spool health (from health checkpoints)
    - RTSP health (from health checkpoints)
    - Recovery stage

    Returns:
        JSON with overall health status.
    """
    health: Dict[str, Any] = {
        "status": "healthy",
        "components": {}
    }

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

        # Spool health from checkpoints
        checkpoints = codec_status.get("health_checkpoints", {})
        spool_cp = checkpoints.get("spool_input", {})
        health["components"]["spool"] = {
            "healthy": spool_cp.get("alive", False) if spool_cp else None,
            "reason": spool_cp.get("reason") if spool_cp else "no_data",
        }
        if spool_cp and not spool_cp.get("alive", False):
            health["status"] = "degraded"

        # RTSP health from checkpoints
        rtsp_cp = checkpoints.get("rtsp_ingest", {})
        health["components"]["rtsp"] = {
            "healthy": rtsp_cp.get("alive", False) if rtsp_cp else None,
            "reason": rtsp_cp.get("reason") if rtsp_cp else "no_data",
        }
        if rtsp_cp and not rtsp_cp.get("alive", False):
            health["status"] = "degraded"

        # Recovery stage
        health["recovery_stage"] = codec_status.get("current_recovery_stage", 1)
        health["escalation_count"] = codec_status.get("escalation_count", 0)
    else:
        if is_rdk_platform():
            health["components"]["codec"] = {
                "healthy": False,
                "state": "unknown",
                "reason": "Status file not found — main.py may not be running"
            }
            health["components"]["spool"] = {"healthy": None, "reason": "no_data"}
            health["components"]["rtsp"] = {"healthy": None, "reason": "no_data"}
            health["status"] = "degraded"
        else:
            health["components"]["codec"] = {
                "healthy": True,
                "note": "Not applicable (not on RDK)"
            }

    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)


@router.post("/pipeline/recover")
async def trigger_pipeline_recovery(body: RecoverRequest) -> JSONResponse:
    """
    Manually trigger pipeline recovery at a specified stage (1-4).

    Stages:
        1 - Restart hobot_codec only
        2 - Restart spool_processor + ros2
        3 - Restart full media stack (ros2 + spool-processor + spool-recorder)
        4 - Restart all services except uvicorn

    Returns:
        JSON with recovery result.
    """
    if not is_rdk_platform():
        return JSONResponse(
            content={"success": False, "reason": "Not on RDK platform"},
            status_code=400
        )

    stage = body.stage
    stage_commands = {
        1: None,  # handled specially below
        2: "sudo supervisorctl restart breadcount-ros2 breadcount-spool-processor",
        3: "sudo supervisorctl restart breadcount-ros2 breadcount-spool-processor breadcount-spool-recorder",
        4: "sudo supervisorctl restart breadcount-ros2 breadcount-spool-processor breadcount-spool-recorder breadcount-main",
    }

    try:
        if stage == 1:
            # Stage 1: kill hobot_codec, let ROS2 respawn it
            result = subprocess.run(
                ["pkill", "-f", "hobot_codec"],
                capture_output=True, text=True, timeout=10
            )
            logger.warning(f"[HealthAPI] Manual recovery stage 1: pkill hobot_codec rc={result.returncode}")
            return JSONResponse(
                content={
                    "success": True,
                    "stage": stage,
                    "message": "Stage 1: hobot_codec killed, awaiting respawn."
                },
                status_code=200
            )

        cmd = stage_commands[stage]
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True, text=True, timeout=30
        )

        ok = result.returncode == 0
        logger.warning(
            f"[HealthAPI] Manual recovery stage {stage}: rc={result.returncode} "
            f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )
        return JSONResponse(
            content={
                "success": ok,
                "stage": stage,
                "message": f"Stage {stage} recovery {'succeeded' if ok else 'failed'}.",
                "details": result.stderr.strip() if not ok else "",
            },
            status_code=200 if ok else 500
        )

    except Exception as e:
        logger.error(f"[HealthAPI] Recovery stage {stage} error: {e}")
        return JSONResponse(
            content={"success": False, "stage": stage, "error": str(e)},
            status_code=500
        )


@router.get("/logs")
async def get_monitoring_logs(
    level: Optional[str] = Query(None, description="Filter by level: WARNING, ERROR, CRITICAL"),
    limit: int = Query(100, ge=1, le=500, description="Max rows to return"),
    since_minutes: Optional[int] = Query(None, ge=1, description="Only logs from last N minutes"),
) -> JSONResponse:
    """
    Query monitoring log entries stored in the database.

    WARNING, ERROR, and CRITICAL log messages are automatically captured
    and stored with 7-day retention.  Use this endpoint to inspect recent
    issues without SSH-ing into the machine.

    Query params:
        level: Optional level filter (WARNING / ERROR / CRITICAL)
        limit: Max rows (default 100, max 500)
        since_minutes: Only return logs from last N minutes

    Returns:
        JSON with log entries and a summary.
    """
    try:
        db = get_db()
        logs = db.get_monitoring_logs(
            level=level,
            limit=limit,
            since_minutes=since_minutes,
        )
        summary = db.get_monitoring_log_summary()
        return JSONResponse(
            content={
                "logs": logs,
                "summary": summary,
                "filters": {
                    "level": level,
                    "limit": limit,
                    "since_minutes": since_minutes,
                },
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(f"[HealthAPI] Error querying monitoring logs: {e}")
        return JSONResponse(
            content={"error": str(e), "logs": []},
            status_code=500,
        )

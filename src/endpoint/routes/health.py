"""
Health status endpoints for system diagnostics.

Provides API endpoints for checking the health of various system components
including the codec health monitor, frame processing, and ROS2 pipeline.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Dict, Any

from src.utils.AppLogging import logger

router = APIRouter(prefix="/api/health", tags=["health"])

# Global reference to codec health monitor (set by app initialization)
_codec_health_monitor = None


def set_codec_health_monitor(monitor):
    """Set the codec health monitor reference for API access."""
    global _codec_health_monitor
    _codec_health_monitor = monitor


def get_codec_health_monitor():
    """Get the codec health monitor reference."""
    return _codec_health_monitor


@router.get("/codec")
async def get_codec_health() -> JSONResponse:
    """
    Get the health status of the codec (hobot_codec VPU decoder).

    Returns:
        JSON with health status, statistics, and diagnostic information.

    Example response:
        {
            "enabled": true,
            "state": "healthy",
            "checks_total": 100,
            "checks_healthy": 98,
            "checks_failed": 2,
            "restarts_total": 1,
            "restarts_this_hour": 1,
            "last_healthy": "2026-03-15T20:00:00",
            "last_restart": "2026-03-15T19:55:00",
            "last_failure_reason": null
        }
    """
    if _codec_health_monitor is None:
        return JSONResponse(
            content={
                "enabled": False,
                "reason": "Codec health monitor not initialized (not on RDK or not started)"
            },
            status_code=200
        )

    try:
        stats = _codec_health_monitor.get_stats()
        stats["enabled"] = True
        return JSONResponse(content=stats, status_code=200)
    except Exception as e:
        logger.error(f"[HealthAPI] Error getting codec health: {e}")
        return JSONResponse(
            content={
                "enabled": True,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/codec/restart")
async def restart_codec() -> JSONResponse:
    """
    Manually trigger a codec restart (for debugging/recovery).

    This bypasses the normal health check and immediately kills hobot_codec.
    The ROS2 launch system should respawn it automatically.

    Returns:
        JSON with restart result.
    """
    if _codec_health_monitor is None:
        return JSONResponse(
            content={
                "success": False,
                "reason": "Codec health monitor not initialized"
            },
            status_code=400
        )

    try:
        result = _codec_health_monitor.force_restart()
        return JSONResponse(
            content={
                "success": result,
                "message": "Restart initiated" if result else "Restart failed"
            },
            status_code=200 if result else 500
        )
    except Exception as e:
        logger.error(f"[HealthAPI] Error restarting codec: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/pipeline")
async def get_pipeline_health() -> JSONResponse:
    """
    Get overall pipeline health status.

    Checks:
    - Codec health (if on RDK)
    - Frame source status
    - Processing status

    Returns:
        JSON with overall health status.
    """
    health: Dict[str, Any] = {
        "status": "healthy",
        "components": {}
    }

    # Check codec health
    if _codec_health_monitor is not None:
        try:
            codec_healthy = _codec_health_monitor.is_healthy()
            codec_stats = _codec_health_monitor.get_stats()
            health["components"]["codec"] = {
                "healthy": codec_healthy,
                "state": codec_stats.get("state", "unknown"),
                "restarts_total": codec_stats.get("restarts_total", 0)
            }
            if not codec_healthy:
                health["status"] = "degraded"
        except Exception as e:
            health["components"]["codec"] = {
                "healthy": False,
                "error": str(e)
            }
            health["status"] = "error"
    else:
        health["components"]["codec"] = {
            "healthy": True,
            "note": "Not applicable (not on RDK)"
        }

    # Overall status code
    status_code = 200 if health["status"] == "healthy" else 503

    return JSONResponse(content=health, status_code=status_code)



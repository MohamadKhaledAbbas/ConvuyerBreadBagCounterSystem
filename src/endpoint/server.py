"""
FastAPI Server for Conveyor Bread Bag Counter System.

Production-quality implementation with:
- Clean modular architecture
- Proper error handling and logging
- Resource management (lifespan)
- Static file serving
- Health monitoring

Based on V1 logic with enhanced code quality and maintainability.
"""

import os
import platform
import socket
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from fastapi import Request
from fastapi.responses import HTMLResponse

from src.endpoint.routes import analytics
from src.endpoint.routes import track_lifecycle
from src.endpoint.routes import snapshot
from src.endpoint.routes import counts
from src.endpoint.shared import init_shared_resources, cleanup_shared_resources, get_templates, get_db
from src.endpoint.pipeline_state import read_state
from src.utils.AppLogging import logger
from src.utils.system_info import get_system_info

from src.config.settings import AppConfig
from src.config.paths import (
    CODEC_HEALTH_STATUS_FILE,
    SPOOL_PROCESSOR_STATUS_FILE,
    SPOOL_RECORDER_STATUS_FILE,
)

# Application version
APP_VERSION = AppConfig.APP_VERSION

# Server start time for uptime tracking
_SERVER_START_TIME = time.time()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Application lifespan handler for startup/shutdown.

    Manages:
    - Database connections
    - Template engine initialization
    - Resource cleanup on shutdown
    """
    # Startup
    logger.info("[Endpoint] Starting up...")
    init_shared_resources()
    # Restore persisted conveyor ROI settings
    from src.endpoint.routes.conveyor_roi import load_persisted_roi_settings
    load_persisted_roi_settings()
    yield
    # Shutdown
    logger.info("[Endpoint] Shutting down...")
    cleanup_shared_resources()


# Create FastAPI application
app = FastAPI(
    title="Conveyor Bread Bag Counter API",
    description="Analytics endpoint for bread bag counting system",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include analytics router
app.include_router(analytics.router)

# Include track lifecycle router
app.include_router(track_lifecycle.router)

# Include snapshot router for live camera feed
app.include_router(snapshot.router)

# Include counts router for real-time pipeline counts
app.include_router(counts.router)

# Include conveyor ROI settings router
from src.endpoint.routes import conveyor_roi
app.include_router(conveyor_roi.router)

# Include guidelines router
from src.endpoint.routes import guidelines
app.include_router(guidelines.router)

# Include health status router (codec health, pipeline health)
from src.endpoint.routes import health
app.include_router(health.router)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Root endpoint - Dashboard with shortcuts to all system features (Arabic version).

    Returns:
        HTMLResponse: Main dashboard page with navigation cards in Arabic
    """
    templates = get_templates()
    return templates.TemplateResponse('index_ar.html', {
        'request': request,
        'version': APP_VERSION
    })


@app.get("/endpoints", response_class=HTMLResponse)
async def endpoints_page(request: Request):
    """
    API endpoints directory page.

    Lists all available REST API endpoints and pages with descriptions,
    organized by category. Useful for developers integrating with the system.
    """
    templates = get_templates()
    return templates.TemplateResponse('endpoints_ar.html', {
        'request': request,
    })


@app.get("/health")
async def health() -> JSONResponse:
    """
    Enhanced health check endpoint with system diagnostics.

    Returns version, uptime, DB connectivity, live pipeline metrics,
    per-component pipeline health, end-to-end throughput (FPS at each
    stage, time behind real-time), power-save status, and a 24-hour
    monitoring log summary so the UI and external monitors get a
    complete picture of the entire system.

    Cache-Control: no-store is set explicitly so that browsers and any
    intermediate proxies never serve a stale snapshot.  The frame-throttle
    power-mode card on the health page must always reflect the current mode
    (FULL vs DEGRADED), which changes independently of any HTTP activity.
    """
    now = time.time()
    uptime_seconds = now - _SERVER_START_TIME

    # Format uptime as HH:MM:SS clock format
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Check DB connectivity
    db_ok = False
    db = None
    try:
        db = get_db()
        if db is not None:
            db_ok = True
    except Exception:
        pass

    # Read pipeline state for live metrics
    pipeline = read_state()
    pipeline_active = bool(pipeline.get("confirmed_total", 0) > 0 or pipeline.get("pending_total", 0) > 0
                           or pipeline.get("current_batch_type"))

    # ── Production line status ──
    _IDLE_RESET_SECS   = 2 * 60 * 60   # 2 hours  – mirrors counts.py
    _WORK_ACTIVE_SECS  = 5 * 60         # 5 minutes – mirrors counts.py
    _last_count_ts     = pipeline.get("last_count_timestamp", 0) or 0
    _line_idle_secs    = (now - _last_count_ts) if _last_count_ts > 0 else 0
    line_stopped       = _last_count_ts > 0 and _line_idle_secs >= _IDLE_RESET_SECS
    work_active        = _last_count_ts > 0 and _line_idle_secs < _WORK_ACTIVE_SECS
    if line_stopped:
        line_status = "stopped"
    elif work_active:
        line_status = "active"
    elif _last_count_ts > 0:
        line_status = "idle"
    else:
        line_status = "unknown"
    last_count_dt = (
        datetime.fromtimestamp(_last_count_ts).strftime("%Y/%m/%d - %H:%M")
        if _last_count_ts > 0 else None
    )
    line_idle_minutes  = round(_line_idle_secs / 60) if _last_count_ts > 0 else 0

    # Resolve arabic names for batch types from bag_types table
    current_batch = pipeline.get("current_batch_type")
    last_classified = pipeline.get("last_classified_type")
    current_batch_arabic = current_batch
    last_classified_arabic = last_classified
    try:
        if db_ok and db is not None and (current_batch or last_classified):
            bag_types = db.get_all_bag_types()
            name_map = {bt['name']: bt.get('arabic_name') or bt['name'] for bt in bag_types}
            if current_batch:
                current_batch_arabic = name_map.get(current_batch, current_batch)
            if last_classified:
                last_classified_arabic = name_map.get(last_classified, last_classified)
    except Exception:
        pass

    # ── Pipeline component health (from codec health monitor status file) ──
    overall_status = "healthy"
    components: Dict[str, Any] = {}
    degraded_reasons: list = []

    # Database component
    components["database"] = {
        "healthy": db_ok,
        "status": "connected" if db_ok else "disconnected",
    }
    if not db_ok:
        overall_status = "degraded"
        degraded_reasons.append("قاعدة البيانات غير متصلة")

    # Codec / spool / RTSP from shared status file
    codec_status_file = CODEC_HEALTH_STATUS_FILE
    codec_data = None
    try:
        if os.path.exists(codec_status_file):
            import json
            with open(codec_status_file, 'r') as _f:
                codec_data = json.load(_f)
    except Exception:
        pass

    if codec_data is not None:
        state = codec_data.get("state", "unknown")
        ts = codec_data.get("timestamp", 0)
        age = (time.time() - ts) if ts else None
        is_stale = (age is not None and age > 60)

        codec_healthy = (state == "healthy") and not is_stale
        components["codec"] = {
            "healthy": codec_healthy,
            "state": state,
            "stale": is_stale,
            "restarts_total": codec_data.get("restarts_total", 0),
        }
        if not codec_healthy:
            overall_status = "degraded"
            # Constrain state to known values for safe display
            safe_state = state if state in ("healthy", "degraded", "critical",
                                            "recovering", "escalating", "unknown") else "unknown"
            degraded_reasons.append(f"وحدة فك الترميز: {safe_state}")

        # Checkpoints
        checkpoints = codec_data.get("health_checkpoints", {})
        spool_cp = checkpoints.get("spool_input", {})
        if spool_cp:
            spool_alive = spool_cp.get("alive", False)
            components["spool"] = {
                "healthy": spool_alive,
                "reason": spool_cp.get("reason", ""),
            }
            if not spool_alive:
                overall_status = "degraded"
                degraded_reasons.append("مدخل التسجيل متوقف")

        rtsp_cp = checkpoints.get("rtsp_ingest", {})
        if rtsp_cp:
            rtsp_alive = rtsp_cp.get("alive", False)
            components["rtsp"] = {
                "healthy": rtsp_alive,
                "reason": rtsp_cp.get("reason", ""),
            }
            if not rtsp_alive:
                overall_status = "degraded"
                degraded_reasons.append("بث الكاميرا متوقف")

        components["recovery_stage"] = codec_data.get("current_recovery_stage", 1)
        components["escalation_count"] = codec_data.get("escalation_count", 0)
    else:
        from src.utils.platform import is_rdk_platform
        if is_rdk_platform():
            components["codec"] = {
                "healthy": False, "state": "unknown",
                "reason": "ملف الحالة غير موجود — main.py قد لا يعمل"
            }
            overall_status = "degraded"
            degraded_reasons.append("مراقب فك الترميز غير متصل")

    # ── Spool recorder stats (RTSP → disk, cross-process) ──
    spool_recorder: Dict[str, Any] = {"available": False}
    try:
        _rec_path = SPOOL_RECORDER_STATUS_FILE
        if os.path.exists(_rec_path):
            import json
            with open(_rec_path, "r") as _f:
                _rec = json.load(_f)
            _rec_ts = _rec.get("timestamp", 0)
            _rec_age = (now - _rec_ts) if _rec_ts else 0
            spool_recorder = {
                "available": True,
                "healthy": _rec_age < 30 and _rec.get("avg_fps", 0) > 1.0,
                "avg_fps": _rec.get("avg_fps", 0),
                "frames_received": _rec.get("frames_received", 0),
                "segments_completed": _rec.get("segments_completed", 0),
                "write_queue_size": _rec.get("write_queue_size", 0),
                "write_queue_hwm": _rec.get("write_queue_hwm", 0),
                "write_queue_capacity": _rec.get("write_queue_capacity", 0),
                "age_seconds": round(_rec_age, 1),
            }
    except Exception:
        pass

    # ── Spool processor stats (disk → codec, cross-process) ──
    spool_processor: Dict[str, Any] = {"available": False}
    try:
        _proc_path = SPOOL_PROCESSOR_STATUS_FILE
        if os.path.exists(_proc_path):
            import json
            with open(_proc_path, "r") as _f:
                _proc = json.load(_f)
            _proc_ts = _proc.get("timestamp", 0)
            _proc_age = (now - _proc_ts) if _proc_ts else 0
            _proc_sentinel = _proc.get("sentinel_active", False)
            _proc_behind = _proc.get("time_behind_recorder_s", 0)
            spool_processor = {
                "available": True,
                "healthy": _proc_age < 30 and (_proc_sentinel or _proc_behind < 30.0),
                "current_fps": _proc.get("current_fps", 0),
                "avg_fps": _proc.get("avg_fps", 0),
                "time_behind_recorder_s": _proc_behind,
                "segments_behind": _proc.get("segments_behind", 0),
                "segments_on_disk": _proc.get("segments_on_disk", 0),
                "sentinel_active": _proc_sentinel,
                "sentinel_frames_sent": _proc.get("sentinel_frames_sent", 0),
                "frames_published": _proc.get("frames_published", 0),
                "segments_processed": _proc.get("segments_processed", 0),
                "age_seconds": round(_proc_age, 1),
            }
    except Exception:
        pass

    # ── App-level processing metrics ──
    app_metrics = pipeline.get("app_metrics", {})

    # ── End-to-end throughput summary ──
    throughput = {
        "recorder_fps": spool_recorder.get("avg_fps", 0) if spool_recorder.get("available") else None,
        "processor_fps": spool_processor.get("current_fps", 0) if spool_processor.get("available") else None,
        "app_fps": app_metrics.get("fps", 0),
        "time_behind_recorder_s": spool_processor.get("time_behind_recorder_s", 0) if spool_processor.get("available") else None,
        "segments_behind": spool_processor.get("segments_behind", 0) if spool_processor.get("available") else None,
    }

    # ── Monitoring log summary (24h) ──
    log_summary: Dict[str, Any] = {}
    try:
        if db_ok and db is not None:
            log_summary = db.get_monitoring_log_summary()
    except Exception:
        pass

    # ── System info (temperatures, CPU load, DB size, disk space) ──
    system_info: Dict[str, Any] = {}
    try:
        db_path = db.db_path if db_ok and db is not None else None
        system_info = get_system_info(db_path=db_path)
    except Exception:
        pass

    payload = {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION,
        "uptime_seconds": round(uptime_seconds, 1),
        "uptime": uptime_str,
        "database": "connected" if db_ok else "disconnected",
        "pipeline_active": pipeline_active,
        "pipeline": {
            "confirmed_total": pipeline.get("confirmed_total", 0),
            "pending_total": pipeline.get("pending_total", 0),
            "current_batch_type": current_batch,
            "current_batch_type_arabic": current_batch_arabic,
            "last_classified_type": last_classified,
            "last_classified_type_arabic": last_classified_arabic,
            "smoothing_rate": pipeline.get("smoothing_rate", 0),
            # Line activity fields
            "line_status": line_status,           # "active" | "idle" | "stopped" | "unknown"
            "line_stopped": line_stopped,         # True when idle > 2 hours (mirrors counts idle-reset)
            "work_active": work_active,           # True when last bag < 5 minutes ago
            "last_count_datetime": last_count_dt, # "yyyy/mm/dd - HH:MM" of the last counted bag
            "idle_minutes": line_idle_minutes,    # minutes since last bag was counted
            # Adaptive frame throttle (power-saving idle mode)
            "frame_throttle": pipeline.get("frame_throttle", {}),
        },
        # ── NEW: per-stage throughput and media pipeline stats ──
        "throughput": throughput,
        "app_metrics": app_metrics,
        "spool_recorder": spool_recorder,
        "spool_processor": spool_processor,
        "components": components,
        "degraded_reasons": degraded_reasons,
        "monitoring_log_summary": log_summary,
        "system_info": system_info,
    }

    # Return with explicit Cache-Control header so browsers and any reverse-
    # proxies never serve a stale snapshot.  The frame-throttle card must
    # always show the live mode (FULL vs DEGRADED).
    return JSONResponse(
        content=payload,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/whoami")
async def whoami() -> Dict[str, Any]:
    """
    Board identity / reachability probe.

    Lightweight endpoint that always returns HTTP 200 so remote clients
    can verify network connectivity to this board before issuing heavier
    API calls.  The response body carries stable identity fields useful
    for fleet-management dashboards and debugging.

    Used by cloud ``verifyBoard(url)`` to confirm the board is reachable.
    """
    hostname = socket.gethostname()

    return {
        "status": "ok",
        "hostname": hostname,
        "platform": platform.platform(),
        "version": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health/view", response_class=HTMLResponse)
async def health_page(request: Request):
    """
    Health status page with styled UI matching the system theme.
    Auto-refreshes every 30 seconds via JavaScript.
    """
    templates = get_templates()
    return templates.TemplateResponse('health.html', {
        'request': request,
        'version': APP_VERSION
    })


def setup_static_mounts() -> None:
    """
    Setup static file serving directories.

    Configures mounts for:
    - Unknown classes (unrecognized bag images for review)
    - Known classes (reference images for each bag type)
    - Static assets (CSS, JS, fonts)

    Creates directories if they don't exist.
    """
    # Unknown classes directory (images for manual classification)
    from src.config.paths import UNKNOWN_CLASSES_DIR, KNOWN_CLASSES_DIR
    unknown_dir = UNKNOWN_CLASSES_DIR
    os.makedirs(unknown_dir, exist_ok=True)
    app.mount("/unknown_classes", StaticFiles(directory=unknown_dir), name="unknown_classes")

    # Known classes directory (reference images)
    classes_dir = KNOWN_CLASSES_DIR
    os.makedirs(classes_dir, exist_ok=True)
    app.mount("/known_classes", StaticFiles(directory=classes_dir), name="known_classes")

    # Static assets (CSS, JS)
    static_dir = os.getenv("STATIC_DIR", "src/endpoint/static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"[Endpoint] Static mounts configured successfully")
    else:
        logger.warning(f"[Endpoint] Static directory not found: {static_dir}")


# Setup static mounts on module load
try:
    setup_static_mounts()
except Exception as e:
    logger.error(f"[Endpoint] Failed to setup static mounts: {e}")
    # Don't fail startup - app can still serve API responses

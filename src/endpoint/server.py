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
async def health() -> Dict[str, Any]:
    """
    Enhanced health check endpoint with system diagnostics.

    Returns version, uptime, DB connectivity, live pipeline metrics,
    per-component pipeline health, and a 24-hour monitoring log summary
    so the UI and external monitors get a complete picture.
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
    codec_status_file = "/tmp/codec_health_status.json"
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

    return {
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
        },
        "components": components,
        "degraded_reasons": degraded_reasons,
        "monitoring_log_summary": log_summary,
        "system_info": system_info,
    }


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
    unknown_dir = os.getenv("UNKNOWN_CLASSES_DIR", "data/unknown")
    os.makedirs(unknown_dir, exist_ok=True)
    app.mount("/unknown_classes", StaticFiles(directory=unknown_dir), name="unknown_classes")

    # Known classes directory (reference images)
    classes_dir = os.getenv("CLASSES_DIR", "data/classes")
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

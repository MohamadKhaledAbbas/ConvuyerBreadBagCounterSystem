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


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Enhanced health check endpoint with system diagnostics.

    Returns version, uptime, DB connectivity, and live pipeline metrics
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

    return {
        "status": "healthy",
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
        }
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

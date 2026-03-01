"""
Conveyor ROI Settings Routes.

Provides API endpoints for configuring the conveyor detection zone:
- GET  /api/conveyor-roi         → Current ROI settings (JSON)
- POST /api/conveyor-roi         → Update ROI settings (JSON)
- GET  /settings/conveyor-roi    → Visual configuration page (HTML)

The conveyor ROI restricts detection to the belt area only, eliminating
false-positive detections on table edges, operator hands, etc.
"""

import json
import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.config.tracking_config import TrackingConfig, tracking_config
from src.endpoint.shared import get_templates
from src.utils.AppLogging import logger

router = APIRouter()


def _get_config() -> TrackingConfig:
    """Return the global tracking config singleton."""
    return tracking_config


@router.get("/api/conveyor-roi", response_class=JSONResponse)
async def get_conveyor_roi():
    """Return current conveyor ROI settings."""
    cfg = _get_config()
    return {
        "enabled": cfg.conveyor_roi_enabled,
        "x_min": cfg.conveyor_roi_x_min,
        "x_max": cfg.conveyor_roi_x_max,
        "y_min": cfg.conveyor_roi_y_min,
        "y_max": cfg.conveyor_roi_y_max,
        "show_overlay": cfg.conveyor_roi_show_overlay,
    }


@router.post("/api/conveyor-roi", response_class=JSONResponse)
async def update_conveyor_roi(request: Request):
    """
    Update conveyor ROI settings at runtime.

    Accepts JSON body with any subset of:
      enabled, x_min, x_max, y_min, y_max, show_overlay

    Changes take effect immediately (next frame) — no restart required.
    Values are also persisted to a JSON file so they survive restarts.
    """
    cfg = _get_config()
    body = await request.json()

    if "enabled" in body:
        cfg.conveyor_roi_enabled = bool(body["enabled"])
    if "x_min" in body:
        cfg.conveyor_roi_x_min = int(body["x_min"])
    if "x_max" in body:
        cfg.conveyor_roi_x_max = int(body["x_max"])
    if "y_min" in body:
        cfg.conveyor_roi_y_min = int(body["y_min"])
    if "y_max" in body:
        cfg.conveyor_roi_y_max = int(body["y_max"])
    if "show_overlay" in body:
        cfg.conveyor_roi_show_overlay = bool(body["show_overlay"])

    # Persist to file for restart survival
    _persist_roi_settings(cfg)

    logger.info(
        f"[ConveyorROI] Updated: enabled={cfg.conveyor_roi_enabled} "
        f"zone=({cfg.conveyor_roi_x_min},{cfg.conveyor_roi_y_min})-"
        f"({cfg.conveyor_roi_x_max},{cfg.conveyor_roi_y_max})"
    )

    return {
        "status": "ok",
        "enabled": cfg.conveyor_roi_enabled,
        "x_min": cfg.conveyor_roi_x_min,
        "x_max": cfg.conveyor_roi_x_max,
        "y_min": cfg.conveyor_roi_y_min,
        "y_max": cfg.conveyor_roi_y_max,
        "show_overlay": cfg.conveyor_roi_show_overlay,
    }


_ROI_PERSIST_PATH = "data/conveyor_roi.json"


def _persist_roi_settings(cfg: TrackingConfig):
    """Save ROI settings to a JSON file for restart survival."""
    try:
        os.makedirs(os.path.dirname(_ROI_PERSIST_PATH), exist_ok=True)
        data = {
            "enabled": cfg.conveyor_roi_enabled,
            "x_min": cfg.conveyor_roi_x_min,
            "x_max": cfg.conveyor_roi_x_max,
            "y_min": cfg.conveyor_roi_y_min,
            "y_max": cfg.conveyor_roi_y_max,
            "show_overlay": cfg.conveyor_roi_show_overlay,
        }
        with open(_ROI_PERSIST_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"[ConveyorROI] Failed to persist settings: {e}")


def load_persisted_roi_settings():
    """
    Load persisted ROI settings on startup.

    Call this once during app initialization (after TrackingConfig is created).
    File values override env-var defaults so the user's last interactive
    configuration is restored.
    """
    cfg = _get_config()
    if not os.path.exists(_ROI_PERSIST_PATH):
        return

    try:
        with open(_ROI_PERSIST_PATH, "r") as f:
            data = json.load(f)
        if "enabled" in data:
            cfg.conveyor_roi_enabled = bool(data["enabled"])
        if "x_min" in data:
            cfg.conveyor_roi_x_min = int(data["x_min"])
        if "x_max" in data:
            cfg.conveyor_roi_x_max = int(data["x_max"])
        if "y_min" in data:
            cfg.conveyor_roi_y_min = int(data["y_min"])
        if "y_max" in data:
            cfg.conveyor_roi_y_max = int(data["y_max"])
        if "show_overlay" in data:
            cfg.conveyor_roi_show_overlay = bool(data["show_overlay"])
        logger.info(
            f"[ConveyorROI] Loaded persisted settings: enabled={cfg.conveyor_roi_enabled} "
            f"zone=({cfg.conveyor_roi_x_min},{cfg.conveyor_roi_y_min})-"
            f"({cfg.conveyor_roi_x_max},{cfg.conveyor_roi_y_max})"
        )
    except Exception as e:
        logger.warning(f"[ConveyorROI] Failed to load persisted settings: {e}")


@router.get("/settings/conveyor-roi", response_class=HTMLResponse)
async def conveyor_roi_page(request: Request):
    """
    Visual configuration page for the conveyor ROI zone.

    Shows a live camera snapshot with a draggable rectangle overlay
    so the user can visually set the conveyor boundaries.
    """
    templates = get_templates()
    cfg = _get_config()

    context = {
        "request": request,
        "roi": {
            "enabled": cfg.conveyor_roi_enabled,
            "x_min": cfg.conveyor_roi_x_min,
            "x_max": cfg.conveyor_roi_x_max,
            "y_min": cfg.conveyor_roi_y_min,
            "y_max": cfg.conveyor_roi_y_max,
            "show_overlay": cfg.conveyor_roi_show_overlay,
        },
    }
    return templates.TemplateResponse("conveyor_roi_settings.html", context)



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
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.endpoint.routes import analytics
from src.endpoint.shared import init_shared_resources, cleanup_shared_resources
from src.utils.AppLogging import logger


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


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        dict: Health status with UTC timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


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

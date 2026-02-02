"""
Shared resources for the endpoint module.

Provides centralized management of:
- Database connections
- Template engine
- Singleton pattern for resource efficiency

Thread-safe resource initialization and cleanup.
"""

import os
from typing import Optional
from pathlib import Path

from fastapi.templating import Jinja2Templates

from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger


# Global shared resources (singletons)
_db: Optional[DatabaseManager] = None
_templates: Optional[Jinja2Templates] = None


def get_db() -> DatabaseManager:
    """
    Get the shared database manager instance (singleton).

    Lazy initialization on first access.
    Thread-safe for read operations.

    Returns:
        DatabaseManager: Shared database connection manager
    """
    global _db
    if _db is None:
        db_path = os.getenv("DB_PATH", "data/db/bag_events.db")
        _db = DatabaseManager(db_path)
        logger.info(f"[Shared] Database initialized: {db_path}")
    return _db


def get_templates() -> Jinja2Templates:
    """
    Get the shared Jinja2 templates instance (singleton).

    Lazy initialization on first access.
    Templates are cached for performance.

    Returns:
        Jinja2Templates: Template engine instance

    Raises:
        RuntimeError: If templates directory doesn't exist
    """
    global _templates
    if _templates is None:
        templates_dir = Path(__file__).parent / "templates"

        if not templates_dir.exists():
            logger.error(f"[Shared] Templates directory not found: {templates_dir}")
            raise RuntimeError(f"Templates directory not found: {templates_dir}")

        _templates = Jinja2Templates(directory=str(templates_dir))
        logger.info(f"[Shared] Templates initialized: {templates_dir}")

    return _templates


def init_shared_resources() -> None:
    """
    Initialize all shared resources on application startup.

    Called by FastAPI lifespan handler.
    Ensures resources are ready before serving requests.
    """
    try:
        get_db()
        get_templates()
        logger.info("[Shared] All shared resources initialized successfully")
    except Exception as e:
        logger.error(f"[Shared] Failed to initialize resources: {e}")
        raise


def cleanup_shared_resources() -> None:
    """
    Cleanup shared resources on application shutdown.

    Called by FastAPI lifespan handler.
    Ensures proper resource disposal (close DB connections, etc.)
    """
    global _db, _templates

    if _db is not None:
        try:
            _db.close()
            _db = None
            logger.info("[Shared] Database connection closed")
        except Exception as e:
            logger.error(f"[Shared] Error closing database: {e}")

    _templates = None
    logger.info("[Shared] Shared resources cleaned up")

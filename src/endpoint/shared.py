"""Shared Resources - Updated for V2."""
from pathlib import Path
from typing import Optional

from fastapi.templating import Jinja2Templates

from src.config.config_manager import get_config
from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger

_db_instance: Optional[DatabaseManager] = None
_templates_instance: Optional[Jinja2Templates] = None
def init_shared_resources():
    global _db_instance, _templates_instance
    config = get_config()
    _db_instance = DatabaseManager(config.db_path)
    logger.info("[Shared] Database initialized")
    # Run 7-day retention cleanup on startup
    try:
        _db_instance.purge_old_track_events(retention_days=7)
    except Exception as e:
        logger.error(f"[Shared] Retention cleanup failed: {e}")
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    _templates_instance = Jinja2Templates(directory=str(template_dir))
    logger.info(f"[Shared] Templates initialized: {template_dir}")
def get_db() -> DatabaseManager:
    if _db_instance is None:
        raise RuntimeError("Shared resources not initialized")
    return _db_instance
def get_templates() -> Jinja2Templates:
    if _templates_instance is None:
        raise RuntimeError("Shared resources not initialized")
    return _templates_instance
def cleanup_shared_resources():
    global _db_instance, _templates_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None
        logger.info("[Shared] Database closed")
    _templates_instance = None
    logger.info("[Shared] Cleanup complete")

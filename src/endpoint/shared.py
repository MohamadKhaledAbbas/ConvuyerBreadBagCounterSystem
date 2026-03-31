"""Shared Resources - Updated for V2."""
import inspect
from pathlib import Path
from typing import Optional, Any, Dict

from fastapi.templating import Jinja2Templates

from src.config.config_manager import get_config
from src.logging.Database import DatabaseManager
from src.logging.db_log_handler import attach_db_log_handler
from src.utils.AppLogging import logger

_db_instance: Optional[DatabaseManager] = None
_templates_instance: Optional[Jinja2Templates] = None
def init_shared_resources():
    global _db_instance, _templates_instance
    config = get_config()
    _db_instance = DatabaseManager(config.db_path)
    logger.info("[Shared] Database initialized")
    # Run retention cleanup on startup for all event tables
    try:
        _db_instance.purge_old_events(retention_days=7)
        _db_instance.purge_old_track_events(retention_days=3)
        _db_instance.purge_old_monitoring_logs(retention_days=7)
    except Exception as e:
        logger.error(f"[Shared] Retention cleanup failed: {e}")
    # Attach DB-backed log handler so WARNING+ logs are queryable via API
    try:
        attach_db_log_handler(_db_instance)
    except Exception as e:
        logger.error(f"[Shared] Failed to attach DB log handler: {e}")
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


def render_template(templates: Jinja2Templates, request, name: str, context: Optional[Dict[str, Any]] = None):
    """Render a template across both older and newer Starlette signatures."""
    template_context = dict(context or {})
    template_context.setdefault("request", request)

    parameters = list(inspect.signature(type(templates).TemplateResponse).parameters.values())
    uses_request_first = len(parameters) > 1 and parameters[1].name == "request"

    if uses_request_first:
        return templates.TemplateResponse(request, name, template_context)

    return templates.TemplateResponse(name, template_context)

def cleanup_shared_resources():
    global _db_instance, _templates_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None
        logger.info("[Shared] Database closed")
    _templates_instance = None
    logger.info("[Shared] Cleanup complete")

"""
Guidelines Route - Arabic operational guidelines page.

Provides:
- GET /guidelines  - HTML page showing system usage guidelines in Arabic
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.endpoint.shared import get_templates, render_template
from src.utils.AppLogging import logger

router = APIRouter(tags=["guidelines"])


@router.get("/guidelines", response_class=HTMLResponse)
async def guidelines_page(request: Request):
    """
    Operational guidelines page in Arabic.

    Displays the rules and best-practices that workers and supervisors
    must follow for the counting system to operate accurately.
    """
    logger.debug("[Guidelines] Page requested")
    templates = get_templates()
    return render_template(templates, request, "guidelines.html")


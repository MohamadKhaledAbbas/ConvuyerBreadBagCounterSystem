"""Analytics Routes - Updated with Repository Pattern."""
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse

from src.endpoint.repositories.analytics_repository import AnalyticsRepository
from src.endpoint.services.analytics_service import AnalyticsService
from src.endpoint.shared import get_db, get_templates
from src.utils.AppLogging import logger

router = APIRouter()
def _get_service() -> AnalyticsService:
    db = get_db()
    repo = AnalyticsRepository(db)
    return AnalyticsService(repo)
@router.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request, start_time: Optional[str] = Query(None), end_time: Optional[str] = Query(None)):
    templates = get_templates()
    if start_time is None or end_time is None:
        return templates.TemplateResponse('analytics_form.html', {'request': request})
    logger.info(f'[Analytics] Request: start={start_time}, end={end_time}')
    service = _get_service()
    try:
        start_dt = service.parse_datetime(start_time)
        end_dt = service.parse_datetime(end_time)
        if start_dt >= end_dt:
            raise HTTPException(422, 'Start time must be before end time')
        data = service.get_analytics_data(start_dt, end_dt)
        context = {'request': request, 'meta': data['meta'], 'total': data['data']['total'], 'classifications': data['data']['classifications'], 'timeline': data['timeline']}
        logger.info(f'[Analytics] Rendering: {data["data"]["total"]["count"]} bags')
        return templates.TemplateResponse('analytics.html', context)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[Analytics] Error: {e}', exc_info=True)
        raise HTTPException(500, str(e))
@router.get('/analytics/daily', response_class=HTMLResponse)
async def analytics_daily(request: Request):
    logger.info('[Analytics/Daily] Calculating shift times')
    service = _get_service()
    start_dt, end_dt = service.calculate_daily_shift_times()
    start_time = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
    end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%S')
    logger.info(f'[Analytics/Daily] Times: {start_time} to {end_time}')
    return await analytics(request=request, start_time=start_time, end_time=end_time)

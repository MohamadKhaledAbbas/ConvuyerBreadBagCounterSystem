"""Analytics Routes - Updated with Repository Pattern and Async Support."""
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Query, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from src.endpoint.repositories.analytics_repository import AnalyticsRepository
from src.endpoint.services.analytics_service import AnalyticsService
from src.endpoint.shared import get_db, get_templates
from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger

router = APIRouter()

def get_analytics_service(db: DatabaseManager = Depends(get_db)) -> AnalyticsService:
    """Dependency injection for analytics service."""
    repo = AnalyticsRepository(db)
    return AnalyticsService(repo)
@router.get("/analytics", response_class=HTMLResponse)
async def analytics(
    request: Request,
    start_time: Optional[str] = Query(None, description="Start time in ISO format (YYYY-MM-DDTHH:MM:SS)"),
    end_time: Optional[str] = Query(None, description="End time in ISO format (YYYY-MM-DDTHH:MM:SS)"),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Analytics dashboard endpoint with time range selection.
    
    Displays bag counting statistics, classifications breakdown, and timeline.
    If no time range provided, shows form for time selection.
    """
    templates = get_templates()
    
    # Show form if no time range specified
    if start_time is None or end_time is None:
        return templates.TemplateResponse('analytics_form_new.html', {'request': request})

    logger.info(f'[Analytics] Request: start={start_time}, end={end_time}')
    
    try:
        # Parse and validate datetime inputs
        start_dt = service.parse_datetime(start_time)
        end_dt = service.parse_datetime(end_time)
        
        if start_dt >= end_dt:
            raise HTTPException(
                status_code=422,
                detail='Start time must be before end time'
            )
        
        # Run blocking database operations in threadpool for true async
        data = await run_in_threadpool(service.get_analytics_data, start_dt, end_dt)
        
        # Prepare template context
        context = {
            'request': request,
            'meta': data['meta'],
            'total': data['data']['total'],
            'classifications': data['data']['classifications'],
            'timeline': data['timeline']
        }
        
        logger.info(f'[Analytics] Rendering: {data["data"]["total"]["count"]} bags')
        return templates.TemplateResponse('analytics_new.html', context)

    except HTTPException:
        raise
    except ValueError as e:
        # Handle date parsing errors specifically
        logger.warning(f'[Analytics] Invalid date format: {e}')
        raise HTTPException(
            status_code=400,
            detail=f'Invalid date format: {str(e)}'
        )
    except Exception as e:
        # Generic error handler - don't leak internal details
        logger.error(f'[Analytics] Unexpected error: {e}', exc_info=True)
        raise HTTPException(
            status_code=500,
            detail='An error occurred while processing analytics data'
        )
@router.get('/analytics/daily', response_class=HTMLResponse)
async def analytics_daily(
    request: Request,
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Daily analytics endpoint for current shift.
    
    Automatically calculates the current shift's start and end times based on
    configured shift hours and timezone, then displays analytics for that period.
    """
    logger.info('[Analytics/Daily] Calculating shift times')
    
    try:
        # Calculate shift times in threadpool (uses datetime operations)
        start_dt, end_dt = await run_in_threadpool(service.calculate_daily_shift_times)
        
        # Format times for analytics endpoint
        start_time = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
        end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        logger.info(f'[Analytics/Daily] Times: {start_time} to {end_time}')
        
        # Delegate to main analytics endpoint
        return await analytics(
            request=request,
            start_time=start_time,
            end_time=end_time,
            service=service
        )
        
    except Exception as e:
        logger.error(f'[Analytics/Daily] Error: {e}', exc_info=True)
        raise HTTPException(
            status_code=500,
            detail='An error occurred while calculating daily analytics'
        )

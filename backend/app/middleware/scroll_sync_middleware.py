from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from app.utils.scroll_time import get_scroll_time
import logging

logger = logging.getLogger(__name__)

class ScrollSyncMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects scroll timing context into every request.
    This ensures all operations are aligned with the divine scroll timing.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Get current scroll time
        scroll_time = get_scroll_time()
        
        # Inject scroll context into request state
        request.state.scroll_time = scroll_time
        
        # Log the request with scroll context
        logger.info(
            f"Request to {request.url.path} processed during {scroll_time['scroll_day']} "
            f"under Gate {scroll_time['gate']}: {scroll_time['gate_name']} "
            f"at Solar Hour {scroll_time['solar_hour']} with ENano Pulse {scroll_time['enano_pulse']}"
        )
        
        # Process the request
        response = await call_next(request)
        
        # Add scroll context to response headers (optional)
        response.headers["X-Scroll-Day"] = scroll_time["scroll_day"]
        response.headers["X-Scroll-Gate"] = str(scroll_time["gate"])
        response.headers["X-Scroll-Gate-Name"] = scroll_time["gate_name"]
        
        return response 
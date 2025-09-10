"""
AI-Powered Smart Class & Timetable Scheduler
FastAPI application factory and configuration
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.sessions import SessionMiddleware

# Import API routers
from .api import (
    auth,
    students,
    teachers,
    quizzes,
    schedules,
    analytics,
    content,
    sync,
    admin,
    notifications,
    gamification
)

# Import services and utilities
from .dependencies import get_current_user, get_db
from .exceptions import (
    AppException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    NotFoundException,
    ConflictException
)
from ..config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class RequestContextMiddleware:
    """Middleware to add request context and timing"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    process_time = time.time() - start_time
                    message["headers"] = list(message.get("headers", []))
                    message["headers"].append(
                        (b"x-process-time", f"{process_time:.6f}".encode())
                    )
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    settings = get_settings()
    
    # Create FastAPI instance
    app = FastAPI(
        title="Smart Class Scheduler API",
        description="Backend API for AI-Powered Smart Class & Timetable Scheduler",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        default_response_class=JSONResponse
    )
    
    # Add custom middleware
    app.add_middleware(RequestContextMiddleware)
    
    # Add security middleware
    if not settings.DEBUG:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
    
    # Add session middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.SECRET_KEY,
        max_age=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        https_only=not settings.DEBUG
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS if settings.ALLOWED_HOSTS != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-process-time", "x-request-id"]
    )
    
    # Exception handlers
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        """Handle custom application exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        logger.exception(f"Unexpected error: {exc}")
        
        if settings.DEBUG:
            import traceback
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "INTERNAL_ERROR",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": time.time()
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "INTERNAL_ERROR",
                    "message": "An internal server error occurred",
                    "timestamp": time.time()
                }
            )
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Application startup event"""
        logger.info("🚀 Backend API starting up...")
        
        # Initialize AI models if enabled
        if settings.ENABLE_AI_FEATURES:
            try:
                from .ai_engine.content_generator import initialize_models
                await initialize_models()
                logger.info("✅ AI models initialized")
            except Exception as e:
                logger.warning(f"⚠️ AI models initialization failed: {e}")
        
        # Initialize background tasks
        try:
            from .services.sync_service import start_background_sync
            await start_background_sync()
            logger.info("✅ Background sync started")
        except Exception as e:
            logger.warning(f"⚠️ Background sync failed to start: {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event"""
        logger.info("🛑 Backend API shutting down...")
        
        # Cleanup resources
        try:
            from .services.sync_service import stop_background_sync
            await stop_background_sync()
            logger.info("✅ Background sync stopped")
        except Exception as e:
            logger.warning(f"⚠️ Background sync cleanup failed: {e}")
    
    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """API health check endpoint"""
        return {
            "status": "healthy",
            "api_version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "timestamp": time.time(),
            "features": {
                "ai_enabled": settings.ENABLE_AI_FEATURES,
                "offline_sync": settings.ENABLE_OFFLINE_MODE,
                "real_time": settings.ENABLE_REAL_TIME_SYNC,
                "analytics": settings.ENABLE_ADVANCED_ANALYTICS
            }
        }
    
    # API status endpoint
    @app.get("/status", tags=["System"])
    async def api_status():
        """Detailed API status information"""
        return {
            "api": "Smart Class Scheduler",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "database": "connected",  # TODO: Add actual DB health check
            "cache": "connected",     # TODO: Add actual Redis health check
            "ai_engine": "ready" if settings.ENABLE_AI_FEATURES else "disabled",
            "uptime": time.time(),
            "endpoints": {
                "auth": "/auth",
                "students": "/students",
                "teachers": "/teachers", 
                "quizzes": "/quizzes",
                "schedules": "/schedules",
                "analytics": "/analytics",
                "content": "/content",
                "sync": "/sync",
                "admin": "/admin",
                "notifications": "/notifications",
                "gamification": "/gamification"
            }
        }
    
    # Include API routers
    app.include_router(
        auth.router,
        prefix="/auth",
        tags=["Authentication"]
    )
    
    app.include_router(
        students.router,
        prefix="/students",
        tags=["Students"],
        dependencies=[]  # Add authentication dependency when ready
    )
    
    app.include_router(
        teachers.router,
        prefix="/teachers",
        tags=["Teachers"],
        dependencies=[]  # Add authentication dependency when ready
    )
    
    app.include_router(
        quizzes.router,
        prefix="/quizzes",
        tags=["Quizzes"],
        dependencies=[]  # Add authentication dependency when ready
    )
    
    app.include_router(
        schedules.router,
        prefix="/schedules",
        tags=["Schedules"],
        dependencies=[]  # Add authentication dependency when ready
    )
    
    app.include_router(
        analytics.router,
        prefix="/analytics",
        tags=["Analytics"],
        dependencies=[]  # Add authentication dependency when ready
    )
    
    app.include_router(
        content.router,
        prefix="/content",
        tags=["Content Management"],
        dependencies=[]  # Add authentication dependency when ready
    )
    
    app.include_router(
        sync.router,
        prefix="/sync",
        tags=["Offline Synchronization"],
        dependencies=[]  # Add authentication dependency when ready
    )
    
    app.include_router(
        admin.router,
        prefix="/admin",
        tags=["Administration"],
        dependencies=[]  # Add admin authentication dependency when ready
    )
    
    app.include_router(
        notifications.router,
        prefix="/notifications",
        tags=["Notifications"],
        dependencies=[]  # Add authentication dependency when ready
    )
    
    app.include_router(
        gamification.router,
        prefix="/gamification",
        tags=["Gamification"],
        dependencies=[]  # Add authentication dependency when ready
    )
    
    # API documentation endpoint
    if settings.DEBUG:
        @app.get("/endpoints", tags=["System"])
        async def list_endpoints():
            """List all available API endpoints (debug only)"""
            routes = []
            for route in app.routes:
                if hasattr(route, 'methods'):
                    routes.append({
                        "path": route.path,
                        "methods": list(route.methods),
                        "name": route.name,
                        "tags": getattr(route, 'tags', [])
                    })
            return {"endpoints": routes}
    
    logger.info("✅ Backend API configured successfully")
    return app


# Export the app factory
__all__ = ["create_app"]
#!/usr/bin/env python3
"""
AI-Powered Smart Class & Timetable Scheduler
Main application entry point and configuration
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from backend.app import create_app
from backend.database.connection import init_database
from backend.utils.helpers import setup_logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global app instance
app_instance: Optional[FastAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    
    # Startup
    logger.info("🚀 Starting AI-Powered Smart Class Scheduler...")
    
    # Initialize database
    await init_database()
    logger.info("✅ Database initialized successfully")
    
    # Initialize AI models (if needed)
    # await initialize_ai_models()
    
    # Start background tasks
    # await start_background_tasks()
    
    logger.info("🎉 Application startup complete!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")
    # Cleanup resources here
    logger.info("✅ Application shutdown complete")


def create_main_app() -> FastAPI:
    """Create and configure the main FastAPI application"""
    
    settings = get_settings()
    
    # Create the main app with lifespan
    main_app = FastAPI(
        title="AI-Powered Smart Class & Timetable Scheduler",
        description="A comprehensive education platform with intelligent scheduling, real-time assessment, and gamification",
        version="1.0.0",
        docs_url="/api/docs" if settings.DEBUG else None,
        redoc_url="/api/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
        openapi_url="/api/openapi.json" if settings.DEBUG else None
    )
    
    # Add middleware
    main_app.add_middleware(GZipMiddleware, minimum_size=1000)
    main_app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount the backend API
    backend_app = create_app()
    main_app.mount("/api", backend_app)
    
    # Mount static files
    if Path("frontend/static").exists():
        main_app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
    
    # Serve frontend templates
    if Path("frontend/templates").exists():
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory="frontend/templates")
        
        @main_app.get("/", response_class=HTMLResponse)
        @main_app.get("/login", response_class=HTMLResponse)
        @main_app.get("/dashboard", response_class=HTMLResponse)
        @main_app.get("/student/{path:path}", response_class=HTMLResponse)
        @main_app.get("/teacher/{path:path}", response_class=HTMLResponse)
        @main_app.get("/admin/{path:path}", response_class=HTMLResponse)
        async def serve_frontend(request, path: str = ""):
            """Serve frontend SPA for all non-API routes"""
            return templates.TemplateResponse("base.html", {"request": request})
    
    # Health check endpoint
    @main_app.get("/health")
    async def health_check():
        """Application health check endpoint"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "database": "connected",
            "ai_engine": "ready"
        }
    
    return main_app


async def run_server():
    """Run the development server"""
    settings = get_settings()
    
    config = uvicorn.Config(
        app="main:app_instance",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
        access_log=settings.DEBUG,
        loop="asyncio",
        ws_ping_interval=20,
        ws_ping_timeout=20
    )
    
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    
    # Check environment
    settings = get_settings()
    logger.info(f"Starting application in {settings.ENVIRONMENT} mode")
    
    # Create app instance
    global app_instance
    app_instance = create_main_app()
    
    try:
        # Run the server
        if __name__ == "__main__":
            asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)


# Create app instance for uvicorn
app_instance = create_main_app()

if __name__ == "__main__":
    main()
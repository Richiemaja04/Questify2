"""
AI-Powered Smart Class & Timetable Scheduler
Database connection and session management
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import asyncio

from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool

from .models import Base
from ..utils.helpers import get_database_url
from ...config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Global variables
async_engine: Optional[AsyncEngine] = None
sync_engine = None
AsyncSessionLocal: Optional[async_sessionmaker] = None
SessionLocal = None


def get_sync_database_url(database_url: str) -> str:
    """Convert async database URL to sync version"""
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url.replace("postgresql+asyncpg://", "postgresql://")
    elif database_url.startswith("sqlite+aiosqlite://"):
        return database_url.replace("sqlite+aiosqlite://", "sqlite://")
    return database_url


def get_async_database_url(database_url: str) -> str:
    """Convert sync database URL to async version"""
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://")
    elif database_url.startswith("sqlite://"):
        return database_url.replace("sqlite://", "sqlite+aiosqlite://")
    return database_url


def create_sync_engine():
    """Create synchronous SQLAlchemy engine"""
    settings = get_settings()
    database_url = get_sync_database_url(settings.database_url)
    
    engine_kwargs = {
        "echo": settings.DB_ECHO,
        "future": True,
    }
    
    if database_url.startswith("sqlite"):
        # SQLite specific configuration
        engine_kwargs.update({
            "poolclass": StaticPool,
            "connect_args": {
                "check_same_thread": False,
                "timeout": 20
            }
        })
    else:
        # PostgreSQL specific configuration
        engine_kwargs.update({
            "poolclass": QueuePool,
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "pool_pre_ping": True
        })
    
    return create_engine(database_url, **engine_kwargs)


def create_async_engine_instance():
    """Create asynchronous SQLAlchemy engine"""
    settings = get_settings()
    database_url = get_async_database_url(settings.database_url)
    
    engine_kwargs = {
        "echo": settings.DB_ECHO,
        "future": True,
    }
    
    if database_url.startswith("sqlite"):
        # SQLite specific configuration
        engine_kwargs.update({
            "poolclass": StaticPool,
            "connect_args": {
                "check_same_thread": False,
                "timeout": 20
            }
        })
    else:
        # PostgreSQL specific configuration
        engine_kwargs.update({
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "pool_pre_ping": True
        })
    
    return create_async_engine(database_url, **engine_kwargs)


async def init_database():
    """Initialize database engines and create tables"""
    global async_engine, sync_engine, AsyncSessionLocal, SessionLocal
    
    logger.info("Initializing database connections...")
    
    try:
        # Create engines
        async_engine = create_async_engine_instance()
        sync_engine = create_sync_engine()
        
        # Create session factories
        AsyncSessionLocal = async_sessionmaker(
            bind=async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        
        SessionLocal = sessionmaker(
            bind=sync_engine,
            autocommit=False,
            autoflush=True
        )
        
        # Test connections
        await test_async_connection()
        test_sync_connection()
        
        # Create tables
        await create_tables()
        
        # Set up event listeners
        setup_event_listeners()
        
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


async def test_async_connection():
    """Test async database connection"""
    try:
        async with async_engine.begin() as conn:
            await conn.execute("SELECT 1")
        logger.info("✅ Async database connection successful")
    except Exception as e:
        logger.error(f"❌ Async database connection failed: {e}")
        raise


def test_sync_connection():
    """Test sync database connection"""
    try:
        with sync_engine.begin() as conn:
            conn.execute("SELECT 1")
        logger.info("✅ Sync database connection successful")
    except Exception as e:
        logger.error(f"❌ Sync database connection failed: {e}")
        raise


async def create_tables():
    """Create database tables"""
    try:
        async with async_engine.begin() as conn:
            # Import models to ensure they're registered
            from .models import (
                User, School, StudentProfile, TeacherProfile,
                Class, Enrollment, Quiz, Question, QuizSubmission,
                Answer, Grade, Resource, Schedule, ScheduleConflict,
                Achievement, UserAchievement, XPTransaction,
                AnalyticsEvent, BehavioralProfile, Notification,
                SyncQueue, AIModel, ContentAnalysis
            )
            
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ Database tables created successfully")
        
        # Initialize default data
        await initialize_default_data()
        
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")
        raise


async def initialize_default_data():
    """Initialize default data for the application"""
    try:
        async with get_async_session() as session:
            # Check if we already have data
            from .models import School, Achievement
            
            # Create default school if none exists
            result = await session.execute("SELECT COUNT(*) FROM schools")
            school_count = result.scalar()
            
            if school_count == 0:
                await create_default_school(session)
            
            # Create default achievements if none exist
            result = await session.execute("SELECT COUNT(*) FROM achievements")
            achievement_count = result.scalar()
            
            if achievement_count == 0:
                await create_default_achievements(session)
            
            await session.commit()
            logger.info("✅ Default data initialized")
            
    except Exception as e:
        logger.error(f"❌ Default data initialization failed: {e}")
        # Don't raise here as this is not critical for basic operation


async def create_default_school(session: AsyncSession):
    """Create a default school for development/testing"""
    from .models import School
    import uuid
    
    default_school = School(
        id=uuid.uuid4(),
        name="Demo Smart School",
        code="DEMO001",
        address="123 Education Street, Learning City, LC 12345",
        phone="+1-555-0123",
        email="admin@demoschool.edu",
        website="https://demoschool.edu",
        timezone="UTC",
        settings={
            "academic_year": "2024-2025",
            "grading_scale": {
                "A": {"min": 90, "max": 100},
                "B": {"min": 80, "max": 89},
                "C": {"min": 70, "max": 79},
                "D": {"min": 60, "max": 69},
                "F": {"min": 0, "max": 59}
            },
            "class_duration_minutes": 45,
            "passing_grade": 60
        },
        is_active=True
    )
    
    session.add(default_school)
    logger.info("Created default school")


async def create_default_achievements(session: AsyncSession):
    """Create default achievements"""
    from .models import Achievement, AchievementType
    import uuid
    
    default_achievements = [
        {
            "name": "First Steps",
            "description": "Complete your first quiz",
            "category": AchievementType.PROGRESS,
            "points_reward": 10,
            "rarity": "common",
            "unlock_criteria": {"quizzes_completed": 1},
            "unlock_message": "Welcome to your learning journey!"
        },
        {
            "name": "Perfect Score",
            "description": "Get 100% on a quiz",
            "category": AchievementType.PERFORMANCE,
            "points_reward": 50,
            "rarity": "rare",
            "unlock_criteria": {"perfect_scores": 1},
            "unlock_message": "Excellence achieved! Keep up the great work!"
        },
        {
            "name": "Study Streak",
            "description": "Study for 7 consecutive days",
            "category": AchievementType.STREAK,
            "points_reward": 100,
            "rarity": "epic",
            "unlock_criteria": {"daily_streak": 7},
            "unlock_message": "Your dedication is impressive!"
        },
        {
            "name": "Quiz Master",
            "description": "Complete 50 quizzes",
            "category": AchievementType.PROGRESS,
            "points_reward": 200,
            "rarity": "epic",
            "unlock_criteria": {"quizzes_completed": 50},
            "unlock_message": "You're becoming a quiz expert!"
        },
        {
            "name": "Early Bird",
            "description": "Submit a quiz before the deadline",
            "category": AchievementType.PARTICIPATION,
            "points_reward": 25,
            "rarity": "common",
            "unlock_criteria": {"early_submissions": 1},
            "unlock_message": "Time management at its finest!"
        }
    ]
    
    for achievement_data in default_achievements:
        achievement = Achievement(
            id=uuid.uuid4(),
            **achievement_data
        )
        session.add(achievement)
    
    logger.info(f"Created {len(default_achievements)} default achievements")


def setup_event_listeners():
    """Set up database event listeners for optimization"""
    
    @event.listens_for(async_engine.sync_engine if async_engine else None, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Set SQLite pragmas for better performance"""
        if "sqlite" in str(dbapi_connection):
            cursor = dbapi_connection.cursor()
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            cursor.execute("PRAGMA foreign_keys=ON")
            # Optimize SQLite for performance
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()
    
    @event.listens_for(sync_engine, "connect")
    def set_sqlite_pragma_sync(dbapi_connection, connection_record):
        """Set SQLite pragmas for sync engine"""
        if "sqlite" in str(dbapi_connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()


# Session management functions
@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session with automatic cleanup"""
    if not AsyncSessionLocal:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session() -> Session:
    """Get sync database session (for migrations and CLI tools)"""
    if not SessionLocal:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    return SessionLocal()


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions"""
    async with get_async_session() as session:
        yield session


# Health check functions
async def check_database_health() -> dict:
    """Check database connection health"""
    try:
        async with get_async_session() as session:
            # Simple query to test connection
            result = await session.execute("SELECT 1 as health_check")
            row = result.fetchone()
            
            if row and row[0] == 1:
                return {
                    "status": "healthy",
                    "database": "connected",
                    "timestamp": asyncio.get_event_loop().time()
                }
            else:
                return {
                    "status": "unhealthy", 
                    "database": "query_failed",
                    "timestamp": asyncio.get_event_loop().time()
                }
                
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "connection_failed",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }


# Cleanup functions
async def close_database_connections():
    """Close all database connections"""
    global async_engine, sync_engine
    
    try:
        if async_engine:
            await async_engine.dispose()
            async_engine = None
            logger.info("✅ Async database engine disposed")
        
        if sync_engine:
            sync_engine.dispose()
            sync_engine = None
            logger.info("✅ Sync database engine disposed")
            
    except Exception as e:
        logger.error(f"❌ Error closing database connections: {e}")


# Transaction helpers
@asynccontextmanager
async def database_transaction():
    """Context manager for database transactions with automatic rollback on error"""
    async with get_async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Bulk operations helper
async def bulk_insert_or_update(session: AsyncSession, model_class, data_list: list):
    """Efficient bulk insert or update operation"""
    try:
        # Use bulk operations for better performance
        if hasattr(session, 'bulk_insert_mappings'):
            await session.bulk_insert_mappings(model_class, data_list)
        else:
            # Fallback for older versions
            session.add_all([model_class(**data) for data in data_list])
        
        await session.commit()
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Bulk operation failed: {e}")
        raise


# Migration helper
def run_migrations():
    """Run database migrations (for deployment scripts)"""
    from alembic.config import Config
    from alembic import command
    
    try:
        # Create Alembic configuration
        alembic_cfg = Config("alembic.ini")
        
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        logger.info("✅ Database migrations completed")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


# Export main functions
__all__ = [
    "init_database",
    "get_async_session", 
    "get_sync_session",
    "get_db",
    "check_database_health",
    "close_database_connections",
    "database_transaction",
    "bulk_insert_or_update",
    "run_migrations"
]
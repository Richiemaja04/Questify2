"""
AI-Powered Smart Class & Timetable Scheduler
Application configuration and settings management
"""

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Any, Dict
from urllib.parse import quote_plus

from pydantic import BaseSettings, validator, Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Settings
    APP_NAME: str = "AI-Powered Smart Class Scheduler"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # Security Settings
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_SPECIAL_CHARS: bool = True
    BCRYPT_ROUNDS: int = 12
    
    # CORS Settings
    ALLOWED_HOSTS: List[str] = Field(
        default=["*"], 
        env="ALLOWED_HOSTS",
        description="Comma-separated list of allowed hosts"
    )
    
    @validator('ALLOWED_HOSTS', pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(',')]
        return v
    
    # Database Configuration
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=5432, env="DB_PORT")
    DB_NAME: str = Field(default="smart_scheduler", env="DB_NAME")
    DB_USER: str = Field(default="postgres", env="DB_USER")
    DB_PASSWORD: str = Field(default="password", env="DB_PASSWORD")
    DB_POOL_SIZE: int = Field(default=10, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=20, env="DB_MAX_OVERFLOW")
    DB_ECHO: bool = Field(default=False, env="DB_ECHO")
    
    @property
    def database_url(self) -> str:
        """Generate database URL from components or use provided URL"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        
        # For development, use SQLite
        if self.ENVIRONMENT == "development":
            return "sqlite:///./smart_scheduler.db"
        
        # For production, use PostgreSQL
        password = quote_plus(self.DB_PASSWORD)
        return f"postgresql://{self.DB_USER}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # Redis Configuration (for caching and sessions)
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    CACHE_TTL: int = Field(default=300, env="CACHE_TTL")  # 5 minutes
    
    # AI/ML Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    HUGGINGFACE_API_TOKEN: Optional[str] = Field(default=None, env="HUGGINGFACE_API_TOKEN")
    AI_MODEL_CACHE_DIR: str = Field(default="./models", env="AI_MODEL_CACHE_DIR")
    MAX_CONTENT_GENERATION_LENGTH: int = 2000
    AI_TIMEOUT_SECONDS: int = 30
    ENABLE_AI_FEATURES: bool = Field(default=True, env="ENABLE_AI_FEATURES")
    
    # Content Generation Models
    QUESTION_GENERATION_MODEL: str = "microsoft/DialoGPT-small"
    ESSAY_ANALYSIS_MODEL: str = "distilbert-base-uncased"
    TEXT_CLASSIFICATION_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 50MB
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = [
        ".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
        ".mp3", ".wav", ".mp4", ".avi", ".mov", ".mkv",
        ".csv", ".xlsx", ".xls", ".json", ".xml"
    ]
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    
    # Email Configuration
    SMTP_HOST: Optional[str] = Field(default=None, env="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USER: Optional[str] = Field(default=None, env="SMTP_USER")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    SMTP_TLS: bool = Field(default=True, env="SMTP_TLS")
    FROM_EMAIL: str = Field(default="noreply@smartscheduler.com", env="FROM_EMAIL")
    
    # Notification Configuration
    ENABLE_PUSH_NOTIFICATIONS: bool = Field(default=True, env="ENABLE_PUSH_NOTIFICATIONS")
    VAPID_PUBLIC_KEY: Optional[str] = Field(default=None, env="VAPID_PUBLIC_KEY")
    VAPID_PRIVATE_KEY: Optional[str] = Field(default=None, env="VAPID_PRIVATE_KEY")
    VAPID_CLAIMS: Dict[str, str] = {"sub": "mailto:admin@smartscheduler.com"}
    
    # Gamification Configuration
    BASE_XP_PER_QUIZ: int = 10
    XP_MULTIPLIER_PERFECT_SCORE: float = 2.0
    XP_MULTIPLIER_STREAK: float = 1.5
    MAX_STREAK_MULTIPLIER: float = 3.0
    DAILY_LOGIN_XP: int = 5
    ACHIEVEMENT_UNLOCK_XP_BONUS: int = 50
    
    # Quiz Configuration
    DEFAULT_QUIZ_TIME_LIMIT: int = 60  # minutes
    MAX_QUIZ_ATTEMPTS: int = 3
    MIN_PASS_PERCENTAGE: float = 60.0
    QUESTION_SHUFFLE_DEFAULT: bool = True
    SHOW_RESULTS_IMMEDIATELY: bool = True
    
    # Scheduling Configuration
    SCHEDULE_OPTIMIZATION_TIMEOUT: int = 30  # seconds
    MAX_DAILY_PERIODS: int = 8
    PERIOD_DURATION_MINUTES: int = 45
    BREAK_DURATION_MINUTES: int = 15
    LUNCH_DURATION_MINUTES: int = 60
    MAX_CONSECUTIVE_PERIODS: int = 3
    
    # Analytics Configuration
    ANALYTICS_RETENTION_DAYS: int = 365
    ANONYMIZE_ANALYTICS_AFTER_DAYS: int = 90
    ENABLE_BEHAVIORAL_TRACKING: bool = Field(default=True, env="ENABLE_BEHAVIORAL_TRACKING")
    MAX_ANALYTICS_BATCH_SIZE: int = 1000
    
    # Backup Configuration
    BACKUP_ENABLED: bool = Field(default=True, env="BACKUP_ENABLED")
    BACKUP_INTERVAL_HOURS: int = Field(default=24, env="BACKUP_INTERVAL_HOURS")
    BACKUP_RETENTION_DAYS: int = Field(default=30, env="BACKUP_RETENTION_DAYS")
    BACKUP_STORAGE_PATH: str = Field(default="./backups", env="BACKUP_STORAGE_PATH")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    RATE_LIMIT_BURST: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # Monitoring and Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ENABLE_REQUEST_LOGGING: bool = Field(default=True, env="ENABLE_REQUEST_LOGGING")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # Feature Flags
    ENABLE_OFFLINE_MODE: bool = Field(default=True, env="ENABLE_OFFLINE_MODE")
    ENABLE_REAL_TIME_SYNC: bool = Field(default=True, env="ENABLE_REAL_TIME_SYNC")
    ENABLE_ADVANCED_ANALYTICS: bool = Field(default=True, env="ENABLE_ADVANCED_ANALYTICS")
    ENABLE_AI_PLAGIARISM_DETECTION: bool = Field(default=True, env="ENABLE_AI_PLAGIARISM_DETECTION")
    ENABLE_VOICE_FEATURES: bool = Field(default=False, env="ENABLE_VOICE_FEATURES")
    
    # Development Settings
    RELOAD_ON_CHANGES: bool = Field(default=True, env="RELOAD_ON_CHANGES")
    PROFILING_ENABLED: bool = Field(default=False, env="PROFILING_ENABLED")
    MOCK_AI_RESPONSES: bool = Field(default=False, env="MOCK_AI_RESPONSES")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class DevelopmentSettings(Settings):
    """Development environment specific settings"""
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    DB_ECHO: bool = True
    ENABLE_REQUEST_LOGGING: bool = True
    RELOAD_ON_CHANGES: bool = True
    MOCK_AI_RESPONSES: bool = True


class ProductionSettings(Settings):
    """Production environment specific settings"""
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    DB_ECHO: bool = False
    ENABLE_REQUEST_LOGGING: bool = False
    RELOAD_ON_CHANGES: bool = False
    MOCK_AI_RESPONSES: bool = False
    
    # Require these in production
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")
    DATABASE_URL: str = Field(..., env="DATABASE_URL")


class TestingSettings(Settings):
    """Testing environment specific settings"""
    DEBUG: bool = True
    ENVIRONMENT: str = "testing"
    DATABASE_URL: str = "sqlite:///./test.db"
    MOCK_AI_RESPONSES: bool = True
    ENABLE_REQUEST_LOGGING: bool = False
    
    # Faster settings for tests
    BCRYPT_ROUNDS: int = 4
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 5
    CACHE_TTL: int = 1


@lru_cache()
def get_settings() -> Settings:
    """Get application settings with caching"""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


def get_database_url() -> str:
    """Get the database URL for the current environment"""
    return get_settings().database_url


def get_redis_url() -> str:
    """Get the Redis URL for the current environment"""
    settings = get_settings()
    if settings.REDIS_PASSWORD:
        return f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
    return f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"


# Export commonly used settings
__all__ = [
    "Settings",
    "DevelopmentSettings", 
    "ProductionSettings",
    "TestingSettings",
    "get_settings",
    "get_database_url",
    "get_redis_url"
]
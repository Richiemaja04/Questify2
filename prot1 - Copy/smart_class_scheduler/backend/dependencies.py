"""
AI-Powered Smart Class & Timetable Scheduler
Dependency injection and middleware components
"""

import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import jwt
import redis.asyncio as redis

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .database.connection import get_db
from .database.models import User, UserRole, UserStatus, School
from .exceptions import (
    AuthenticationException, 
    AuthorizationException,
    NotFoundException,
    ValidationException
)
from ..config import get_settings, get_redis_url

# Configure logging
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Redis connection
_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> redis.Redis:
    """Get Redis client instance"""
    global _redis_client
    
    if _redis_client is None:
        try:
            settings = get_settings()
            _redis_client = redis.from_url(
                get_redis_url(),
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            # Test connection
            await _redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            _redis_client = None
    
    return _redis_client


async def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token"""
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Check token expiration
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            raise AuthenticationException("Token has expired")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationException("Token has expired")
    except jwt.JWTError:
        raise AuthenticationException("Invalid token")


async def get_current_user_from_token(
    token: str,
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current user from JWT token"""
    
    # Verify token
    payload = await verify_jwt_token(token)
    user_id = payload.get("sub")
    
    if not user_id:
        raise AuthenticationException("Invalid token payload")
    
    # Check token blacklist (if Redis is available)
    redis_client = await get_redis_client()
    if redis_client:
        is_blacklisted = await redis_client.get(f"blacklist:{token}")
        if is_blacklisted:
            raise AuthenticationException("Token has been revoked")
    
    # Get user from database
    result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.is_deleted == False
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    if user.status != UserStatus.ACTIVE:
        raise AuthorizationException("User account is not active")
    
    # Update last activity
    user.last_login = datetime.utcnow()
    await db.commit()
    
    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Get current authenticated user (optional)"""
    
    if not credentials:
        return None
    
    return await get_current_user_from_token(credentials.credentials, db)


async def require_authentication(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require user authentication"""
    
    if not current_user:
        raise AuthenticationException("Authentication required")
    
    return current_user


async def require_role(allowed_roles: List[UserRole]):
    """Factory function to create role-based dependencies"""
    
    async def check_role(current_user: User = Depends(require_authentication)) -> User:
        if current_user.role not in allowed_roles:
            raise AuthorizationException(
                f"Access denied. Required roles: {[role.value for role in allowed_roles]}"
            )
        return current_user
    
    return check_role


# Pre-built role dependencies
require_student = Depends(require_role([UserRole.STUDENT]))
require_teacher = Depends(require_role([UserRole.TEACHER]))
require_admin = Depends(require_role([UserRole.ADMIN]))
require_teacher_or_admin = Depends(require_role([UserRole.TEACHER, UserRole.ADMIN]))
require_any_role = Depends(require_role([UserRole.STUDENT, UserRole.TEACHER, UserRole.ADMIN]))


async def get_current_school(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
) -> School:
    """Get current user's school"""
    
    if not current_user.school_id:
        raise ValidationException("User is not associated with any school")
    
    result = await db.execute(
        select(School).where(
            School.id == current_user.school_id,
            School.is_deleted == False,
            School.is_active == True
        )
    )
    school = result.scalar_one_or_none()
    
    if not school:
        raise NotFoundException("School not found or inactive")
    
    return school


class RateLimiter:
    """Rate limiting dependency"""
    
    def __init__(self, requests: int, window: int, scope: str = "global"):
        self.requests = requests
        self.window = window
        self.scope = scope
    
    async def __call__(self, request: Request) -> bool:
        redis_client = await get_redis_client()
        
        if not redis_client:
            # If Redis is not available, allow all requests
            logger.warning("Rate limiting disabled: Redis not available")
            return True
        
        # Create rate limit key
        if self.scope == "ip":
            key = f"rate_limit:ip:{request.client.host}"
        elif self.scope == "user":
            # Extract user ID from token if available
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                try:
                    token = auth_header.split(" ")[1]
                    payload = await verify_jwt_token(token)
                    user_id = payload.get("sub")
                    key = f"rate_limit:user:{user_id}"
                except:
                    key = f"rate_limit:ip:{request.client.host}"
            else:
                key = f"rate_limit:ip:{request.client.host}"
        else:
            key = f"rate_limit:global"
        
        # Check and increment counter
        current_requests = await redis_client.get(key)
        
        if current_requests is None:
            # First request in window
            await redis_client.setex(key, self.window, 1)
            return True
        elif int(current_requests) < self.requests:
            # Within limit
            await redis_client.incr(key)
            return True
        else:
            # Rate limit exceeded
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )


# Common rate limiters
standard_rate_limit = RateLimiter(requests=60, window=60, scope="ip")  # 60 req/min per IP
user_rate_limit = RateLimiter(requests=100, window=60, scope="user")   # 100 req/min per user
strict_rate_limit = RateLimiter(requests=10, window=60, scope="ip")    # 10 req/min per IP


class CacheManager:
    """Cache management dependency"""
    
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        redis_client = await get_redis_client()
        
        if not redis_client:
            return None
        
        try:
            import json
            cached_value = await redis_client.get(f"cache:{key}")
            return json.loads(cached_value) if cached_value else None
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any) -> bool:
        """Set value in cache"""
        redis_client = await get_redis_client()
        
        if not redis_client:
            return False
        
        try:
            import json
            await redis_client.setex(f"cache:{key}", self.ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        redis_client = await get_redis_client()
        
        if not redis_client:
            return False
        
        try:
            await redis_client.delete(f"cache:{key}")
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> bool:
        """Invalidate cache keys matching pattern"""
        redis_client = await get_redis_client()
        
        if not redis_client:
            return False
        
        try:
            keys = await redis_client.keys(f"cache:{pattern}")
            if keys:
                await redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.warning(f"Cache invalidation failed for pattern {pattern}: {e}")
            return False


# Cache instances
short_cache = CacheManager(ttl=60)      # 1 minute
standard_cache = CacheManager(ttl=300)   # 5 minutes
long_cache = CacheManager(ttl=3600)      # 1 hour


class RequestContextManager:
    """Request context management"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_id = None
        self.user_id = None
        self.ip_address = None
        self.user_agent = None
    
    def set_context(self, request: Request, user: Optional[User] = None):
        """Set request context"""
        self.request_id = request.headers.get("x-request-id", f"req_{int(time.time() * 1000000)}")
        self.ip_address = request.client.host
        self.user_agent = request.headers.get("user-agent")
        
        if user:
            self.user_id = str(user.id)
    
    def get_processing_time(self) -> float:
        """Get request processing time"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "processing_time": self.get_processing_time()
        }


async def get_request_context(request: Request) -> RequestContextManager:
    """Get request context manager"""
    context = RequestContextManager()
    context.set_context(request)
    return context


class PermissionChecker:
    """Permission checking system"""
    
    @staticmethod
    async def can_access_user_data(
        current_user: User,
        target_user_id: str,
        db: AsyncSession
    ) -> bool:
        """Check if user can access another user's data"""
        
        # Users can access their own data
        if str(current_user.id) == target_user_id:
            return True
        
        # Admins can access any user's data
        if current_user.role == UserRole.ADMIN:
            return True
        
        # Teachers can access their students' data
        if current_user.role == UserRole.TEACHER:
            # Check if the target user is enrolled in any of teacher's classes
            from .database.models import Enrollment, Class
            
            result = await db.execute(
                select(Enrollment).join(Class).where(
                    Class.teacher_id == current_user.id,
                    Enrollment.student_id == target_user_id,
                    Enrollment.status == "active"
                )
            )
            return result.scalar_one_or_none() is not None
        
        return False
    
    @staticmethod
    async def can_modify_quiz(
        current_user: User,
        quiz_id: str,
        db: AsyncSession
    ) -> bool:
        """Check if user can modify a quiz"""
        from .database.models import Quiz
        
        # Admins can modify any quiz
        if current_user.role == UserRole.ADMIN:
            return True
        
        # Teachers can modify their own quizzes
        if current_user.role == UserRole.TEACHER:
            result = await db.execute(
                select(Quiz).where(
                    Quiz.id == quiz_id,
                    Quiz.created_by_teacher_id == current_user.id
                )
            )
            return result.scalar_one_or_none() is not None
        
        return False
    
    @staticmethod
    async def can_access_class(
        current_user: User,
        class_id: str,
        db: AsyncSession
    ) -> bool:
        """Check if user can access a class"""
        from .database.models import Class, Enrollment
        
        # Admins can access any class
        if current_user.role == UserRole.ADMIN:
            return True
        
        # Teachers can access classes they teach
        if current_user.role == UserRole.TEACHER:
            result = await db.execute(
                select(Class).where(
                    Class.id == class_id,
                    Class.teacher_id == current_user.id
                )
            )
            return result.scalar_one_or_none() is not None
        
        # Students can access classes they're enrolled in
        if current_user.role == UserRole.STUDENT:
            result = await db.execute(
                select(Enrollment).where(
                    Enrollment.class_id == class_id,
                    Enrollment.student_id == current_user.id,
                    Enrollment.status == "active"
                )
            )
            return result.scalar_one_or_none() is not None
        
        return False


def create_permission_dependency(permission_func):
    """Factory to create permission-based dependencies"""
    
    async def check_permission(
        resource_id: str,
        current_user: User = Depends(require_authentication),
        db: AsyncSession = Depends(get_db)
    ) -> User:
        has_permission = await permission_func(current_user, resource_id, db)
        
        if not has_permission:
            raise AuthorizationException("Access denied to this resource")
        
        return current_user
    
    return check_permission


# Permission dependencies
require_user_data_access = create_permission_dependency(PermissionChecker.can_access_user_data)
require_quiz_modify_access = create_permission_dependency(PermissionChecker.can_modify_quiz)
require_class_access = create_permission_dependency(PermissionChecker.can_access_class)


async def log_request_analytics(
    request: Request,
    response_time: float,
    user: Optional[User] = None,
    db: Optional[AsyncSession] = None
):
    """Log request analytics (non-blocking)"""
    try:
        if not db:
            return
        
        from .database.models import AnalyticsEvent
        import uuid
        
        # Create analytics event
        event = AnalyticsEvent(
            id=uuid.uuid4(),
            user_id=user.id if user else None,
            session_id=request.headers.get("x-session-id", "anonymous"),
            event_type="api_request",
            event_name=f"{request.method} {request.url.path}",
            page_url=str(request.url),
            referrer_url=request.headers.get("referer"),
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host,
            properties={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "response_time_ms": response_time * 1000,
                "status_code": 200  # Default, will be updated by middleware
            }
        )
        
        db.add(event)
        await db.commit()
        
    except Exception as e:
        logger.warning(f"Analytics logging failed: {e}")


class SessionManager:
    """User session management"""
    
    @staticmethod
    async def create_session(user: User, device_info: Dict[str, Any]) -> str:
        """Create user session"""
        redis_client = await get_redis_client()
        
        if not redis_client:
            return "no_redis_session"
        
        import uuid
        session_id = str(uuid.uuid4())
        
        session_data = {
            "user_id": str(user.id),
            "username": user.username,
            "role": user.role.value,
            "school_id": str(user.school_id),
            "created_at": datetime.utcnow().isoformat(),
            "device_info": device_info,
            "last_activity": datetime.utcnow().isoformat()
        }
        
        settings = get_settings()
        await redis_client.setex(
            f"session:{session_id}",
            settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            json.dumps(session_data)
        )
        
        return session_id
    
    @staticmethod
    async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        redis_client = await get_redis_client()
        
        if not redis_client:
            return None
        
        try:
            import json
            session_data = await redis_client.get(f"session:{session_id}")
            return json.loads(session_data) if session_data else None
        except:
            return None
    
    @staticmethod
    async def update_session_activity(session_id: str):
        """Update session last activity"""
        redis_client = await get_redis_client()
        
        if not redis_client:
            return
        
        session_data = await SessionManager.get_session(session_id)
        if session_data:
            session_data["last_activity"] = datetime.utcnow().isoformat()
            settings = get_settings()
            await redis_client.setex(
                f"session:{session_id}",
                settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                json.dumps(session_data)
            )
    
    @staticmethod
    async def invalidate_session(session_id: str):
        """Invalidate user session"""
        redis_client = await get_redis_client()
        
        if redis_client:
            await redis_client.delete(f"session:{session_id}")


# Cleanup function
async def cleanup_dependencies():
    """Cleanup dependency resources"""
    global _redis_client
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("✅ Redis connection closed")


# Export main dependencies
__all__ = [
    # Authentication
    "get_current_user",
    "require_authentication", 
    "require_role",
    "require_student",
    "require_teacher", 
    "require_admin",
    "require_teacher_or_admin",
    "require_any_role",
    
    # Authorization
    "get_current_school",
    "PermissionChecker",
    "require_user_data_access",
    "require_quiz_modify_access", 
    "require_class_access",
    
    # Rate limiting
    "RateLimiter",
    "standard_rate_limit",
    "user_rate_limit", 
    "strict_rate_limit",
    
    # Caching
    "CacheManager",
    "short_cache",
    "standard_cache",
    "long_cache",
    
    # Context management
    "RequestContextManager",
    "get_request_context",
    "log_request_analytics",
    
    # Session management
    "SessionManager",
    
    # Utilities
    "get_redis_client",
    "cleanup_dependencies"
]
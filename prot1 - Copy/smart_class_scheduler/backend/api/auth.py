"""
AI-Powered Smart Class & Timetable Scheduler
Authentication and authorization API routes
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import jwt
from passlib.context import CryptContext
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from ..database.connection import get_db
from ..database.models import User, UserRole, UserStatus, School, StudentProfile, TeacherProfile
from ..dependencies import (
    get_current_user, 
    require_authentication,
    get_redis_client,
    SessionManager,
    standard_rate_limit
)
from ..exceptions import (
    AuthenticationException,
    InvalidCredentialsException,
    ValidationException,
    ConflictException,
    NotFoundException,
    AccountDisabledException,
    TwoFactorRequiredException
)
from ...config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()


# Pydantic models for requests/responses
class LoginRequest(BaseModel):
    username_or_email: str
    password: str
    remember_me: bool = False
    device_info: Optional[Dict[str, Any]] = None
    
    @validator('username_or_email')
    def validate_username_or_email(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Username or email is required')
        return v.strip().lower()
    
    @validator('password')
    def validate_password(cls, v):
        if not v or len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]
    school: Dict[str, Any]
    permissions: list[str]
    session_id: str


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str
    first_name: str
    last_name: str
    role: UserRole
    school_code: str
    phone: Optional[str] = None
    date_of_birth: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not v.isalnum() and '_' not in v:
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        settings = get_settings()
        if len(v) < settings.PASSWORD_MIN_LENGTH:
            raise ValueError(f'Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long')
        
        if settings.PASSWORD_REQUIRE_SPECIAL_CHARS:
            if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
                raise ValueError('Password must contain at least one special character')
        
        return v
    
    @validator('confirm_password')
    def validate_password_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('first_name', 'last_name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Name is required')
        return v.strip().title()


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str
    
    @validator('confirm_password')
    def validate_password_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
    confirm_password: str
    
    @validator('confirm_password')
    def validate_password_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class TwoFactorSetupResponse(BaseModel):
    secret: str
    qr_code_url: str
    backup_codes: list[str]


class TwoFactorVerifyRequest(BaseModel):
    token: str
    code: str


class UserProfileResponse(BaseModel):
    id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    status: UserStatus
    avatar_url: Optional[str]
    phone: Optional[str]
    preferences: Dict[str, Any]
    last_login: Optional[datetime]
    is_verified: bool
    two_factor_enabled: bool
    school: Dict[str, Any]
    
    class Config:
        from_attributes = True


# Utility functions
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


async def get_user_by_username_or_email(
    username_or_email: str, 
    db: AsyncSession
) -> Optional[User]:
    """Get user by username or email"""
    
    result = await db.execute(
        select(User).where(
            or_(
                User.username == username_or_email,
                User.email == username_or_email
            ),
            User.is_deleted == False
        )
    )
    return result.scalar_one_or_none()


async def create_user_response_data(user: User, db: AsyncSession) -> Dict[str, Any]:
    """Create user response data with profile information"""
    
    # Get school information
    school_result = await db.execute(
        select(School).where(School.id == user.school_id)
    )
    school = school_result.scalar_one_or_none()
    
    # Get role-specific profile
    profile = None
    if user.role == UserRole.STUDENT:
        profile_result = await db.execute(
            select(StudentProfile).where(StudentProfile.user_id == user.id)
        )
        profile = profile_result.scalar_one_or_none()
    elif user.role == UserRole.TEACHER:
        profile_result = await db.execute(
            select(TeacherProfile).where(TeacherProfile.user_id == user.id)
        )
        profile = profile_result.scalar_one_or_none()
    
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "full_name": user.full_name,
        "role": user.role.value,
        "status": user.status.value,
        "avatar_url": user.avatar_url,
        "phone": user.phone,
        "preferences": user.preferences,
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "is_verified": user.is_verified,
        "two_factor_enabled": user.two_factor_enabled,
        "school": {
            "id": str(school.id) if school else None,
            "name": school.name if school else None,
            "code": school.code if school else None
        },
        "profile": {
            "student_id": profile.student_id if hasattr(profile, 'student_id') else None,
            "employee_id": profile.employee_id if hasattr(profile, 'employee_id') else None,
            "grade_level": getattr(profile, 'grade_level', None),
            "department": getattr(profile, 'department', None),
            "total_xp": getattr(profile, 'total_xp', 0),
            "level": getattr(profile, 'level', 1)
        } if profile else None
    }


def get_user_permissions(user: User) -> list[str]:
    """Get user permissions based on role"""
    
    permissions = []
    
    if user.role == UserRole.STUDENT:
        permissions.extend([
            "quiz:take",
            "quiz:view_own_submissions",
            "profile:view_own",
            "profile:update_own",
            "schedule:view_own",
            "grades:view_own",
            "achievements:view_own"
        ])
    elif user.role == UserRole.TEACHER:
        permissions.extend([
            "quiz:create",
            "quiz:update_own",
            "quiz:delete_own",
            "quiz:grade",
            "student:view_enrolled",
            "class:manage_own",
            "schedule:manage_own",
            "content:create",
            "analytics:view_class"
        ])
    elif user.role == UserRole.ADMIN:
        permissions.extend([
            "user:create",
            "user:view_all",
            "user:update_all",
            "user:delete",
            "school:manage",
            "system:configure",
            "analytics:view_all",
            "backup:manage"
        ])
    
    return permissions


# Authentication routes
@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
    rate_limit: bool = Depends(standard_rate_limit)
):
    """Authenticate user and return access tokens"""
    
    # Get user from database
    user = await get_user_by_username_or_email(request.username_or_email, db)
    
    if not user:
        raise InvalidCredentialsException()
    
    # Verify password
    if not verify_password(request.password, user.hashed_password):
        raise InvalidCredentialsException()
    
    # Check account status
    if user.status != UserStatus.ACTIVE:
        if user.status == UserStatus.SUSPENDED:
            raise AccountDisabledException("Account is suspended")
        elif user.status == UserStatus.INACTIVE:
            raise AccountDisabledException("Account is inactive")
        else:
            raise AccountDisabledException("Account is not available")
    
    # Check if 2FA is required
    if user.two_factor_enabled:
        # In a real implementation, you'd handle 2FA flow here
        # For now, we'll skip it but mark it as required
        pass
    
    # Update login tracking
    user.last_login = datetime.utcnow()
    user.login_count = (user.login_count or 0) + 1
    await db.commit()
    
    # Create tokens
    settings = get_settings()
    token_data = {"sub": str(user.id), "username": user.username, "role": user.role.value}
    
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data=token_data, expires_delta=access_token_expires)
    refresh_token = create_refresh_token(data=token_data)
    
    # Create session
    device_info = request.device_info or {}
    session_id = await SessionManager.create_session(user, device_info)
    
    # Get user response data
    user_data = await create_user_response_data(user, db)
    
    # Get permissions
    permissions = get_user_permissions(user)
    
    logger.info(f"User {user.username} logged in successfully")
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_data,
        school=user_data["school"],
        permissions=permissions,
        session_id=session_id
    )


@router.post("/register")
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    rate_limit: bool = Depends(strict_rate_limit)
):
    """Register a new user account"""
    
    # Check if username already exists
    existing_user = await db.execute(
        select(User).where(
            or_(
                User.username == request.username,
                User.email == request.email
            ),
            User.is_deleted == False
        )
    )
    
    if existing_user.scalar_one_or_none():
        raise ConflictException("Username or email already exists")
    
    # Verify school exists and is active
    school_result = await db.execute(
        select(School).where(
            School.code == request.school_code,
            School.is_active == True,
            School.is_deleted == False
        )
    )
    school = school_result.scalar_one_or_none()
    
    if not school:
        raise NotFoundException("Invalid school code")
    
    # Create user
    import uuid
    from dateutil.parser import parse as parse_date
    
    user = User(
        id=uuid.uuid4(),
        username=request.username,
        email=request.email,
        hashed_password=hash_password(request.password),
        first_name=request.first_name,
        last_name=request.last_name,
        role=request.role,
        status=UserStatus.PENDING,  # Require verification
        school_id=school.id,
        phone=request.phone,
        date_of_birth=parse_date(request.date_of_birth).date() if request.date_of_birth else None,
        verification_token=secrets.token_urlsafe(32),
        preferences={
            "theme": "light",
            "language": "en",
            "notifications": {
                "email": True,
                "push": True,
                "achievement": True
            }
        }
    )
    
    db.add(user)
    await db.flush()  # Get the user ID
    
    # Create role-specific profile
    if request.role == UserRole.STUDENT:
        student_profile = StudentProfile(
            id=uuid.uuid4(),
            user_id=user.id,
            student_id=f"{school.code}-{user.username.upper()}",
            admission_date=datetime.utcnow().date(),
            learning_style="mixed",
            academic_interests=[],
            goals=[],
            total_xp=0,
            level=1,
            streak_days=0,
            performance_metrics={}
        )
        db.add(student_profile)
    
    elif request.role == UserRole.TEACHER:
        teacher_profile = TeacherProfile(
            id=uuid.uuid4(),
            user_id=user.id,
            employee_id=f"{school.code}-T-{user.username.upper()}",
            hire_date=datetime.utcnow().date(),
            subjects=[],
            qualifications=[],
            experience_years=0,
            office_hours={},
            teaching_load=1.0,
            specializations=[],
            certifications=[]
        )
        db.add(teacher_profile)
    
    await db.commit()
    
    # TODO: Send verification email
    logger.info(f"New user registered: {user.username} ({user.email})")
    
    return {
        "message": "Registration successful. Please check your email for verification instructions.",
        "user_id": str(user.id),
        "verification_required": True
    }


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token"""
    
    settings = get_settings()
    
    try:
        # Decode refresh token
        payload = jwt.decode(
            request.refresh_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Verify token type
        if payload.get("type") != "refresh":
            raise AuthenticationException("Invalid token type")
        
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationException("Invalid token payload")
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationException("Refresh token has expired")
    except jwt.JWTError:
        raise AuthenticationException("Invalid refresh token")
    
    # Get user from database
    result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.is_deleted == False,
            User.status == UserStatus.ACTIVE
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise AuthenticationException("User not found or inactive")
    
    # Create new tokens
    token_data = {"sub": str(user.id), "username": user.username, "role": user.role.value}
    
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data=token_data, expires_delta=access_token_expires)
    new_refresh_token = create_refresh_token(data=token_data)
    
    # Get user response data
    user_data = await create_user_response_data(user, db)
    permissions = get_user_permissions(user)
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_data,
        school=user_data["school"],
        permissions=permissions,
        session_id="refreshed"
    )


@router.post("/logout")
async def logout(
    response: Response,
    current_user: User = Depends(require_authentication)
):
    """Logout user and invalidate tokens"""
    
    # TODO: Add token to blacklist in Redis
    redis_client = await get_redis_client()
    if redis_client:
        # In a real implementation, you'd blacklist the current token
        pass
    
    # Clear any session cookies
    response.delete_cookie("session_id")
    
    logger.info(f"User {current_user.username} logged out")
    
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's profile information"""
    
    user_data = await create_user_response_data(current_user, db)
    
    return UserProfileResponse(**user_data)


@router.put("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Change user password"""
    
    # Verify current password
    if not verify_password(request.current_password, current_user.hashed_password):
        raise AuthenticationException("Current password is incorrect")
    
    # Update password
    current_user.hashed_password = hash_password(request.new_password)
    current_user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Password changed for user {current_user.username}")
    
    return {"message": "Password changed successfully"}


@router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
    rate_limit: bool = Depends(strict_rate_limit)
):
    """Send password reset email"""
    
    # Find user by email
    result = await db.execute(
        select(User).where(
            User.email == request.email,
            User.is_deleted == False
        )
    )
    user = result.scalar_one_or_none()
    
    if user:
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        reset_expires = datetime.utcnow() + timedelta(hours=24)
        
        user.password_reset_token = reset_token
        user.password_reset_expires = reset_expires
        
        await db.commit()
        
        # TODO: Send reset email
        logger.info(f"Password reset requested for {user.email}")
    
    # Always return success to prevent email enumeration
    return {"message": "If an account with that email exists, a password reset link has been sent."}


@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """Reset password using reset token"""
    
    # Find user by reset token
    result = await db.execute(
        select(User).where(
            User.password_reset_token == request.token,
            User.password_reset_expires > datetime.utcnow(),
            User.is_deleted == False
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise AuthenticationException("Invalid or expired reset token")
    
    # Update password and clear reset token
    user.hashed_password = hash_password(request.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Password reset completed for user {user.username}")
    
    return {"message": "Password reset successfully"}


@router.post("/verify-email")
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """Verify email address using verification token"""
    
    # Find user by verification token
    result = await db.execute(
        select(User).where(
            User.verification_token == token,
            User.is_deleted == False
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise AuthenticationException("Invalid verification token")
    
    # Update user verification status
    user.is_verified = True
    user.status = UserStatus.ACTIVE
    user.verification_token = None
    user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Email verified for user {user.username}")
    
    return {"message": "Email verified successfully"}


# 2FA routes (placeholder for future implementation)
@router.post("/2fa/setup")
async def setup_two_factor(
    current_user: User = Depends(require_authentication)
):
    """Setup two-factor authentication"""
    
    # TODO: Implement 2FA setup
    raise FeatureDisabledException("Two-factor authentication", "Coming soon")


@router.post("/2fa/verify")
async def verify_two_factor(
    request: TwoFactorVerifyRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify two-factor authentication code"""
    
    # TODO: Implement 2FA verification
    raise FeatureDisabledException("Two-factor authentication", "Coming soon")


@router.delete("/2fa/disable")
async def disable_two_factor(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Disable two-factor authentication"""
    
    current_user.two_factor_enabled = False
    current_user.two_factor_secret = None
    current_user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Two-factor authentication disabled"}


# Admin routes for user management
@router.get("/users", dependencies=[Depends(require_admin)])
async def list_users(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """List all users (admin only)"""
    
    result = await db.execute(
        select(User)
        .where(User.is_deleted == False)
        .offset(skip)
        .limit(limit)
        .order_by(User.created_at.desc())
    )
    users = result.scalars().all()
    
    user_list = []
    for user in users:
        user_data = await create_user_response_data(user, db)
        user_list.append(user_data)
    
    return {"users": user_list, "total": len(user_list)}


# Export router
__all__ = ["router"]
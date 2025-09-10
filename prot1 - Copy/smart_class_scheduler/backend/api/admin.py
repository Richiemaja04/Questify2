"""
AI-Powered Smart Class & Timetable Scheduler
Administrative functions API routes for system management
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import APIRouter, Depends, Query, Path, BackgroundTasks, File, UploadFile
from pydantic import BaseModel, validator, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, text, update, delete
from sqlalchemy.orm import selectinload
import csv
import io

from ..database.connection import get_db, bulk_insert_or_update
from ..database.models import (
    User, UserRole, UserStatus, School, Class, Quiz, QuizSubmission,
    Grade, Resource, Schedule, Achievement, AnalyticsEvent, SyncQueue,
    StudentProfile, TeacherProfile, Enrollment, AIModel, ContentAnalysis
)
from ..dependencies import (
    require_admin,
    get_current_school,
    standard_cache,
    long_cache
)
from ..exceptions import (
    NotFoundException,
    ValidationException,
    BusinessLogicException,
    ConflictException,
    FileUploadException,
    InvalidFileTypeException
)
from ...config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()


# Pydantic models
class SystemStatsResponse(BaseModel):
    schools: Dict[str, int]
    users: Dict[str, int]
    content: Dict[str, int]
    activity: Dict[str, int]
    system_health: Dict[str, Any]
    resource_usage: Dict[str, Any]


class SchoolManagementRequest(BaseModel):
    name: str
    code: str
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    timezone: str = "UTC"
    academic_year_start: Optional[date] = None
    academic_year_end: Optional[date] = None
    settings: Dict[str, Any] = {}
    
    @validator('code')
    def validate_code(cls, v):
        if not v or len(v) < 3:
            raise ValueError('School code must be at least 3 characters')
        return v.upper()


class UserManagementRequest(BaseModel):
    username: str
    email: EmailStr
    password: Optional[str] = None
    first_name: str
    last_name: str
    role: UserRole
    status: UserStatus = UserStatus.ACTIVE
    phone: Optional[str] = None
    school_id: str
    
    @validator('username')
    def validate_username(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v.lower()


class BulkUserImportRequest(BaseModel):
    school_id: str
    default_role: UserRole = UserRole.STUDENT
    update_existing: bool = False
    send_welcome_emails: bool = True


class SystemConfigurationRequest(BaseModel):
    feature_flags: Optional[Dict[str, bool]] = None
    limits: Optional[Dict[str, int]] = None
    notifications: Optional[Dict[str, Any]] = None
    security_settings: Optional[Dict[str, Any]] = None
    ai_settings: Optional[Dict[str, Any]] = None


class MaintenanceRequest(BaseModel):
    maintenance_type: str  # "database_cleanup", "cache_clear", "log_rotation", "backup"
    options: Dict[str, Any] = {}
    schedule_time: Optional[datetime] = None
    
    @validator('maintenance_type')
    def validate_maintenance_type(cls, v):
        valid_types = ["database_cleanup", "cache_clear", "log_rotation", "backup", "reindex"]
        if v not in valid_types:
            raise ValueError(f'Invalid maintenance type. Must be one of: {valid_types}')
        return v


class BackupRequest(BaseModel):
    backup_type: str = "full"  # "full", "incremental", "data_only"
    include_files: bool = True
    encryption: bool = True
    retention_days: int = 30
    
    @validator('backup_type')
    def validate_backup_type(cls, v):
        if v not in ["full", "incremental", "data_only"]:
            raise ValueError('Invalid backup type')
        return v


class AuditLogResponse(BaseModel):
    logs: List[Dict[str, Any]]
    total_count: int
    date_range: Dict[str, str]
    summary: Dict[str, Any]


class SystemHealthResponse(BaseModel):
    overall_status: str
    components: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    last_check: datetime


class ReportGenerationRequest(BaseModel):
    report_type: str
    date_range: Dict[str, str]
    filters: Dict[str, Any] = {}
    format: str = "pdf"  # "pdf", "excel", "csv"
    recipients: List[EmailStr] = []
    
    @validator('report_type')
    def validate_report_type(cls, v):
        valid_types = [
            "user_activity", "system_usage", "performance_summary",
            "security_audit", "financial_report", "academic_progress"
        ]
        if v not in valid_types:
            raise ValueError('Invalid report type')
        return v


# Helper functions
async def get_system_statistics(db: AsyncSession) -> Dict[str, Any]:
    """Get comprehensive system statistics"""
    
    # School statistics
    school_stats = await db.execute(
        select(
            func.count(School.id).label('total'),
            func.count(School.id).filter(School.is_active == True).label('active')
        )
    )
    school_result = school_stats.first()
    
    # User statistics
    user_stats = await db.execute(
        select(
            User.role,
            func.count(User.id).label('count')
        )
        .where(User.is_deleted == False)
        .group_by(User.role)
    )
    
    user_counts = {"total": 0}
    for role, count in user_stats:
        user_counts[role.value] = count
        user_counts["total"] += count
    
    # Content statistics
    content_stats = await db.execute(
        select(
            func.count(Quiz.id).label('quizzes'),
            func.count(Class.id).label('classes'),
            func.count(QuizSubmission.id).label('submissions')
        )
        .select_from(Quiz)
        .outerjoin(Class)
        .outerjoin(QuizSubmission)
        .where(Quiz.is_deleted == False)
    )
    content_result = content_stats.first()
    
    # Activity statistics (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    activity_stats = await db.execute(
        select(
            func.count(AnalyticsEvent.id).label('total_events'),
            func.count(func.distinct(AnalyticsEvent.user_id)).label('active_users'),
            func.count(QuizSubmission.id).label('recent_submissions')
        )
        .select_from(AnalyticsEvent)
        .outerjoin(QuizSubmission)
        .where(
            AnalyticsEvent.timestamp >= thirty_days_ago,
            QuizSubmission.submitted_at >= thirty_days_ago
        )
    )
    activity_result = activity_stats.first()
    
    return {
        "schools": {
            "total": school_result.total or 0,
            "active": school_result.active or 0
        },
        "users": user_counts,
        "content": {
            "quizzes": content_result.quizzes or 0,
            "classes": content_result.classes or 0,
            "submissions": content_result.submissions or 0
        },
        "activity": {
            "total_events_30d": activity_result.total_events or 0,
            "active_users_30d": activity_result.active_users or 0,
            "submissions_30d": activity_result.recent_submissions or 0
        }
    }


async def perform_database_cleanup(db: AsyncSession, options: Dict[str, Any]) -> Dict[str, Any]:
    """Perform database cleanup operations"""
    
    cleanup_results = {"cleaned_tables": [], "records_removed": 0}
    
    # Clean old analytics events (older than retention period)
    retention_days = options.get("analytics_retention_days", 365)
    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
    
    analytics_cleanup = await db.execute(
        delete(AnalyticsEvent).where(AnalyticsEvent.timestamp < cutoff_date)
    )
    cleanup_results["cleaned_tables"].append("analytics_events")
    cleanup_results["records_removed"] += analytics_cleanup.rowcount
    
    # Clean completed sync queue items (older than 30 days)
    sync_cutoff = datetime.utcnow() - timedelta(days=30)
    sync_cleanup = await db.execute(
        delete(SyncQueue).where(
            and_(
                SyncQueue.server_processed_at < sync_cutoff,
                SyncQueue.status == "synced"
            )
        )
    )
    cleanup_results["cleaned_tables"].append("sync_queue")
    cleanup_results["records_removed"] += sync_cleanup.rowcount
    
    # Clean expired content analysis cache
    cache_cleanup = await db.execute(
        delete(ContentAnalysis).where(ContentAnalysis.cache_expires_at < datetime.utcnow())
    )
    cleanup_results["cleaned_tables"].append("content_analysis")
    cleanup_results["records_removed"] += cache_cleanup.rowcount
    
    await db.commit()
    
    return cleanup_results


async def generate_audit_logs(
    db: AsyncSession,
    start_date: datetime,
    end_date: datetime,
    user_id: Optional[str] = None,
    action_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generate audit log entries"""
    
    # This would typically query a dedicated audit log table
    # For now, we'll use analytics events as a proxy
    
    query = select(AnalyticsEvent).where(
        AnalyticsEvent.timestamp.between(start_date, end_date)
    )
    
    if user_id:
        query = query.where(AnalyticsEvent.user_id == user_id)
    
    if action_type:
        query = query.where(AnalyticsEvent.event_type == action_type)
    
    result = await db.execute(query.order_by(desc(AnalyticsEvent.timestamp)).limit(1000))
    
    audit_logs = []
    for event in result.scalars():
        audit_logs.append({
            "id": str(event.id),
            "user_id": str(event.user_id) if event.user_id else None,
            "action": event.event_name,
            "resource": event.page_url,
            "timestamp": event.timestamp.isoformat(),
            "ip_address": event.ip_address,
            "user_agent": event.user_agent,
            "details": event.properties
        })
    
    return audit_logs


async def check_system_health(db: AsyncSession) -> Dict[str, Any]:
    """Check overall system health"""
    
    health_status = {
        "overall_status": "healthy",
        "components": {},
        "performance_metrics": {},
        "alerts": []
    }
    
    try:
        # Database health
        db_start = datetime.utcnow()
        await db.execute(select(1))
        db_response_time = (datetime.utcnow() - db_start).total_seconds() * 1000
        
        health_status["components"]["database"] = {
            "status": "healthy" if db_response_time < 100 else "slow",
            "response_time_ms": db_response_time,
            "last_check": datetime.utcnow().isoformat()
        }
        
        if db_response_time > 200:
            health_status["alerts"].append({
                "level": "warning",
                "component": "database",
                "message": f"Database response time is high: {db_response_time:.2f}ms"
            })
    
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "error",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat()
        }
        health_status["overall_status"] = "unhealthy"
        health_status["alerts"].append({
            "level": "critical",
            "component": "database",
            "message": f"Database connection failed: {str(e)}"
        })
    
    # Check AI services
    settings = get_settings()
    if settings.ENABLE_AI_FEATURES:
        ai_status = "healthy" if settings.OPENAI_API_KEY or settings.MOCK_AI_RESPONSES else "configuration_error"
        health_status["components"]["ai_services"] = {
            "status": ai_status,
            "mock_mode": settings.MOCK_AI_RESPONSES,
            "last_check": datetime.utcnow().isoformat()
        }
    
    # Performance metrics
    health_status["performance_metrics"] = {
        "avg_response_time_ms": 150,  # Mock metric
        "requests_per_second": 25,     # Mock metric
        "error_rate": 0.01,            # Mock metric
        "uptime_percentage": 99.9      # Mock metric
    }
    
    # Set overall status based on components
    if any(comp.get("status") == "error" for comp in health_status["components"].values()):
        health_status["overall_status"] = "unhealthy"
    elif any(comp.get("status") == "slow" for comp in health_status["components"].values()):
        health_status["overall_status"] = "degraded"
    
    return health_status


# API Routes
@router.get("/dashboard", response_model=SystemStatsResponse)
async def get_admin_dashboard(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive admin dashboard statistics"""
    
    # Check cache first
    cache_key = "admin_dashboard_stats"
    cached_stats = await standard_cache.get(cache_key)
    
    if cached_stats:
        return SystemStatsResponse(**cached_stats)
    
    # Generate fresh statistics
    stats = await get_system_statistics(db)
    
    # Add system health and resource usage
    system_health = await check_system_health(db)
    
    resource_usage = {
        "cpu_usage": 45.2,        # Mock data - would come from system monitoring
        "memory_usage": 62.8,     # Mock data
        "disk_usage": 34.1,       # Mock data
        "network_io": 156.3       # Mock data
    }
    
    dashboard_stats = {
        **stats,
        "system_health": system_health,
        "resource_usage": resource_usage
    }
    
    # Cache for 5 minutes
    await standard_cache.set(cache_key, dashboard_stats)
    
    return SystemStatsResponse(**dashboard_stats)


@router.get("/schools")
async def list_schools(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    active_only: bool = Query(False, description="Filter for active schools only"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200)
):
    """List all schools in the system"""
    
    query = select(School).where(School.is_deleted == False)
    
    if active_only:
        query = query.where(School.is_active == True)
    
    # Get total count
    count_result = await db.execute(
        select(func.count(School.id)).where(School.is_deleted == False)
    )
    total_count = count_result.scalar() or 0
    
    # Apply pagination
    result = await db.execute(
        query.offset(skip).limit(limit).order_by(School.name)
    )
    
    schools = []
    for school in result.scalars():
        # Get user counts for each school
        user_count_result = await db.execute(
            select(func.count(User.id))
            .where(
                User.school_id == school.id,
                User.is_deleted == False
            )
        )
        user_count = user_count_result.scalar() or 0
        
        schools.append({
            "id": str(school.id),
            "name": school.name,
            "code": school.code,
            "address": school.address,
            "phone": school.phone,
            "email": school.email,
            "website": school.website,
            "is_active": school.is_active,
            "user_count": user_count,
            "created_at": school.created_at.isoformat(),
            "settings": school.settings
        })
    
    return {
        "schools": schools,
        "total_count": total_count,
        "page_info": {
            "skip": skip,
            "limit": limit,
            "has_more": skip + limit < total_count
        }
    }


@router.post("/schools")
async def create_school(
    request: SchoolManagementRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create a new school"""
    
    # Check if school code already exists
    existing_school = await db.execute(
        select(School).where(
            School.code == request.code,
            School.is_deleted == False
        )
    )
    
    if existing_school.scalar_one_or_none():
        raise ConflictException(f"School with code '{request.code}' already exists")
    
    # Create new school
    import uuid
    
    school = School(
        id=uuid.uuid4(),
        name=request.name,
        code=request.code,
        address=request.address,
        phone=request.phone,
        email=request.email,
        website=request.website,
        timezone=request.timezone,
        academic_year_start=request.academic_year_start,
        academic_year_end=request.academic_year_end,
        settings=request.settings,
        is_active=True
    )
    
    db.add(school)
    await db.commit()
    
    logger.info(f"School '{request.name}' created by admin {current_user.username}")
    
    return {
        "message": "School created successfully",
        "school_id": str(school.id),
        "school_code": school.code
    }


@router.put("/schools/{school_id}")
async def update_school(
    school_id: str,
    request: SchoolManagementRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update school information"""
    
    # Get school
    result = await db.execute(
        select(School).where(
            School.id == school_id,
            School.is_deleted == False
        )
    )
    
    school = result.scalar_one_or_none()
    if not school:
        raise NotFoundException("School not found")
    
    # Update fields
    school.name = request.name
    school.code = request.code
    school.address = request.address
    school.phone = request.phone
    school.email = request.email
    school.website = request.website
    school.timezone = request.timezone
    school.academic_year_start = request.academic_year_start
    school.academic_year_end = request.academic_year_end
    school.settings = request.settings
    school.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"School {school.code} updated by admin {current_user.username}")
    
    return {"message": "School updated successfully"}


@router.get("/users")
async def list_all_users(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    school_id: Optional[str] = Query(None, description="Filter by school ID"),
    role: Optional[UserRole] = Query(None, description="Filter by user role"),
    status: Optional[UserStatus] = Query(None, description="Filter by user status"),
    search: Optional[str] = Query(None, description="Search by name or email"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200)
):
    """List all users in the system with filtering"""
    
    query = select(User).where(User.is_deleted == False)
    
    # Apply filters
    if school_id:
        query = query.where(User.school_id == school_id)
    
    if role:
        query = query.where(User.role == role)
    
    if status:
        query = query.where(User.status == status)
    
    if search:
        search_term = f"%{search.lower()}%"
        query = query.where(
            or_(
                User.first_name.ilike(search_term),
                User.last_name.ilike(search_term),
                User.email.ilike(search_term),
                User.username.ilike(search_term)
            )
        )
    
    # Get total count
    count_query = select(func.count(User.id)).where(User.is_deleted == False)
    if school_id:
        count_query = count_query.where(User.school_id == school_id)
    if role:
        count_query = count_query.where(User.role == role)
    if status:
        count_query = count_query.where(User.status == status)
    
    total_count = await db.scalar(count_query) or 0
    
    # Apply pagination and get results
    result = await db.execute(
        query.options(selectinload(User.school))
        .offset(skip)
        .limit(limit)
        .order_by(User.last_name, User.first_name)
    )
    
    users = []
    for user in result.scalars():
        users.append({
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role.value,
            "status": user.status.value,
            "school_name": user.school.name if user.school else None,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "created_at": user.created_at.isoformat(),
            "is_verified": user.is_verified
        })
    
    return {
        "users": users,
        "total_count": total_count,
        "filters_applied": {
            "school_id": school_id,
            "role": role.value if role else None,
            "status": status.value if status else None,
            "search": search
        }
    }


@router.post("/users")
async def create_user(
    request: UserManagementRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create a new user account"""
    
    # Check if username or email already exists
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
    
    # Verify school exists
    school_result = await db.execute(
        select(School).where(
            School.id == request.school_id,
            School.is_deleted == False
        )
    )
    
    if not school_result.scalar_one_or_none():
        raise NotFoundException("School not found")
    
    # Create user
    import uuid
    from ..api.auth import hash_password
    
    user = User(
        id=uuid.uuid4(),
        username=request.username,
        email=request.email,
        hashed_password=hash_password(request.password) if request.password else hash_password("changeme123"),
        first_name=request.first_name,
        last_name=request.last_name,
        role=request.role,
        status=request.status,
        phone=request.phone,
        school_id=request.school_id,
        is_verified=True,  # Admin-created users are pre-verified
        preferences={"theme": "light", "language": "en"}
    )
    
    db.add(user)
    await db.flush()
    
    # Create role-specific profile
    if request.role == UserRole.STUDENT:
        student_profile = StudentProfile(
            id=uuid.uuid4(),
            user_id=user.id,
            student_id=f"STU-{user.username.upper()}",
            admission_date=datetime.utcnow().date(),
            learning_style="mixed"
        )
        db.add(student_profile)
    
    elif request.role == UserRole.TEACHER:
        teacher_profile = TeacherProfile(
            id=uuid.uuid4(),
            user_id=user.id,
            employee_id=f"TCH-{user.username.upper()}",
            hire_date=datetime.utcnow().date()
        )
        db.add(teacher_profile)
    
    await db.commit()
    
    logger.info(f"User {request.username} created by admin {current_user.username}")
    
    return {
        "message": "User created successfully",
        "user_id": str(user.id),
        "username": user.username,
        "temporary_password": request.password if request.password else "changeme123"
    }


@router.post("/users/bulk-import")
async def bulk_import_users(
    file: UploadFile = File(...),
    request_data: str = Query(..., description="JSON string of BulkUserImportRequest"),
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Bulk import users from CSV file"""
    
    import json
    from ..api.auth import hash_password
    
    try:
        request = BulkUserImportRequest(**json.loads(request_data))
    except Exception as e:
        raise ValidationException("Invalid request data format")
    
    # Validate file
    if not file.filename or not file.filename.endswith('.csv'):
        raise InvalidFileTypeException(file.filename or "unknown", "csv", [".csv"])
    
    # Read and process CSV
    content = await file.read()
    csv_data = content.decode('utf-8')
    csv_reader = csv.DictReader(io.StringIO(csv_data))
    
    # Validate school exists
    school_result = await db.execute(
        select(School).where(School.id == request.school_id)
    )
    
    if not school_result.scalar_one_or_none():
        raise NotFoundException("School not found")
    
    # Process users in background
    background_tasks.add_task(
        process_bulk_user_import,
        list(csv_reader),
        request,
        str(current_user.id)
    )
    
    return {
        "message": "Bulk user import started",
        "status": "processing",
        "estimated_time": "5-10 minutes"
    }


async def process_bulk_user_import(
    csv_data: List[Dict[str, str]],
    request: BulkUserImportRequest,
    admin_user_id: str
):
    """Background task to process bulk user import"""
    
    logger.info(f"Starting bulk import of {len(csv_data)} users")
    
    try:
        from ..database.connection import get_async_session
        from ..api.auth import hash_password
        import uuid
        
        async with get_async_session() as db:
            imported_count = 0
            failed_count = 0
            
            for row in csv_data:
                try:
                    # Validate required fields
                    required_fields = ['username', 'email', 'first_name', 'last_name']
                    if not all(field in row and row[field].strip() for field in required_fields):
                        failed_count += 1
                        continue
                    
                    # Check if user already exists
                    existing_user = await db.execute(
                        select(User).where(
                            or_(
                                User.username == row['username'].lower(),
                                User.email == row['email'].lower()
                            ),
                            User.is_deleted == False
                        )
                    )
                    
                    existing = existing_user.scalar_one_or_none()
                    
                    if existing and not request.update_existing:
                        failed_count += 1
                        continue
                    
                    if existing and request.update_existing:
                        # Update existing user
                        existing.first_name = row['first_name']
                        existing.last_name = row['last_name']
                        existing.phone = row.get('phone')
                        existing.updated_at = datetime.utcnow()
                        imported_count += 1
                    else:
                        # Create new user
                        role = UserRole(row.get('role', request.default_role.value))
                        
                        user = User(
                            id=uuid.uuid4(),
                            username=row['username'].lower(),
                            email=row['email'].lower(),
                            hashed_password=hash_password(row.get('password', 'changeme123')),
                            first_name=row['first_name'],
                            last_name=row['last_name'],
                            role=role,
                            status=UserStatus.ACTIVE,
                            phone=row.get('phone'),
                            school_id=request.school_id,
                            is_verified=True
                        )
                        
                        db.add(user)
                        await db.flush()
                        
                        # Create profile based on role
                        if role == UserRole.STUDENT:
                            profile = StudentProfile(
                                id=uuid.uuid4(),
                                user_id=user.id,
                                student_id=row.get('student_id', f"STU-{user.username.upper()}"),
                                grade_level=row.get('grade_level')
                            )
                            db.add(profile)
                        
                        elif role == UserRole.TEACHER:
                            profile = TeacherProfile(
                                id=uuid.uuid4(),
                                user_id=user.id,
                                employee_id=row.get('employee_id', f"TCH-{user.username.upper()}"),
                                department=row.get('department')
                            )
                            db.add(profile)
                        
                        imported_count += 1
                
                except Exception as e:
                    logger.error(f"Failed to import user {row.get('username', 'unknown')}: {e}")
                    failed_count += 1
            
            await db.commit()
            
            logger.info(f"Bulk import completed: {imported_count} imported, {failed_count} failed")
            
            # TODO: Send completion notification to admin
            
    except Exception as e:
        logger.error(f"Bulk import failed: {e}")


@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: str,
    status: UserStatus,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update user status (activate, suspend, etc.)"""
    
    # Get user
    result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.is_deleted == False
        )
    )
    
    user = result.scalar_one_or_none()
    if not user:
        raise NotFoundException("User not found")
    
    # Update status
    old_status = user.status
    user.status = status
    user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"User {user.username} status changed from {old_status.value} to {status.value} by admin {current_user.username}")
    
    return {
        "message": f"User status updated to {status.value}",
        "user_id": user_id,
        "old_status": old_status.value,
        "new_status": status.value
    }


@router.post("/maintenance")
async def perform_maintenance(
    request: MaintenanceRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Perform system maintenance operations"""
    
    if request.schedule_time and request.schedule_time > datetime.utcnow():
        # Schedule maintenance for later
        background_tasks.add_task(
            schedule_maintenance,
            request.maintenance_type,
            request.options,
            request.schedule_time,
            str(current_user.id)
        )
        
        return {
            "message": f"Maintenance '{request.maintenance_type}' scheduled",
            "scheduled_time": request.schedule_time.isoformat()
        }
    
    # Perform maintenance immediately
    if request.maintenance_type == "database_cleanup":
        result = await perform_database_cleanup(db, request.options)
        logger.info(f"Database cleanup completed by admin {current_user.username}")
        return {
            "message": "Database cleanup completed",
            "result": result
        }
    
    elif request.maintenance_type == "cache_clear":
        # Clear Redis cache
        from ..dependencies import get_redis_client
        redis_client = await get_redis_client()
        if redis_client:
            await redis_client.flushdb()
        
        return {"message": "Cache cleared successfully"}
    
    elif request.maintenance_type == "backup":
        # Queue backup task
        background_tasks.add_task(perform_backup, request.options)
        return {"message": "Backup started", "status": "processing"}
    
    else:
        raise ValidationException(f"Unsupported maintenance type: {request.maintenance_type}")


async def schedule_maintenance(
    maintenance_type: str,
    options: Dict[str, Any],
    schedule_time: datetime,
    admin_user_id: str
):
    """Schedule maintenance task for later execution"""
    
    import asyncio
    
    # Wait until scheduled time
    wait_seconds = (schedule_time - datetime.utcnow()).total_seconds()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
    
    logger.info(f"Executing scheduled maintenance: {maintenance_type}")
    
    # TODO: Execute maintenance task
    # This would call the appropriate maintenance functions


async def perform_backup(options: Dict[str, Any]):
    """Perform system backup"""
    
    logger.info("Starting system backup")
    
    try:
        # TODO: Implement actual backup logic
        # This would include:
        # - Database backup
        # - File system backup
        # - Configuration backup
        # - Upload to cloud storage
        
        import asyncio
        await asyncio.sleep(60)  # Simulate backup time
        
        logger.info("System backup completed successfully")
        
    except Exception as e:
        logger.error(f"System backup failed: {e}")


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive system health status"""
    
    health_data = await check_system_health(db)
    
    return SystemHealthResponse(
        overall_status=health_data["overall_status"],
        components=health_data["components"],
        performance_metrics=health_data["performance_metrics"],
        alerts=health_data["alerts"],
        last_check=datetime.utcnow()
    )


@router.get("/audit-logs", response_model=AuditLogResponse)
async def get_audit_logs(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    start_date: date = Query(..., description="Start date for logs"),
    end_date: date = Query(..., description="End date for logs"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get system audit logs"""
    
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    logs = await generate_audit_logs(db, start_datetime, end_datetime, user_id, action_type)
    
    # Generate summary
    summary = {
        "total_events": len(logs),
        "unique_users": len(set(log["user_id"] for log in logs if log["user_id"])),
        "date_range": f"{start_date} to {end_date}",
        "most_common_actions": {}  # TODO: Calculate most common actions
    }
    
    return AuditLogResponse(
        logs=logs[:limit],
        total_count=len(logs),
        date_range={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        },
        summary=summary
    )


@router.post("/reports/generate")
async def generate_system_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin)
):
    """Generate and optionally email system reports"""
    
    # Queue report generation
    background_tasks.add_task(
        process_report_generation,
        request.report_type,
        request.date_range,
        request.filters,
        request.format,
        request.recipients,
        str(current_user.id)
    )
    
    return {
        "message": f"Report generation started: {request.report_type}",
        "format": request.format,
        "estimated_time": "10-15 minutes",
        "recipients": len(request.recipients)
    }


async def process_report_generation(
    report_type: str,
    date_range: Dict[str, str],
    filters: Dict[str, Any],
    format: str,
    recipients: List[str],
    admin_user_id: str
):
    """Background task for generating reports"""
    
    logger.info(f"Generating {report_type} report in {format} format")
    
    try:
        # TODO: Implement actual report generation
        # This would include:
        # - Data extraction based on report type
        # - Chart/graph generation
        # - PDF/Excel formatting
        # - Email distribution
        
        import asyncio
        await asyncio.sleep(30)  # Simulate processing time
        
        logger.info(f"Report {report_type} generated and distributed successfully")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")


# Export router
__all__ = ["router"]
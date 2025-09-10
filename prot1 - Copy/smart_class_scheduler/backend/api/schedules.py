"""
AI-Powered Smart Class & Timetable Scheduler
Schedule and timetable management API routes
"""

import logging
from datetime import datetime, date, time as dt_time, timedelta
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Query, Path, BackgroundTasks
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..database.connection import get_db
from ..database.models import (
    User, UserRole, Schedule, EventType, Class, Resource, 
    ScheduleConflict, Enrollment, TeacherProfile, School
)
from ..dependencies import (
    require_authentication,
    require_teacher,
    require_admin,
    require_teacher_or_admin,
    get_current_school,
    PermissionChecker,
    standard_cache,
    user_rate_limit
)
from ..exceptions import (
    NotFoundException,
    AuthorizationException,
    ValidationException,
    BusinessLogicException,
    ScheduleConflictException,
    ConflictException
)

# Configure logging
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()


# Pydantic models
class ScheduleResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    event_type: EventType
    start_time: datetime
    end_time: datetime
    day_of_week: int
    is_recurring: bool
    recurrence_rule: Optional[str]
    class_name: Optional[str]
    class_code: Optional[str]
    resource_name: Optional[str]
    resource_location: Optional[str]
    teacher_name: Optional[str]
    status: str
    attendance_required: bool
    max_attendees: Optional[int]
    enrolled_count: Optional[int]
    conflicts: List[Dict[str, Any]]
    optimization_score: Optional[float]
    
    class Config:
        from_attributes = True


class ScheduleCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    event_type: EventType = EventType.CLASS
    start_time: datetime
    end_time: datetime
    class_id: Optional[str] = None
    resource_id: Optional[str] = None
    is_recurring: bool = False
    recurrence_rule: Optional[str] = None
    attendance_required: bool = True
    max_attendees: Optional[int] = None
    special_instructions: Optional[str] = None
    
    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title is required')
        return v.strip()
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be after start time')
        return v
    
    @validator('max_attendees')
    def validate_max_attendees(cls, v):
        if v is not None and v < 1:
            raise ValueError('Max attendees must be at least 1')
        return v


class ScheduleUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    resource_id: Optional[str] = None
    status: Optional[str] = None
    attendance_required: Optional[bool] = None
    max_attendees: Optional[int] = None
    special_instructions: Optional[str] = None


class ResourceResponse(BaseModel):
    id: str
    name: str
    type: str
    capacity: Optional[int]
    location: Optional[str]
    description: Optional[str]
    features: List[str]
    is_active: bool
    availability: Dict[str, Any]
    current_bookings: List[Dict[str, Any]]
    
    class Config:
        from_attributes = True


class ResourceCreateRequest(BaseModel):
    name: str
    type: str
    capacity: Optional[int] = None
    location: Optional[str] = None
    description: Optional[str] = None
    features: List[str] = []
    availability: Dict[str, Any] = {}
    booking_rules: Dict[str, Any] = {}
    cost_per_hour: float = 0.0
    
    @validator('name', 'type')
    def validate_required_fields(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field is required')
        return v.strip()


class TimetableOptimizationRequest(BaseModel):
    school_id: str
    academic_year: str
    semester: str
    optimization_goals: List[str] = ["minimize_conflicts", "balance_workload", "optimize_resources"]
    constraints: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}
    
    @validator('optimization_goals')
    def validate_goals(cls, v):
        valid_goals = [
            "minimize_conflicts", "balance_workload", "optimize_resources",
            "minimize_gaps", "respect_preferences", "maximize_utilization"
        ]
        for goal in v:
            if goal not in valid_goals:
                raise ValueError(f'Invalid optimization goal: {goal}')
        return v


class ConflictResolutionRequest(BaseModel):
    conflict_id: str
    resolution_type: str  # "reschedule", "reassign_resource", "split_class", "ignore"
    new_schedule_data: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    
    @validator('resolution_type')
    def validate_resolution_type(cls, v):
        valid_types = ["reschedule", "reassign_resource", "split_class", "ignore"]
        if v not in valid_types:
            raise ValueError('Invalid resolution type')
        return v


class TimetableResponse(BaseModel):
    school_id: str
    academic_year: str
    semester: str
    schedules: List[ScheduleResponse]
    conflicts: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    optimization_score: float
    generated_at: datetime


class AvailabilityCheckRequest(BaseModel):
    start_time: datetime
    end_time: datetime
    resource_ids: Optional[List[str]] = None
    teacher_id: Optional[str] = None
    exclude_schedule_id: Optional[str] = None


# Helper functions
async def check_schedule_conflicts(
    schedule_data: Dict[str, Any],
    db: AsyncSession,
    exclude_schedule_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Check for scheduling conflicts"""
    
    conflicts = []
    
    # Check resource conflicts
    if schedule_data.get("resource_id"):
        resource_conflicts = await db.execute(
            select(Schedule)
            .where(
                Schedule.resource_id == schedule_data["resource_id"],
                Schedule.status == "active",
                or_(
                    and_(
                        Schedule.start_time <= schedule_data["start_time"],
                        Schedule.end_time > schedule_data["start_time"]
                    ),
                    and_(
                        Schedule.start_time < schedule_data["end_time"],
                        Schedule.end_time >= schedule_data["end_time"]
                    ),
                    and_(
                        Schedule.start_time >= schedule_data["start_time"],
                        Schedule.end_time <= schedule_data["end_time"]
                    )
                )
            )
        )
        
        if exclude_schedule_id:
            resource_conflicts = resource_conflicts.where(Schedule.id != exclude_schedule_id)
        
        for conflict_schedule in resource_conflicts.scalars():
            conflicts.append({
                "type": "resource",
                "description": f"Resource '{schedule_data['resource_id']}' is already booked",
                "conflicting_schedule_id": str(conflict_schedule.id),
                "conflict_time": f"{conflict_schedule.start_time} - {conflict_schedule.end_time}",
                "severity": "high"
            })
    
    # Check teacher conflicts
    if schedule_data.get("class_id"):
        class_result = await db.execute(
            select(Class).where(Class.id == schedule_data["class_id"])
        )
        class_obj = class_result.scalar_one_or_none()
        
        if class_obj:
            teacher_conflicts = await db.execute(
                select(Schedule)
                .join(Class)
                .where(
                    Class.teacher_id == class_obj.teacher_id,
                    Schedule.status == "active",
                    or_(
                        and_(
                            Schedule.start_time <= schedule_data["start_time"],
                            Schedule.end_time > schedule_data["start_time"]
                        ),
                        and_(
                            Schedule.start_time < schedule_data["end_time"],
                            Schedule.end_time >= schedule_data["end_time"]
                        ),
                        and_(
                            Schedule.start_time >= schedule_data["start_time"],
                            Schedule.end_time <= schedule_data["end_time"]
                        )
                    )
                )
            )
            
            if exclude_schedule_id:
                teacher_conflicts = teacher_conflicts.where(Schedule.id != exclude_schedule_id)
            
            for conflict_schedule in teacher_conflicts.scalars():
                conflicts.append({
                    "type": "teacher",
                    "description": f"Teacher has conflicting schedule",
                    "conflicting_schedule_id": str(conflict_schedule.id),
                    "conflict_time": f"{conflict_schedule.start_time} - {conflict_schedule.end_time}",
                    "severity": "high"
                })
    
    return conflicts


async def create_conflict_records(
    schedule_id: str,
    conflicts: List[Dict[str, Any]],
    db: AsyncSession
) -> None:
    """Create conflict records in database"""
    
    import uuid
    
    for conflict in conflicts:
        conflict_record = ScheduleConflict(
            id=uuid.uuid4(),
            schedule_id=schedule_id,
            conflict_type=conflict["type"],
            description=conflict["description"],
            severity=conflict.get("severity", "medium"),
            conflicting_schedule_id=conflict.get("conflicting_schedule_id"),
            resolution_status="pending",
            auto_resolvable=conflict.get("auto_resolvable", False),
            suggested_resolution=conflict.get("suggested_resolution", {})
        )
        
        db.add(conflict_record)


async def calculate_schedule_optimization_score(
    schedule: Schedule,
    db: AsyncSession
) -> float:
    """Calculate optimization score for a schedule"""
    
    score = 100.0
    
    # Penalty for conflicts
    conflicts_count = await db.execute(
        select(func.count(ScheduleConflict.id))
        .where(
            ScheduleConflict.schedule_id == schedule.id,
            ScheduleConflict.resolution_status == "pending"
        )
    )
    
    conflict_penalty = (conflicts_count.scalar() or 0) * 20
    score -= conflict_penalty
    
    # Bonus for resource utilization efficiency
    if schedule.resource_id:
        resource_result = await db.execute(
            select(Resource).where(Resource.id == schedule.resource_id)
        )
        resource = resource_result.scalar_one_or_none()
        
        if resource and resource.capacity:
            # Get enrolled count for class
            if schedule.class_id:
                enrolled_count = await db.execute(
                    select(func.count(Enrollment.id))
                    .where(
                        Enrollment.class_id == schedule.class_id,
                        Enrollment.status == "active"
                    )
                )
                
                utilization = (enrolled_count.scalar() or 0) / resource.capacity
                if 0.7 <= utilization <= 0.9:  # Optimal utilization range
                    score += 10
    
    # Penalty for scheduling outside optimal hours
    hour = schedule.start_time.hour
    if hour < 8 or hour > 17:  # Outside normal school hours
        score -= 5
    
    return max(0.0, min(100.0, score))


# API Routes
@router.get("/", response_model=List[ScheduleResponse])
async def list_schedules(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db),
    start_date: Optional[date] = Query(None, description="Filter schedules from this date"),
    end_date: Optional[date] = Query(None, description="Filter schedules until this date"),
    class_id: Optional[str] = Query(None, description="Filter by class ID"),
    resource_id: Optional[str] = Query(None, description="Filter by resource ID"),
    event_type: Optional[EventType] = Query(None, description="Filter by event type"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """List schedules based on user role and filters"""
    
    query = select(Schedule).options(
        selectinload(Schedule.class_obj),
        selectinload(Schedule.resource),
        selectinload(Schedule.conflicts)
    ).where(Schedule.is_deleted == False)
    
    # Role-based filtering
    if current_user.role == UserRole.STUDENT:
        # Students see schedules for their enrolled classes
        query = query.join(Class).join(Enrollment).where(
            Enrollment.student_id == current_user.id,
            Enrollment.status == "active"
        )
    elif current_user.role == UserRole.TEACHER:
        # Teachers see schedules for their classes
        query = query.join(Class).where(
            Class.teacher_id == current_user.id
        )
    # Admins see all schedules (no additional filter)
    
    # Apply date filters
    if start_date:
        query = query.where(Schedule.start_time >= datetime.combine(start_date, dt_time.min))
    
    if end_date:
        query = query.where(Schedule.end_time <= datetime.combine(end_date, dt_time.max))
    
    # Apply other filters
    if class_id:
        query = query.where(Schedule.class_id == class_id)
    
    if resource_id:
        query = query.where(Schedule.resource_id == resource_id)
    
    if event_type:
        query = query.where(Schedule.event_type == event_type)
    
    # Apply pagination and ordering
    query = query.offset(skip).limit(limit).order_by(Schedule.start_time)
    
    result = await db.execute(query)
    schedules = result.scalars().all()
    
    schedule_list = []
    for schedule in schedules:
        # Get enrolled count if it's a class
        enrolled_count = None
        if schedule.class_obj:
            enrollment_result = await db.execute(
                select(func.count(Enrollment.id))
                .where(
                    Enrollment.class_id == schedule.class_id,
                    Enrollment.status == "active"
                )
            )
            enrolled_count = enrollment_result.scalar()
        
        # Format conflicts
        conflicts = []
        for conflict in schedule.conflicts:
            conflicts.append({
                "id": str(conflict.id),
                "type": conflict.conflict_type,
                "description": conflict.description,
                "severity": conflict.severity,
                "status": conflict.resolution_status
            })
        
        schedule_list.append(ScheduleResponse(
            id=str(schedule.id),
            title=schedule.title,
            description=schedule.description,
            event_type=schedule.event_type,
            start_time=schedule.start_time,
            end_time=schedule.end_time,
            day_of_week=schedule.day_of_week,
            is_recurring=schedule.is_recurring,
            recurrence_rule=schedule.recurrence_rule,
            class_name=schedule.class_obj.name if schedule.class_obj else None,
            class_code=schedule.class_obj.code if schedule.class_obj else None,
            resource_name=schedule.resource.name if schedule.resource else None,
            resource_location=schedule.resource.location if schedule.resource else None,
            teacher_name=f"{schedule.class_obj.teacher.user.first_name} {schedule.class_obj.teacher.user.last_name}" if schedule.class_obj else None,
            status=schedule.status,
            attendance_required=schedule.attendance_required,
            max_attendees=schedule.max_attendees,
            enrolled_count=enrolled_count,
            conflicts=conflicts,
            optimization_score=schedule.optimization_score
        ))
    
    return schedule_list


@router.post("/", response_model=Dict[str, str])
async def create_schedule(
    request: ScheduleCreateRequest,
    current_user: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks
):
    """Create a new schedule entry"""
    
    # Validate class access for teachers
    if request.class_id and current_user.role == UserRole.TEACHER:
        can_access = await PermissionChecker.can_access_class(
            current_user, request.class_id, db
        )
        if not can_access:
            raise AuthorizationException("Access denied to this class")
    
    # Check for conflicts
    schedule_data = {
        "start_time": request.start_time,
        "end_time": request.end_time,
        "class_id": request.class_id,
        "resource_id": request.resource_id
    }
    
    conflicts = await check_schedule_conflicts(schedule_data, db)
    
    # Create schedule
    import uuid
    
    schedule = Schedule(
        id=uuid.uuid4(),
        title=request.title,
        description=request.description,
        event_type=request.event_type,
        start_time=request.start_time,
        end_time=request.end_time,
        day_of_week=request.start_time.weekday(),
        class_id=request.class_id,
        resource_id=request.resource_id,
        is_recurring=request.is_recurring,
        recurrence_rule=request.recurrence_rule,
        status="active",
        attendance_required=request.attendance_required,
        max_attendees=request.max_attendees,
        special_instructions=request.special_instructions
    )
    
    db.add(schedule)
    await db.flush()  # Get the ID
    
    # Calculate optimization score
    optimization_score = await calculate_schedule_optimization_score(schedule, db)
    schedule.optimization_score = optimization_score
    
    # Create conflict records
    if conflicts:
        await create_conflict_records(str(schedule.id), conflicts, db)
        
        # If there are high-severity conflicts, notify admin
        high_severity_conflicts = [c for c in conflicts if c.get("severity") == "high"]
        if high_severity_conflicts:
            background_tasks.add_task(
                notify_scheduling_conflicts,
                str(schedule.id),
                high_severity_conflicts
            )
    
    await db.commit()
    
    logger.info(f"Schedule created: {request.title} by {current_user.username}")
    
    response = {
        "message": "Schedule created successfully",
        "schedule_id": str(schedule.id),
        "optimization_score": optimization_score
    }
    
    if conflicts:
        response["warning"] = f"Schedule created with {len(conflicts)} conflict(s)"
        response["conflicts"] = conflicts
    
    return response


async def notify_scheduling_conflicts(
    schedule_id: str,
    conflicts: List[Dict[str, Any]]
):
    """Background task to notify about scheduling conflicts"""
    # TODO: Implement notification system
    logger.warning(f"Schedule {schedule_id} has {len(conflicts)} high-severity conflicts")


@router.get("/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(
    schedule_id: str = Path(..., description="Schedule ID"),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed schedule information"""
    
    result = await db.execute(
        select(Schedule)
        .options(
            selectinload(Schedule.class_obj),
            selectinload(Schedule.resource),
            selectinload(Schedule.conflicts)
        )
        .where(
            Schedule.id == schedule_id,
            Schedule.is_deleted == False
        )
    )
    
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise NotFoundException("Schedule not found")
    
    # Check permissions
    if current_user.role == UserRole.TEACHER and schedule.class_obj:
        if schedule.class_obj.teacher_id != current_user.id:
            raise AuthorizationException("Access denied to this schedule")
    
    elif current_user.role == UserRole.STUDENT:
        # Check if student is enrolled in the class
        if schedule.class_obj:
            enrollment_result = await db.execute(
                select(Enrollment).where(
                    Enrollment.student_id == current_user.id,
                    Enrollment.class_id == schedule.class_id,
                    Enrollment.status == "active"
                )
            )
            
            if not enrollment_result.scalar_one_or_none():
                raise AuthorizationException("Access denied to this schedule")
    
    # Get enrollment count
    enrolled_count = None
    if schedule.class_obj:
        enrollment_result = await db.execute(
            select(func.count(Enrollment.id))
            .where(
                Enrollment.class_id == schedule.class_id,
                Enrollment.status == "active"
            )
        )
        enrolled_count = enrollment_result.scalar()
    
    # Format conflicts
    conflicts = []
    for conflict in schedule.conflicts:
        conflicts.append({
            "id": str(conflict.id),
            "type": conflict.conflict_type,
            "description": conflict.description,
            "severity": conflict.severity,
            "status": conflict.resolution_status,
            "suggested_resolution": conflict.suggested_resolution
        })
    
    return ScheduleResponse(
        id=str(schedule.id),
        title=schedule.title,
        description=schedule.description,
        event_type=schedule.event_type,
        start_time=schedule.start_time,
        end_time=schedule.end_time,
        day_of_week=schedule.day_of_week,
        is_recurring=schedule.is_recurring,
        recurrence_rule=schedule.recurrence_rule,
        class_name=schedule.class_obj.name if schedule.class_obj else None,
        class_code=schedule.class_obj.code if schedule.class_obj else None,
        resource_name=schedule.resource.name if schedule.resource else None,
        resource_location=schedule.resource.location if schedule.resource else None,
        teacher_name=f"{schedule.class_obj.teacher.user.first_name} {schedule.class_obj.teacher.user.last_name}" if schedule.class_obj else None,
        status=schedule.status,
        attendance_required=schedule.attendance_required,
        max_attendees=schedule.max_attendees,
        enrolled_count=enrolled_count,
        conflicts=conflicts,
        optimization_score=schedule.optimization_score
    )


@router.put("/{schedule_id}")
async def update_schedule(
    schedule_id: str,
    request: ScheduleUpdateRequest,
    current_user: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update schedule information"""
    
    # Get schedule
    result = await db.execute(
        select(Schedule)
        .options(selectinload(Schedule.class_obj))
        .where(
            Schedule.id == schedule_id,
            Schedule.is_deleted == False
        )
    )
    
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise NotFoundException("Schedule not found")
    
    # Check permissions
    if current_user.role == UserRole.TEACHER and schedule.class_obj:
        if schedule.class_obj.teacher_id != current_user.id:
            raise AuthorizationException("Access denied to modify this schedule")
    
    # Store original data for conflict checking
    original_data = {
        "start_time": schedule.start_time,
        "end_time": schedule.end_time,
        "resource_id": schedule.resource_id
    }
    
    # Update fields
    updated = False
    
    if request.title is not None:
        schedule.title = request.title
        updated = True
    
    if request.description is not None:
        schedule.description = request.description
        updated = True
    
    if request.start_time is not None:
        schedule.start_time = request.start_time
        schedule.day_of_week = request.start_time.weekday()
        updated = True
    
    if request.end_time is not None:
        schedule.end_time = request.end_time
        updated = True
    
    if request.resource_id is not None:
        schedule.resource_id = request.resource_id
        updated = True
    
    if request.status is not None:
        schedule.status = request.status
        updated = True
    
    if request.attendance_required is not None:
        schedule.attendance_required = request.attendance_required
        updated = True
    
    if request.max_attendees is not None:
        schedule.max_attendees = request.max_attendees
        updated = True
    
    if request.special_instructions is not None:
        schedule.special_instructions = request.special_instructions
        updated = True
    
    if not updated:
        return {"message": "No changes made"}
    
    # Check for new conflicts if time or resource changed
    if (request.start_time is not None or 
        request.end_time is not None or 
        request.resource_id is not None):
        
        schedule_data = {
            "start_time": schedule.start_time,
            "end_time": schedule.end_time,
            "class_id": schedule.class_id,
            "resource_id": schedule.resource_id
        }
        
        conflicts = await check_schedule_conflicts(
            schedule_data, db, exclude_schedule_id=schedule_id
        )
        
        # Remove existing conflicts and create new ones
        await db.execute(
            select(ScheduleConflict).where(ScheduleConflict.schedule_id == schedule_id)
        )
        # TODO: Actually delete existing conflicts
        
        if conflicts:
            await create_conflict_records(schedule_id, conflicts, db)
    
    # Recalculate optimization score
    schedule.optimization_score = await calculate_schedule_optimization_score(schedule, db)
    schedule.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Schedule {schedule_id} updated by {current_user.username}")
    
    return {"message": "Schedule updated successfully"}


@router.delete("/{schedule_id}")
async def delete_schedule(
    schedule_id: str,
    current_user: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete (soft delete) a schedule"""
    
    # Get schedule
    result = await db.execute(
        select(Schedule)
        .options(selectinload(Schedule.class_obj))
        .where(
            Schedule.id == schedule_id,
            Schedule.is_deleted == False
        )
    )
    
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise NotFoundException("Schedule not found")
    
    # Check permissions
    if current_user.role == UserRole.TEACHER and schedule.class_obj:
        if schedule.class_obj.teacher_id != current_user.id:
            raise AuthorizationException("Access denied to delete this schedule")
    
    # Soft delete
    schedule.is_deleted = True
    schedule.deleted_at = datetime.utcnow()
    schedule.status = "cancelled"
    
    await db.commit()
    
    logger.info(f"Schedule {schedule_id} deleted by {current_user.username}")
    
    return {"message": "Schedule deleted successfully"}


@router.get("/resources/", response_model=List[ResourceResponse])
async def list_resources(
    current_user: User = Depends(require_authentication),
    school = Depends(get_current_school),
    db: AsyncSession = Depends(get_db),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    available_only: bool = Query(False, description="Show only available resources")
):
    """List available resources"""
    
    query = select(Resource).where(
        Resource.school_id == school.id,
        Resource.is_deleted == False
    )
    
    if resource_type:
        query = query.where(Resource.type == resource_type)
    
    if available_only:
        query = query.where(Resource.is_active == True)
    
    result = await db.execute(query.order_by(Resource.name))
    resources = result.scalars().all()
    
    resource_list = []
    for resource in resources:
        # Get current bookings
        current_bookings = await db.execute(
            select(Schedule)
            .where(
                Schedule.resource_id == resource.id,
                Schedule.status == "active",
                Schedule.end_time > datetime.utcnow()
            )
            .order_by(Schedule.start_time)
        )
        
        bookings = []
        for booking in current_bookings.scalars():
            bookings.append({
                "schedule_id": str(booking.id),
                "title": booking.title,
                "start_time": booking.start_time.isoformat(),
                "end_time": booking.end_time.isoformat()
            })
        
        resource_list.append(ResourceResponse(
            id=str(resource.id),
            name=resource.name,
            type=resource.type,
            capacity=resource.capacity,
            location=resource.location,
            description=resource.description,
            features=resource.features,
            is_active=resource.is_active,
            availability=resource.availability,
            current_bookings=bookings
        ))
    
    return resource_list


@router.post("/resources/", dependencies=[Depends(require_admin)])
async def create_resource(
    request: ResourceCreateRequest,
    school = Depends(get_current_school),
    db: AsyncSession = Depends(get_db)
):
    """Create a new resource (admin only)"""
    
    # Check if resource name already exists
    existing_resource = await db.execute(
        select(Resource).where(
            Resource.school_id == school.id,
            Resource.name == request.name,
            Resource.is_deleted == False
        )
    )
    
    if existing_resource.scalar_one_or_none():
        raise ConflictException(f"Resource with name '{request.name}' already exists")
    
    # Create resource
    import uuid
    
    resource = Resource(
        id=uuid.uuid4(),
        name=request.name,
        type=request.type,
        capacity=request.capacity,
        location=request.location,
        description=request.description,
        features=request.features,
        availability=request.availability,
        booking_rules=request.booking_rules,
        cost_per_hour=request.cost_per_hour,
        school_id=school.id,
        is_active=True
    )
    
    db.add(resource)
    await db.commit()
    
    logger.info(f"Resource '{request.name}' created")
    
    return {
        "message": "Resource created successfully",
        "resource_id": str(resource.id)
    }


@router.post("/check-availability")
async def check_availability(
    request: AvailabilityCheckRequest,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Check availability for scheduling"""
    
    availability_result = {
        "available": True,
        "conflicts": [],
        "suggestions": []
    }
    
    # Check resource availability
    if request.resource_ids:
        for resource_id in request.resource_ids:
            conflicts = await db.execute(
                select(Schedule)
                .where(
                    Schedule.resource_id == resource_id,
                    Schedule.status == "active",
                    or_(
                        and_(
                            Schedule.start_time <= request.start_time,
                            Schedule.end_time > request.start_time
                        ),
                        and_(
                            Schedule.start_time < request.end_time,
                            Schedule.end_time >= request.end_time
                        ),
                        and_(
                            Schedule.start_time >= request.start_time,
                            Schedule.end_time <= request.end_time
                        )
                    )
                )
            )
            
            if request.exclude_schedule_id:
                conflicts = conflicts.where(Schedule.id != request.exclude_schedule_id)
            
            conflict_schedules = conflicts.scalars().all()
            
            if conflict_schedules:
                availability_result["available"] = False
                for conflict in conflict_schedules:
                    availability_result["conflicts"].append({
                        "type": "resource",
                        "resource_id": resource_id,
                        "conflicting_schedule": str(conflict.id),
                        "conflict_time": f"{conflict.start_time} - {conflict.end_time}"
                    })
    
    # Check teacher availability
    if request.teacher_id:
        teacher_conflicts = await db.execute(
            select(Schedule)
            .join(Class)
            .where(
                Class.teacher_id == request.teacher_id,
                Schedule.status == "active",
                or_(
                    and_(
                        Schedule.start_time <= request.start_time,
                        Schedule.end_time > request.start_time
                    ),
                    and_(
                        Schedule.start_time < request.end_time,
                        Schedule.end_time >= request.end_time
                    ),
                    and_(
                        Schedule.start_time >= request.start_time,
                        Schedule.end_time <= request.end_time
                    )
                )
            )
        )
        
        if request.exclude_schedule_id:
            teacher_conflicts = teacher_conflicts.where(Schedule.id != request.exclude_schedule_id)
        
        conflict_schedules = teacher_conflicts.scalars().all()
        
        if conflict_schedules:
            availability_result["available"] = False
            for conflict in conflict_schedules:
                availability_result["conflicts"].append({
                    "type": "teacher",
                    "teacher_id": request.teacher_id,
                    "conflicting_schedule": str(conflict.id),
                    "conflict_time": f"{conflict.start_time} - {conflict.end_time}"
                })
    
    # Generate suggestions if not available
    if not availability_result["available"]:
        # TODO: Implement smart suggestions based on conflicts
        availability_result["suggestions"] = [
            "Try scheduling 1 hour earlier or later",
            "Consider using a different resource",
            "Check if the class can be split into smaller groups"
        ]
    
    return availability_result


@router.post("/optimize-timetable", dependencies=[Depends(require_admin)])
async def optimize_timetable(
    request: TimetableOptimizationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Optimize school timetable using AI algorithms"""
    
    # This would trigger the AI optimization engine
    background_tasks.add_task(
        run_timetable_optimization,
        request.school_id,
        request.academic_year,
        request.semester,
        request.optimization_goals,
        request.constraints,
        request.preferences
    )
    
    return {
        "message": "Timetable optimization started",
        "status": "processing",
        "estimated_completion": "5-10 minutes"
    }


async def run_timetable_optimization(
    school_id: str,
    academic_year: str,
    semester: str,
    goals: List[str],
    constraints: Dict[str, Any],
    preferences: Dict[str, Any]
):
    """Background task for timetable optimization"""
    
    logger.info(f"Starting timetable optimization for school {school_id}")
    
    try:
        # TODO: Implement actual AI optimization algorithm
        # This would use OR-Tools, genetic algorithms, or other optimization techniques
        
        # Placeholder for optimization logic
        import asyncio
        await asyncio.sleep(10)  # Simulate processing time
        
        logger.info("Timetable optimization completed successfully")
        
        # TODO: Update schedules with optimized times
        # TODO: Send notification to admin about completion
        
    except Exception as e:
        logger.error(f"Timetable optimization failed: {e}")
        # TODO: Send error notification to admin


@router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(
    conflict_id: str,
    request: ConflictResolutionRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Resolve a scheduling conflict"""
    
    # Get conflict
    result = await db.execute(
        select(ScheduleConflict).where(ScheduleConflict.id == conflict_id)
    )
    
    conflict = result.scalar_one_or_none()
    if not conflict:
        raise NotFoundException("Conflict not found")
    
    # Apply resolution
    if request.resolution_type == "reschedule":
        if not request.new_schedule_data:
            raise ValidationException("New schedule data required for rescheduling")
        
        # Update the conflicting schedule
        schedule_result = await db.execute(
            select(Schedule).where(Schedule.id == conflict.schedule_id)
        )
        schedule = schedule_result.scalar_one()
        
        if request.new_schedule_data.get("start_time"):
            schedule.start_time = datetime.fromisoformat(request.new_schedule_data["start_time"])
        if request.new_schedule_data.get("end_time"):
            schedule.end_time = datetime.fromisoformat(request.new_schedule_data["end_time"])
        if request.new_schedule_data.get("resource_id"):
            schedule.resource_id = request.new_schedule_data["resource_id"]
        
        # Recalculate optimization score
        schedule.optimization_score = await calculate_schedule_optimization_score(schedule, db)
    
    elif request.resolution_type == "ignore":
        # Mark conflict as resolved but don't change anything
        pass
    
    # Update conflict status
    conflict.resolution_status = "resolved"
    conflict.resolution_notes = request.notes
    
    await db.commit()
    
    logger.info(f"Conflict {conflict_id} resolved using {request.resolution_type}")
    
    return {"message": f"Conflict resolved using {request.resolution_type}"}


# Export router
__all__ = ["router"]
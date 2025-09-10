"""
AI-Powered Smart Class & Timetable Scheduler
Teacher-related API routes
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Query, Path
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, or_
from sqlalchemy.orm import selectinload

from ..database.connection import get_db
from ..database.models import (
    User, UserRole, TeacherProfile, Class, Enrollment, 
    Quiz, QuizSubmission, Grade, Student, StudentProfile,
    Schedule, Resource, AnalyticsEvent, Notification,
    Achievement, XPTransaction
)
from ..dependencies import (
    require_authentication,
    require_teacher,
    require_teacher_or_admin,
    require_admin,
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
    ConflictException
)

# Configure logging
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()


# Pydantic models
class TeacherProfileResponse(BaseModel):
    id: str
    user_id: str
    employee_id: str
    department: Optional[str]
    subjects: List[str]
    qualifications: List[str]
    experience_years: int
    hire_date: Optional[date]
    office_location: Optional[str]
    office_hours: Dict[str, Any]
    teaching_load: float
    specializations: List[str]
    certifications: List[str]
    performance_rating: float
    
    class Config:
        from_attributes = True


class UpdateTeacherProfileRequest(BaseModel):
    department: Optional[str] = None
    subjects: Optional[List[str]] = None
    qualifications: Optional[List[str]] = None
    experience_years: Optional[int] = None
    office_location: Optional[str] = None
    office_hours: Optional[Dict[str, Any]] = None
    specializations: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    
    @validator('experience_years')
    def validate_experience(cls, v):
        if v is not None and (v < 0 or v > 50):
            raise ValueError('Experience years must be between 0 and 50')
        return v


class TeacherDashboardResponse(BaseModel):
    profile: TeacherProfileResponse
    classes_taught: List[Dict[str, Any]]
    recent_submissions: List[Dict[str, Any]]
    pending_grades: int
    total_students: int
    average_class_performance: float
    upcoming_classes: List[Dict[str, Any]]
    notifications: List[Dict[str, Any]]
    quick_stats: Dict[str, Any]


class ClassCreationRequest(BaseModel):
    name: str
    code: str
    description: Optional[str] = None
    subject: str
    grade_level: Optional[str] = None
    section: Optional[str] = None
    max_students: int = 30
    credits: float = 1.0
    academic_year: str
    semester: str
    meeting_schedule: Optional[Dict[str, Any]] = None
    grading_scheme: Optional[Dict[str, Any]] = None
    
    @validator('name', 'code', 'subject')
    def validate_required_fields(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field is required')
        return v.strip()
    
    @validator('max_students')
    def validate_max_students(cls, v):
        if v < 1 or v > 200:
            raise ValueError('Max students must be between 1 and 200')
        return v
    
    @validator('credits')
    def validate_credits(cls, v):
        if v < 0.1 or v > 10.0:
            raise ValueError('Credits must be between 0.1 and 10.0')
        return v


class ClassUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    max_students: Optional[int] = None
    meeting_schedule: Optional[Dict[str, Any]] = None
    grading_scheme: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class StudentEnrollmentRequest(BaseModel):
    student_ids: List[str]
    enrollment_date: Optional[date] = None
    
    @validator('student_ids')
    def validate_student_ids(cls, v):
        if not v:
            raise ValueError('At least one student ID is required')
        return v


class BulkGradeRequest(BaseModel):
    grades: List[Dict[str, Any]]  # [{"student_id": "...", "grade": 85.5, "feedback": "..."}]
    
    @validator('grades')
    def validate_grades(cls, v):
        if not v:
            raise ValueError('At least one grade is required')
        
        for grade in v:
            if 'student_id' not in grade or 'grade' not in grade:
                raise ValueError('Each grade must have student_id and grade')
            
            if not isinstance(grade['grade'], (int, float)):
                raise ValueError('Grade must be a number')
            
            if grade['grade'] < 0 or grade['grade'] > 100:
                raise ValueError('Grade must be between 0 and 100')
        
        return v


class TeacherAnalyticsResponse(BaseModel):
    class_performance: List[Dict[str, Any]]
    student_engagement: Dict[str, Any]
    quiz_statistics: Dict[str, Any]
    grade_distribution: Dict[str, Any]
    improvement_trends: List[Dict[str, Any]]
    at_risk_students: List[Dict[str, Any]]
    teaching_effectiveness: Dict[str, Any]


# Helper functions
async def get_teacher_profile(
    teacher_id: str,
    db: AsyncSession,
    current_user: User
) -> TeacherProfile:
    """Get teacher profile with permission checking"""
    
    # Check if user can access this teacher's data
    if str(current_user.id) != teacher_id and current_user.role != UserRole.ADMIN:
        raise AuthorizationException("Access denied to teacher data")
    
    result = await db.execute(
        select(TeacherProfile)
        .options(selectinload(TeacherProfile.user))
        .where(
            TeacherProfile.user_id == teacher_id,
            TeacherProfile.is_deleted == False
        )
    )
    
    profile = result.scalar_one_or_none()
    if not profile:
        raise NotFoundException("Teacher profile not found")
    
    return profile


async def calculate_teacher_statistics(
    teacher_profile: TeacherProfile,
    db: AsyncSession
) -> Dict[str, Any]:
    """Calculate teacher performance statistics"""
    
    # Get total number of students across all classes
    students_count_result = await db.execute(
        select(func.count(func.distinct(Enrollment.student_id)))
        .join(Class)
        .where(
            Class.teacher_id == teacher_profile.user_id,
            Enrollment.status == "active",
            Class.is_active == True
        )
    )
    total_students = students_count_result.scalar() or 0
    
    # Get average class performance
    avg_performance_result = await db.execute(
        select(func.avg(QuizSubmission.percentage))
        .join(Quiz)
        .join(Class)
        .where(
            Class.teacher_id == teacher_profile.user_id,
            QuizSubmission.status == "submitted"
        )
    )
    average_class_performance = float(avg_performance_result.scalar() or 0)
    
    # Get pending grades count
    pending_grades_result = await db.execute(
        select(func.count(QuizSubmission.id))
        .join(Quiz)
        .join(Class)
        .where(
            Class.teacher_id == teacher_profile.user_id,
            QuizSubmission.status == "submitted"
        )
    )
    pending_grades = pending_grades_result.scalar() or 0
    
    # Get number of classes taught
    classes_count_result = await db.execute(
        select(func.count(Class.id))
        .where(
            Class.teacher_id == teacher_profile.user_id,
            Class.is_active == True
        )
    )
    classes_taught = classes_count_result.scalar() or 0
    
    # Get total quizzes created
    quizzes_count_result = await db.execute(
        select(func.count(Quiz.id))
        .where(Quiz.created_by_teacher_id == teacher_profile.user_id)
    )
    total_quizzes = quizzes_count_result.scalar() or 0
    
    return {
        "total_students": total_students,
        "average_class_performance": round(average_class_performance, 2),
        "pending_grades": pending_grades,
        "classes_taught": classes_taught,
        "total_quizzes": total_quizzes
    }


async def get_at_risk_students(
    teacher_profile: TeacherProfile,
    db: AsyncSession
) -> List[Dict[str, Any]]:
    """Identify at-risk students based on performance"""
    
    # Students with average score < 60% in last 5 submissions
    at_risk_result = await db.execute(
        select(
            User.id,
            User.first_name,
            User.last_name,
            StudentProfile.student_id,
            func.avg(QuizSubmission.percentage).label('avg_score'),
            func.count(QuizSubmission.id).label('submission_count')
        )
        .select_from(User)
        .join(StudentProfile)
        .join(Enrollment)
        .join(Class)
        .join(Quiz)
        .join(QuizSubmission)
        .where(
            Class.teacher_id == teacher_profile.user_id,
            Enrollment.status == "active",
            QuizSubmission.status == "submitted"
        )
        .group_by(User.id, User.first_name, User.last_name, StudentProfile.student_id)
        .having(func.avg(QuizSubmission.percentage) < 60)
        .order_by(func.avg(QuizSubmission.percentage))
    )
    
    at_risk_students = []
    for row in at_risk_result:
        at_risk_students.append({
            "student_id": str(row.id),
            "name": f"{row.first_name} {row.last_name}",
            "student_number": row.student_id,
            "average_score": round(row.avg_score, 2),
            "submission_count": row.submission_count,
            "risk_level": "high" if row.avg_score < 50 else "medium"
        })
    
    return at_risk_students


# API Routes
@router.get("/profile", response_model=TeacherProfileResponse)
async def get_my_teacher_profile(
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db)
):
    """Get current teacher's profile"""
    
    profile = await get_teacher_profile(str(current_user.id), db, current_user)
    
    return TeacherProfileResponse(
        id=str(profile.id),
        user_id=str(profile.user_id),
        employee_id=profile.employee_id,
        department=profile.department,
        subjects=profile.subjects,
        qualifications=profile.qualifications,
        experience_years=profile.experience_years,
        hire_date=profile.hire_date,
        office_location=profile.office_location,
        office_hours=profile.office_hours,
        teaching_load=profile.teaching_load,
        specializations=profile.specializations,
        certifications=profile.certifications,
        performance_rating=profile.performance_rating
    )


@router.put("/profile")
async def update_my_teacher_profile(
    request: UpdateTeacherProfileRequest,
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db)
):
    """Update current teacher's profile"""
    
    profile = await get_teacher_profile(str(current_user.id), db, current_user)
    
    # Update fields
    if request.department is not None:
        profile.department = request.department
    if request.subjects is not None:
        profile.subjects = request.subjects
    if request.qualifications is not None:
        profile.qualifications = request.qualifications
    if request.experience_years is not None:
        profile.experience_years = request.experience_years
    if request.office_location is not None:
        profile.office_location = request.office_location
    if request.office_hours is not None:
        profile.office_hours = request.office_hours
    if request.specializations is not None:
        profile.specializations = request.specializations
    if request.certifications is not None:
        profile.certifications = request.certifications
    
    profile.updated_at = datetime.utcnow()
    await db.commit()
    
    logger.info(f"Profile updated for teacher {profile.employee_id}")
    
    return {"message": "Profile updated successfully"}


@router.get("/dashboard", response_model=TeacherDashboardResponse)
async def get_teacher_dashboard(
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive teacher dashboard data"""
    
    profile = await get_teacher_profile(str(current_user.id), db, current_user)
    
    # Get classes taught
    classes_result = await db.execute(
        select(Class)
        .where(
            Class.teacher_id == current_user.id,
            Class.is_active == True
        )
        .order_by(Class.name)
    )
    classes_taught = [
        {
            "id": str(cls.id),
            "name": cls.name,
            "code": cls.code,
            "subject": cls.subject,
            "grade_level": cls.grade_level,
            "section": cls.section,
            "max_students": cls.max_students,
            "enrolled_count": await get_class_enrollment_count(cls.id, db)
        }
        for cls in classes_result.scalars()
    ]
    
    # Get recent quiz submissions
    recent_submissions_result = await db.execute(
        select(QuizSubmission, Quiz, User, Class)
        .join(Quiz)
        .join(User, QuizSubmission.student_id == User.id)
        .join(Class, Quiz.class_id == Class.id)
        .where(
            Class.teacher_id == current_user.id,
            QuizSubmission.status == "submitted"
        )
        .order_by(desc(QuizSubmission.submitted_at))
        .limit(10)
    )
    
    recent_submissions = []
    for submission, quiz, student, class_obj in recent_submissions_result:
        recent_submissions.append({
            "id": str(submission.id),
            "student_name": f"{student.first_name} {student.last_name}",
            "quiz_title": quiz.title,
            "class_name": class_obj.name,
            "score": submission.percentage,
            "submitted_at": submission.submitted_at.isoformat(),
            "needs_grading": submission.status == "submitted"
        })
    
    # Get upcoming classes (from schedule)
    upcoming_classes_result = await db.execute(
        select(Schedule, Class)
        .join(Class)
        .where(
            Class.teacher_id == current_user.id,
            Schedule.start_time > datetime.utcnow(),
            Schedule.status == "active"
        )
        .order_by(Schedule.start_time)
        .limit(5)
    )
    
    upcoming_classes = []
    for schedule, class_obj in upcoming_classes_result:
        upcoming_classes.append({
            "id": str(schedule.id),
            "class_name": class_obj.name,
            "start_time": schedule.start_time.isoformat(),
            "end_time": schedule.end_time.isoformat(),
            "event_type": schedule.event_type.value
        })
    
    # Get recent notifications
    notifications_result = await db.execute(
        select(Notification)
        .where(
            Notification.user_id == current_user.id,
            Notification.is_archived == False
        )
        .order_by(desc(Notification.created_at))
        .limit(5)
    )
    notifications = [
        {
            "id": str(notif.id),
            "title": notif.title,
            "message": notif.message,
            "type": notif.notification_type.value,
            "is_read": notif.is_read,
            "created_at": notif.created_at.isoformat()
        }
        for notif in notifications_result.scalars()
    ]
    
    # Get quick statistics
    quick_stats = await calculate_teacher_statistics(profile, db)
    
    return TeacherDashboardResponse(
        profile=TeacherProfileResponse(
            id=str(profile.id),
            user_id=str(profile.user_id),
            employee_id=profile.employee_id,
            department=profile.department,
            subjects=profile.subjects,
            qualifications=profile.qualifications,
            experience_years=profile.experience_years,
            hire_date=profile.hire_date,
            office_location=profile.office_location,
            office_hours=profile.office_hours,
            teaching_load=profile.teaching_load,
            specializations=profile.specializations,
            certifications=profile.certifications,
            performance_rating=profile.performance_rating
        ),
        classes_taught=classes_taught,
        recent_submissions=recent_submissions,
        pending_grades=quick_stats["pending_grades"],
        total_students=quick_stats["total_students"],
        average_class_performance=quick_stats["average_class_performance"],
        upcoming_classes=upcoming_classes,
        notifications=notifications,
        quick_stats=quick_stats
    )


async def get_class_enrollment_count(class_id: str, db: AsyncSession) -> int:
    """Get number of enrolled students in a class"""
    result = await db.execute(
        select(func.count(Enrollment.id))
        .where(
            Enrollment.class_id == class_id,
            Enrollment.status == "active"
        )
    )
    return result.scalar() or 0


@router.get("/classes")
async def get_teacher_classes(
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db),
    active_only: bool = Query(True, description="Filter for active classes only")
):
    """Get list of classes taught by teacher"""
    
    query = select(Class).where(Class.teacher_id == current_user.id)
    
    if active_only:
        query = query.where(Class.is_active == True)
    
    result = await db.execute(query.order_by(Class.name))
    
    classes = []
    for class_obj in result.scalars():
        enrollment_count = await get_class_enrollment_count(class_obj.id, db)
        
        classes.append({
            "id": str(class_obj.id),
            "name": class_obj.name,
            "code": class_obj.code,
            "description": class_obj.description,
            "subject": class_obj.subject,
            "grade_level": class_obj.grade_level,
            "section": class_obj.section,
            "max_students": class_obj.max_students,
            "enrolled_count": enrollment_count,
            "credits": class_obj.credits,
            "academic_year": class_obj.academic_year,
            "semester": class_obj.semester,
            "is_active": class_obj.is_active,
            "created_at": class_obj.created_at.isoformat()
        })
    
    return {"classes": classes}


@router.post("/classes")
async def create_class(
    request: ClassCreationRequest,
    current_user: User = Depends(require_teacher),
    school = Depends(get_current_school),
    db: AsyncSession = Depends(get_db)
):
    """Create a new class"""
    
    # Check if class code already exists in school
    existing_class = await db.execute(
        select(Class)
        .where(
            Class.school_id == school.id,
            Class.code == request.code,
            Class.academic_year == request.academic_year,
            Class.is_deleted == False
        )
    )
    
    if existing_class.scalar_one_or_none():
        raise ConflictException(
            f"Class with code '{request.code}' already exists for academic year {request.academic_year}"
        )
    
    # Create new class
    import uuid
    
    new_class = Class(
        id=uuid.uuid4(),
        name=request.name,
        code=request.code,
        description=request.description,
        subject=request.subject,
        grade_level=request.grade_level,
        section=request.section,
        max_students=request.max_students,
        credits=request.credits,
        academic_year=request.academic_year,
        semester=request.semester,
        meeting_schedule=request.meeting_schedule or {},
        grading_scheme=request.grading_scheme or {
            "A": {"min": 90, "max": 100},
            "B": {"min": 80, "max": 89},
            "C": {"min": 70, "max": 79},
            "D": {"min": 60, "max": 69},
            "F": {"min": 0, "max": 59}
        },
        school_id=school.id,
        teacher_id=current_user.id,
        is_active=True
    )
    
    db.add(new_class)
    await db.commit()
    
    logger.info(f"Class {request.code} created by teacher {current_user.username}")
    
    return {
        "message": "Class created successfully",
        "class_id": str(new_class.id),
        "class_code": new_class.code
    }


@router.put("/classes/{class_id}")
async def update_class(
    class_id: str,
    request: ClassUpdateRequest,
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db)
):
    """Update class information"""
    
    # Check if teacher owns the class
    can_modify = await PermissionChecker.can_modify_quiz(current_user, class_id, db)
    if not can_modify and current_user.role != UserRole.ADMIN:
        raise AuthorizationException("You don't have permission to modify this class")
    
    # Get class
    result = await db.execute(
        select(Class).where(
            Class.id == class_id,
            Class.is_deleted == False
        )
    )
    class_obj = result.scalar_one_or_none()
    
    if not class_obj:
        raise NotFoundException("Class not found")
    
    # Update fields
    if request.name is not None:
        class_obj.name = request.name
    if request.description is not None:
        class_obj.description = request.description
    if request.max_students is not None:
        class_obj.max_students = request.max_students
    if request.meeting_schedule is not None:
        class_obj.meeting_schedule = request.meeting_schedule
    if request.grading_scheme is not None:
        class_obj.grading_scheme = request.grading_scheme
    if request.is_active is not None:
        class_obj.is_active = request.is_active
    
    class_obj.updated_at = datetime.utcnow()
    await db.commit()
    
    logger.info(f"Class {class_obj.code} updated by teacher {current_user.username}")
    
    return {"message": "Class updated successfully"}


@router.get("/classes/{class_id}/students")
async def get_class_students(
    class_id: str,
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db)
):
    """Get students enrolled in a specific class"""
    
    # Check access
    can_access = await PermissionChecker.can_access_class(current_user, class_id, db)
    if not can_access:
        raise AuthorizationException("Access denied to this class")
    
    result = await db.execute(
        select(User, StudentProfile, Enrollment)
        .join(StudentProfile)
        .join(Enrollment)
        .where(
            Enrollment.class_id == class_id,
            Enrollment.status == "active",
            User.is_deleted == False
        )
        .order_by(User.last_name, User.first_name)
    )
    
    students = []
    for user, profile, enrollment in result:
        # Get recent performance
        recent_performance = await get_student_recent_performance(user.id, class_id, db)
        
        students.append({
            "id": str(user.id),
            "name": f"{user.first_name} {user.last_name}",
            "email": user.email,
            "student_id": profile.student_id,
            "grade_level": profile.grade_level,
            "enrollment_date": enrollment.enrollment_date.isoformat(),
            "recent_performance": recent_performance,
            "total_xp": profile.total_xp,
            "level": profile.level,
            "last_activity": profile.last_activity.isoformat() if profile.last_activity else None
        })
    
    return {"students": students}


async def get_student_recent_performance(
    student_id: str, 
    class_id: str, 
    db: AsyncSession
) -> Dict[str, Any]:
    """Get student's recent performance in a class"""
    
    result = await db.execute(
        select(
            func.avg(QuizSubmission.percentage).label('avg_score'),
            func.count(QuizSubmission.id).label('quiz_count'),
            func.max(QuizSubmission.submitted_at).label('last_submission')
        )
        .join(Quiz)
        .where(
            Quiz.class_id == class_id,
            QuizSubmission.student_id == student_id,
            QuizSubmission.status == "submitted"
        )
    )
    
    row = result.first()
    if row and row.quiz_count > 0:
        return {
            "average_score": round(row.avg_score, 2),
            "quiz_count": row.quiz_count,
            "last_submission": row.last_submission.isoformat() if row.last_submission else None
        }
    else:
        return {
            "average_score": 0,
            "quiz_count": 0,
            "last_submission": None
        }


@router.post("/classes/{class_id}/enroll")
async def enroll_students(
    class_id: str,
    request: StudentEnrollmentRequest,
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db)
):
    """Enroll students in a class"""
    
    # Check access
    can_access = await PermissionChecker.can_access_class(current_user, class_id, db)
    if not can_access:
        raise AuthorizationException("Access denied to this class")
    
    # Get class
    result = await db.execute(
        select(Class).where(Class.id == class_id, Class.is_deleted == False)
    )
    class_obj = result.scalar_one_or_none()
    
    if not class_obj:
        raise NotFoundException("Class not found")
    
    # Check class capacity
    current_enrollment = await get_class_enrollment_count(class_id, db)
    if current_enrollment + len(request.student_ids) > class_obj.max_students:
        raise BusinessLogicException(
            f"Class capacity exceeded. Available slots: {class_obj.max_students - current_enrollment}"
        )
    
    # Enroll students
    import uuid
    enrolled_students = []
    enrollment_date = request.enrollment_date or date.today()
    
    for student_id in request.student_ids:
        # Check if student exists and is a student
        student_result = await db.execute(
            select(User).where(
                User.id == student_id,
                User.role == UserRole.STUDENT,
                User.is_deleted == False
            )
        )
        student = student_result.scalar_one_or_none()
        
        if not student:
            logger.warning(f"Student {student_id} not found or invalid")
            continue
        
        # Check if already enrolled
        existing_enrollment = await db.execute(
            select(Enrollment).where(
                Enrollment.student_id == student_id,
                Enrollment.class_id == class_id
            )
        )
        
        if existing_enrollment.scalar_one_or_none():
            logger.warning(f"Student {student_id} already enrolled in class {class_id}")
            continue
        
        # Create enrollment
        enrollment = Enrollment(
            id=uuid.uuid4(),
            student_id=student_id,
            class_id=class_id,
            enrollment_date=enrollment_date,
            status="active"
        )
        
        db.add(enrollment)
        enrolled_students.append(student.username)
    
    await db.commit()
    
    logger.info(f"{len(enrolled_students)} students enrolled in class {class_obj.code}")
    
    return {
        "message": f"Successfully enrolled {len(enrolled_students)} students",
        "enrolled_students": enrolled_students
    }


@router.get("/analytics", response_model=TeacherAnalyticsResponse)
async def get_teacher_analytics(
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=7, le=365, description="Number of days for analytics")
):
    """Get comprehensive teacher analytics"""
    
    from datetime import timedelta
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get class performance
    class_performance_result = await db.execute(
        select(
            Class.name,
            Class.code,
            func.avg(QuizSubmission.percentage).label('avg_score'),
            func.count(QuizSubmission.id).label('total_submissions'),
            func.count(func.distinct(QuizSubmission.student_id)).label('active_students')
        )
        .select_from(Class)
        .join(Quiz)
        .join(QuizSubmission)
        .where(
            Class.teacher_id == current_user.id,
            QuizSubmission.submitted_at >= start_date,
            QuizSubmission.status == "submitted"
        )
        .group_by(Class.id, Class.name, Class.code)
        .order_by(func.avg(QuizSubmission.percentage).desc())
    )
    
    class_performance = []
    for row in class_performance_result:
        class_performance.append({
            "class_name": row.name,
            "class_code": row.code,
            "average_score": round(row.avg_score, 2),
            "total_submissions": row.total_submissions,
            "active_students": row.active_students
        })
    
    # Get at-risk students
    teacher_profile = await get_teacher_profile(str(current_user.id), db, current_user)
    at_risk_students = await get_at_risk_students(teacher_profile, db)
    
    # TODO: Implement more analytics
    return TeacherAnalyticsResponse(
        class_performance=class_performance,
        student_engagement={
            "total_active_students": sum(cp["active_students"] for cp in class_performance),
            "engagement_trend": "stable"  # TODO: Calculate actual trend
        },
        quiz_statistics={
            "total_quizzes": len(class_performance),
            "average_score": sum(cp["average_score"] for cp in class_performance) / len(class_performance) if class_performance else 0
        },
        grade_distribution={
            "A": 0,  # TODO: Calculate actual distribution
            "B": 0,
            "C": 0,
            "D": 0,
            "F": 0
        },
        improvement_trends=[],  # TODO: Implement
        at_risk_students=at_risk_students,
        teaching_effectiveness={
            "overall_rating": teacher_profile.performance_rating,
            "student_satisfaction": 0.0  # TODO: Implement surveys
        }
    )


@router.post("/bulk-grade")
async def submit_bulk_grades(
    request: BulkGradeRequest,
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db)
):
    """Submit grades for multiple students"""
    
    import uuid
    grades_created = []
    
    for grade_data in request.grades:
        student_id = grade_data["student_id"]
        grade_value = grade_data["grade"]
        feedback = grade_data.get("feedback", "")
        
        # Verify student exists and teacher has access
        can_access = await PermissionChecker.can_access_user_data(
            current_user, student_id, db
        )
        
        if not can_access:
            logger.warning(f"Teacher {current_user.id} attempted to grade student {student_id} without permission")
            continue
        
        # Create grade record (this is simplified - in real implementation you'd tie to specific assignments)
        grade = Grade(
            id=uuid.uuid4(),
            student_id=student_id,
            class_id=grade_data.get("class_id"),  # Should be provided in request
            grade_type="manual",
            title="Manual Grade Entry",
            points_earned=grade_value,
            points_possible=100,
            percentage=grade_value,
            letter_grade=calculate_letter_grade(grade_value),
            graded_at=datetime.utcnow(),
            graded_by_id=current_user.id,
            feedback=feedback
        )
        
        db.add(grade)
        grades_created.append(student_id)
    
    await db.commit()
    
    logger.info(f"Bulk grades submitted by teacher {current_user.username} for {len(grades_created)} students")
    
    return {
        "message": f"Successfully submitted {len(grades_created)} grades",
        "graded_students": len(grades_created)
    }


def calculate_letter_grade(percentage: float) -> str:
    """Convert percentage to letter grade"""
    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F"


# Export router
__all__ = ["router"]
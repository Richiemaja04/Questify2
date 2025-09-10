"""
AI-Powered Smart Class & Timetable Scheduler
Student-related API routes
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Query, Path
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from sqlalchemy.orm import selectinload

from ..database.connection import get_db
from ..database.models import (
    User, UserRole, StudentProfile, Enrollment, Class, 
    Quiz, QuizSubmission, Grade, XPTransaction, 
    UserAchievement, Achievement, AnalyticsEvent,
    BehavioralProfile, Notification
)
from ..dependencies import (
    require_authentication,
    require_student,
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
    BusinessLogicException
)

# Configure logging
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()


# Pydantic models
class StudentProfileResponse(BaseModel):
    id: str
    user_id: str
    student_id: str
    admission_date: Optional[date]
    grade_level: Optional[str]
    section: Optional[str]
    guardian_name: Optional[str]
    guardian_email: Optional[str]
    guardian_phone: Optional[str]
    learning_style: Optional[str]
    academic_interests: List[str]
    goals: List[str]
    accessibility_needs: Dict[str, Any]
    total_xp: int
    level: int
    streak_days: int
    last_activity: Optional[datetime]
    performance_metrics: Dict[str, Any]
    xp_to_next_level: int
    
    class Config:
        from_attributes = True


class StudentDashboardResponse(BaseModel):
    profile: StudentProfileResponse
    enrolled_classes: List[Dict[str, Any]]
    recent_quizzes: List[Dict[str, Any]]
    upcoming_assignments: List[Dict[str, Any]]
    achievements: List[Dict[str, Any]]
    progress_summary: Dict[str, Any]
    notifications: List[Dict[str, Any]]
    leaderboard_position: Optional[int]


class UpdateStudentProfileRequest(BaseModel):
    guardian_name: Optional[str] = None
    guardian_email: Optional[str] = None
    guardian_phone: Optional[str] = None
    learning_style: Optional[str] = None
    academic_interests: Optional[List[str]] = None
    goals: Optional[List[str]] = None
    accessibility_needs: Optional[Dict[str, Any]] = None
    
    @validator('learning_style')
    def validate_learning_style(cls, v):
        if v and v not in ['visual', 'auditory', 'kinesthetic', 'reading', 'mixed']:
            raise ValueError('Invalid learning style')
        return v


class StudentProgressResponse(BaseModel):
    total_xp: int
    level: int
    xp_to_next_level: int
    streak_days: int
    quizzes_completed: int
    perfect_scores: int
    average_score: float
    improvement_percentage: float
    time_spent_minutes: int
    achievements_unlocked: int
    current_rank: Optional[int]
    progress_chart: List[Dict[str, Any]]


class StudentPerformanceRequest(BaseModel):
    subject: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    metric: str = "score"  # score, time, attempts
    
    @validator('metric')
    def validate_metric(cls, v):
        if v not in ['score', 'time', 'attempts', 'improvement']:
            raise ValueError('Invalid metric')
        return v


class LearningGoalRequest(BaseModel):
    title: str
    description: Optional[str] = None
    target_value: float
    target_date: date
    metric: str  # xp, quiz_score, streak_days
    
    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title is required')
        return v.strip()
    
    @validator('target_value')
    def validate_target_value(cls, v):
        if v <= 0:
            raise ValueError('Target value must be positive')
        return v


class StudySessionRequest(BaseModel):
    duration_minutes: int
    subject: Optional[str] = None
    topics: Optional[List[str]] = None
    notes: Optional[str] = None
    
    @validator('duration_minutes')
    def validate_duration(cls, v):
        if v < 1 or v > 480:  # Max 8 hours
            raise ValueError('Duration must be between 1 and 480 minutes')
        return v


# Helper functions
async def get_student_profile(
    student_id: str,
    db: AsyncSession,
    current_user: User
) -> StudentProfile:
    """Get student profile with permission checking"""
    
    # Check if user can access this student's data
    can_access = await PermissionChecker.can_access_user_data(
        current_user, student_id, db
    )
    
    if not can_access:
        raise AuthorizationException("Access denied to student data")
    
    result = await db.execute(
        select(StudentProfile)
        .options(selectinload(StudentProfile.user))
        .where(
            StudentProfile.user_id == student_id,
            StudentProfile.is_deleted == False
        )
    )
    
    profile = result.scalar_one_or_none()
    if not profile:
        raise NotFoundException("Student profile not found")
    
    return profile


async def calculate_student_progress(
    student_profile: StudentProfile,
    db: AsyncSession
) -> Dict[str, Any]:
    """Calculate student progress metrics"""
    
    # Get quiz submissions count
    quiz_count_result = await db.execute(
        select(func.count(QuizSubmission.id))
        .where(
            QuizSubmission.student_id == student_profile.user_id,
            QuizSubmission.status == "submitted"
        )
    )
    quizzes_completed = quiz_count_result.scalar() or 0
    
    # Get perfect scores count
    perfect_scores_result = await db.execute(
        select(func.count(QuizSubmission.id))
        .where(
            QuizSubmission.student_id == student_profile.user_id,
            QuizSubmission.percentage >= 100,
            QuizSubmission.status == "submitted"
        )
    )
    perfect_scores = perfect_scores_result.scalar() or 0
    
    # Get average score
    avg_score_result = await db.execute(
        select(func.avg(QuizSubmission.percentage))
        .where(
            QuizSubmission.student_id == student_profile.user_id,
            QuizSubmission.status == "submitted"
        )
    )
    average_score = float(avg_score_result.scalar() or 0)
    
    # Get recent performance for improvement calculation
    recent_scores_result = await db.execute(
        select(QuizSubmission.percentage)
        .where(
            QuizSubmission.student_id == student_profile.user_id,
            QuizSubmission.status == "submitted"
        )
        .order_by(desc(QuizSubmission.submitted_at))
        .limit(10)
    )
    recent_scores = [score for score in recent_scores_result.scalars()]
    
    # Calculate improvement percentage
    improvement_percentage = 0.0
    if len(recent_scores) >= 5:
        recent_avg = sum(recent_scores[:5]) / 5
        older_avg = sum(recent_scores[5:]) / len(recent_scores[5:])
        if older_avg > 0:
            improvement_percentage = ((recent_avg - older_avg) / older_avg) * 100
    
    # Get achievements count
    achievements_result = await db.execute(
        select(func.count(UserAchievement.id))
        .where(UserAchievement.user_id == student_profile.user_id)
    )
    achievements_unlocked = achievements_result.scalar() or 0
    
    # Get time spent (from analytics events)
    time_spent_result = await db.execute(
        select(func.sum(AnalyticsEvent.duration))
        .where(
            AnalyticsEvent.user_id == student_profile.user_id,
            AnalyticsEvent.event_type == "study_session"
        )
    )
    time_spent_seconds = time_spent_result.scalar() or 0
    time_spent_minutes = int(time_spent_seconds / 60)
    
    return {
        "total_xp": student_profile.total_xp,
        "level": student_profile.level,
        "xp_to_next_level": student_profile.xp_to_next_level,
        "streak_days": student_profile.streak_days,
        "quizzes_completed": quizzes_completed,
        "perfect_scores": perfect_scores,
        "average_score": round(average_score, 2),
        "improvement_percentage": round(improvement_percentage, 2),
        "time_spent_minutes": time_spent_minutes,
        "achievements_unlocked": achievements_unlocked
    }


async def get_student_leaderboard_position(
    student_profile: StudentProfile,
    db: AsyncSession
) -> Optional[int]:
    """Get student's position in school leaderboard"""
    
    result = await db.execute(
        select(func.count(StudentProfile.id))
        .join(User)
        .where(
            User.school_id == student_profile.user.school_id,
            StudentProfile.total_xp > student_profile.total_xp,
            User.is_deleted == False
        )
    )
    
    higher_ranked_count = result.scalar() or 0
    return higher_ranked_count + 1


# API Routes
@router.get("/profile", response_model=StudentProfileResponse)
async def get_my_profile(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Get current student's profile"""
    
    profile = await get_student_profile(str(current_user.id), db, current_user)
    
    return StudentProfileResponse(
        id=str(profile.id),
        user_id=str(profile.user_id),
        student_id=profile.student_id,
        admission_date=profile.admission_date,
        grade_level=profile.grade_level,
        section=profile.section,
        guardian_name=profile.guardian_name,
        guardian_email=profile.guardian_email,
        guardian_phone=profile.guardian_phone,
        learning_style=profile.learning_style,
        academic_interests=profile.academic_interests,
        goals=profile.goals,
        accessibility_needs=profile.accessibility_needs,
        total_xp=profile.total_xp,
        level=profile.level,
        streak_days=profile.streak_days,
        last_activity=profile.last_activity,
        performance_metrics=profile.performance_metrics,
        xp_to_next_level=profile.xp_to_next_level
    )


@router.put("/profile")
async def update_my_profile(
    request: UpdateStudentProfileRequest,
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Update current student's profile"""
    
    profile = await get_student_profile(str(current_user.id), db, current_user)
    
    # Update fields
    if request.guardian_name is not None:
        profile.guardian_name = request.guardian_name
    if request.guardian_email is not None:
        profile.guardian_email = request.guardian_email
    if request.guardian_phone is not None:
        profile.guardian_phone = request.guardian_phone
    if request.learning_style is not None:
        profile.learning_style = request.learning_style
    if request.academic_interests is not None:
        profile.academic_interests = request.academic_interests
    if request.goals is not None:
        profile.goals = request.goals
    if request.accessibility_needs is not None:
        profile.accessibility_needs = request.accessibility_needs
    
    profile.updated_at = datetime.utcnow()
    await db.commit()
    
    logger.info(f"Profile updated for student {profile.student_id}")
    
    return {"message": "Profile updated successfully"}


@router.get("/dashboard", response_model=StudentDashboardResponse)
async def get_student_dashboard(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db),
    cache: bool = Depends(standard_cache.get)
):
    """Get comprehensive student dashboard data"""
    
    # Check cache first
    cache_key = f"student_dashboard:{current_user.id}"
    cached_data = await standard_cache.get(cache_key)
    if cached_data:
        return StudentDashboardResponse(**cached_data)
    
    profile = await get_student_profile(str(current_user.id), db, current_user)
    
    # Get enrolled classes
    classes_result = await db.execute(
        select(Class)
        .join(Enrollment)
        .where(
            Enrollment.student_id == current_user.id,
            Enrollment.status == "active",
            Class.is_active == True
        )
    )
    enrolled_classes = [
        {
            "id": str(cls.id),
            "name": cls.name,
            "code": cls.code,
            "subject": cls.subject,
            "grade_level": cls.grade_level,
            "section": cls.section
        }
        for cls in classes_result.scalars()
    ]
    
    # Get recent quizzes
    recent_quizzes_result = await db.execute(
        select(Quiz)
        .join(Class)
        .join(Enrollment)
        .where(
            Enrollment.student_id == current_user.id,
            Quiz.status == "published",
            Quiz.available_from <= datetime.utcnow()
        )
        .order_by(desc(Quiz.created_at))
        .limit(5)
    )
    recent_quizzes = [
        {
            "id": str(quiz.id),
            "title": quiz.title,
            "class_name": quiz.class_obj.name,
            "due_date": quiz.available_until.isoformat() if quiz.available_until else None,
            "status": "completed" if await check_quiz_completed(quiz.id, current_user.id, db) else "available"
        }
        for quiz in recent_quizzes_result.scalars()
    ]
    
    # Get recent achievements
    achievements_result = await db.execute(
        select(UserAchievement, Achievement)
        .join(Achievement)
        .where(UserAchievement.user_id == current_user.id)
        .order_by(desc(UserAchievement.unlocked_at))
        .limit(5)
    )
    achievements = [
        {
            "id": str(achievement.id),
            "name": achievement.name,
            "description": achievement.description,
            "icon_url": achievement.icon_url,
            "unlocked_at": user_achievement.unlocked_at.isoformat(),
            "points_reward": achievement.points_reward
        }
        for user_achievement, achievement in achievements_result
    ]
    
    # Get progress summary
    progress_summary = await calculate_student_progress(profile, db)
    
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
            "created_at": notif.created_at.isoformat(),
            "action_url": notif.action_url
        }
        for notif in notifications_result.scalars()
    ]
    
    # Get leaderboard position
    leaderboard_position = await get_student_leaderboard_position(profile, db)
    
    dashboard_data = {
        "profile": StudentProfileResponse(
            id=str(profile.id),
            user_id=str(profile.user_id),
            student_id=profile.student_id,
            admission_date=profile.admission_date,
            grade_level=profile.grade_level,
            section=profile.section,
            guardian_name=profile.guardian_name,
            guardian_email=profile.guardian_email,
            guardian_phone=profile.guardian_phone,
            learning_style=profile.learning_style,
            academic_interests=profile.academic_interests,
            goals=profile.goals,
            accessibility_needs=profile.accessibility_needs,
            total_xp=profile.total_xp,
            level=profile.level,
            streak_days=profile.streak_days,
            last_activity=profile.last_activity,
            performance_metrics=profile.performance_metrics,
            xp_to_next_level=profile.xp_to_next_level
        ),
        "enrolled_classes": enrolled_classes,
        "recent_quizzes": recent_quizzes,
        "upcoming_assignments": [],  # TODO: Implement assignments
        "achievements": achievements,
        "progress_summary": progress_summary,
        "notifications": notifications,
        "leaderboard_position": leaderboard_position
    }
    
    # Cache the data
    await standard_cache.set(cache_key, dashboard_data)
    
    return StudentDashboardResponse(**dashboard_data)


async def check_quiz_completed(quiz_id: str, student_id: str, db: AsyncSession) -> bool:
    """Check if student has completed a quiz"""
    result = await db.execute(
        select(func.count(QuizSubmission.id))
        .where(
            QuizSubmission.quiz_id == quiz_id,
            QuizSubmission.student_id == student_id,
            QuizSubmission.status.in_(["submitted", "graded"])
        )
    )
    return result.scalar() > 0


@router.get("/progress", response_model=StudentProgressResponse)
async def get_student_progress(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=7, le=365, description="Number of days for progress chart")
):
    """Get detailed student progress and analytics"""
    
    profile = await get_student_profile(str(current_user.id), db, current_user)
    progress_data = await calculate_student_progress(profile, db)
    
    # Get leaderboard position
    current_rank = await get_student_leaderboard_position(profile, db)
    
    # Get progress chart data (XP over time)
    from datetime import timedelta
    start_date = datetime.utcnow() - timedelta(days=days)
    
    progress_chart_result = await db.execute(
        select(
            func.date(XPTransaction.transaction_date).label('date'),
            func.sum(XPTransaction.amount).label('xp_gained')
        )
        .where(
            XPTransaction.user_id == current_user.id,
            XPTransaction.transaction_date >= start_date
        )
        .group_by(func.date(XPTransaction.transaction_date))
        .order_by('date')
    )
    
    progress_chart = []
    cumulative_xp = 0
    for row in progress_chart_result:
        cumulative_xp += row.xp_gained
        progress_chart.append({
            "date": row.date.isoformat(),
            "xp_gained": row.xp_gained,
            "cumulative_xp": cumulative_xp
        })
    
    return StudentProgressResponse(
        **progress_data,
        current_rank=current_rank,
        progress_chart=progress_chart
    )


@router.get("/classes")
async def get_enrolled_classes(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Get list of classes student is enrolled in"""
    
    result = await db.execute(
        select(Class, Enrollment)
        .join(Enrollment)
        .where(
            Enrollment.student_id == current_user.id,
            Class.is_active == True
        )
        .order_by(Class.name)
    )
    
    classes = []
    for class_obj, enrollment in result:
        classes.append({
            "id": str(class_obj.id),
            "name": class_obj.name,
            "code": class_obj.code,
            "subject": class_obj.subject,
            "grade_level": class_obj.grade_level,
            "section": class_obj.section,
            "teacher_name": f"{class_obj.teacher.user.first_name} {class_obj.teacher.user.last_name}",
            "enrollment_date": enrollment.enrollment_date.isoformat(),
            "status": enrollment.status
        })
    
    return {"classes": classes}


@router.get("/achievements")
async def get_student_achievements(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Get all student achievements"""
    
    # Get unlocked achievements
    unlocked_result = await db.execute(
        select(UserAchievement, Achievement)
        .join(Achievement)
        .where(UserAchievement.user_id == current_user.id)
        .order_by(desc(UserAchievement.unlocked_at))
    )
    
    unlocked_achievements = []
    unlocked_ids = []
    for user_achievement, achievement in unlocked_result:
        unlocked_ids.append(str(achievement.id))
        unlocked_achievements.append({
            "id": str(achievement.id),
            "name": achievement.name,
            "description": achievement.description,
            "category": achievement.category.value,
            "icon_url": achievement.icon_url,
            "points_reward": achievement.points_reward,
            "rarity": achievement.rarity,
            "unlocked_at": user_achievement.unlocked_at.isoformat(),
            "current_level": user_achievement.current_level,
            "progress": user_achievement.progress
        })
    
    # Get available achievements (not yet unlocked)
    available_result = await db.execute(
        select(Achievement)
        .where(
            Achievement.is_active == True,
            Achievement.is_hidden == False,
            ~Achievement.id.in_(unlocked_ids) if unlocked_ids else True
        )
        .order_by(Achievement.points_reward)
    )
    
    available_achievements = []
    for achievement in available_result.scalars():
        available_achievements.append({
            "id": str(achievement.id),
            "name": achievement.name,
            "description": achievement.description,
            "category": achievement.category.value,
            "icon_url": achievement.icon_url,
            "points_reward": achievement.points_reward,
            "rarity": achievement.rarity,
            "unlock_criteria": achievement.unlock_criteria
        })
    
    return {
        "unlocked": unlocked_achievements,
        "available": available_achievements,
        "total_unlocked": len(unlocked_achievements),
        "total_available": len(available_achievements)
    }


@router.post("/goals")
async def create_learning_goal(
    request: LearningGoalRequest,
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Create a new learning goal"""
    
    profile = await get_student_profile(str(current_user.id), db, current_user)
    
    # Add goal to profile
    new_goal = {
        "id": str(uuid.uuid4()),
        "title": request.title,
        "description": request.description,
        "target_value": request.target_value,
        "target_date": request.target_date.isoformat(),
        "metric": request.metric,
        "created_at": datetime.utcnow().isoformat(),
        "status": "active",
        "progress": 0.0
    }
    
    goals = profile.goals or []
    goals.append(new_goal)
    profile.goals = goals
    profile.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Learning goal created successfully", "goal": new_goal}


@router.post("/study-session")
async def log_study_session(
    request: StudySessionRequest,
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Log a study session"""
    
    import uuid
    
    # Create analytics event
    analytics_event = AnalyticsEvent(
        id=uuid.uuid4(),
        user_id=current_user.id,
        session_id=f"study_{int(datetime.utcnow().timestamp())}",
        event_type="study_session",
        event_name="manual_study_log",
        duration=request.duration_minutes * 60,  # Convert to seconds
        properties={
            "subject": request.subject,
            "topics": request.topics or [],
            "notes": request.notes,
            "duration_minutes": request.duration_minutes
        }
    )
    
    db.add(analytics_event)
    
    # Award XP for study session
    xp_amount = max(1, request.duration_minutes // 10)  # 1 XP per 10 minutes
    
    xp_transaction = XPTransaction(
        id=uuid.uuid4(),
        user_id=current_user.id,
        amount=xp_amount,
        source_type="study_session",
        source_id=str(analytics_event.id),
        description=f"Study session ({request.duration_minutes} minutes)",
        multiplier=1.0
    )
    
    db.add(xp_transaction)
    
    # Update student profile
    profile = await get_student_profile(str(current_user.id), db, current_user)
    profile.total_xp += xp_amount
    profile.last_activity = datetime.utcnow()
    
    await db.commit()
    
    return {
        "message": "Study session logged successfully",
        "xp_earned": xp_amount,
        "duration_minutes": request.duration_minutes
    }


# Teacher/Admin endpoints for student management
@router.get("/{student_id}/profile", dependencies=[Depends(require_teacher_or_admin)])
async def get_student_profile_by_id(
    student_id: str = Path(..., description="Student user ID"),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get student profile by ID (teachers/admins only)"""
    
    profile = await get_student_profile(student_id, db, current_user)
    
    return StudentProfileResponse(
        id=str(profile.id),
        user_id=str(profile.user_id),
        student_id=profile.student_id,
        admission_date=profile.admission_date,
        grade_level=profile.grade_level,
        section=profile.section,
        guardian_name=profile.guardian_name,
        guardian_email=profile.guardian_email,
        guardian_phone=profile.guardian_phone,
        learning_style=profile.learning_style,
        academic_interests=profile.academic_interests,
        goals=profile.goals,
        accessibility_needs=profile.accessibility_needs,
        total_xp=profile.total_xp,
        level=profile.level,
        streak_days=profile.streak_days,
        last_activity=profile.last_activity,
        performance_metrics=profile.performance_metrics,
        xp_to_next_level=profile.xp_to_next_level
    )


@router.get("/{student_id}/performance", dependencies=[Depends(require_teacher_or_admin)])
async def get_student_performance_by_id(
    student_id: str = Path(..., description="Student user ID"),
    request: StudentPerformanceRequest = Depends(),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get student performance analytics (teachers/admins only)"""
    
    # Check permissions
    can_access = await PermissionChecker.can_access_user_data(
        current_user, student_id, db
    )
    if not can_access:
        raise AuthorizationException("Access denied to student data")
    
    # TODO: Implement detailed performance analytics
    return {"message": "Performance analytics coming soon"}


# Export router
__all__ = ["router"]
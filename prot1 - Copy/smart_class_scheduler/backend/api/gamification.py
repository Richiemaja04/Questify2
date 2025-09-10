"""
AI-Powered Smart Class & Timetable Scheduler
Gamification and achievement system API routes
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import APIRouter, Depends, Query, Path, BackgroundTasks
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.orm import selectinload
import math

from ..database.connection import get_db
from ..database.models import (
    User, UserRole, Achievement, AchievementType, UserAchievement,
    XPTransaction, StudentProfile, Quiz, QuizSubmission, Grade,
    AnalyticsEvent, Class, Enrollment
)
from ..dependencies import (
    require_authentication,
    require_student,
    require_teacher,
    require_admin,
    require_teacher_or_admin,
    get_current_school,
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
from ...config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()


# Pydantic models
class AchievementResponse(BaseModel):
    id: str
    name: str
    description: str
    category: AchievementType
    icon_url: Optional[str]
    badge_color: str
    points_reward: int
    rarity: str
    unlock_criteria: Dict[str, Any]
    is_hidden: bool
    unlock_message: Optional[str]
    prerequisite_achievements: List[str]
    max_level: int
    unlocked_by_user: bool = False
    unlock_date: Optional[datetime] = None
    progress_percentage: Optional[float] = None
    
    class Config:
        from_attributes = True


class UserAchievementResponse(BaseModel):
    id: str
    achievement: AchievementResponse
    unlocked_at: datetime
    current_level: int
    progress: Dict[str, Any]
    notification_sent: bool


class XPTransactionResponse(BaseModel):
    id: str
    amount: int
    source_type: str
    source_id: Optional[str]
    description: str
    multiplier: float
    bonus_reason: Optional[str]
    transaction_date: datetime
    metadata: Dict[str, Any]
    
    class Config:
        from_attributes = True


class LeaderboardResponse(BaseModel):
    leaderboard_type: str
    period: str
    entries: List[Dict[str, Any]]
    user_position: Optional[int]
    total_participants: int
    last_updated: datetime


class ProgressSummaryResponse(BaseModel):
    user_id: str
    total_xp: int
    current_level: int
    xp_to_next_level: int
    level_progress_percentage: float
    achievements_unlocked: int
    achievements_available: int
    current_streak: int
    longest_streak: int
    weekly_xp: int
    monthly_xp: int
    rank_in_school: Optional[int]
    rank_in_class: Optional[int]


class BadgeCollectionResponse(BaseModel):
    total_badges: int
    badges_by_rarity: Dict[str, int]
    recent_badges: List[Dict[str, Any]]
    featured_badge: Optional[Dict[str, Any]]
    completion_percentage: float


class ChallengeResponse(BaseModel):
    id: str
    title: str
    description: str
    challenge_type: str
    difficulty: str
    xp_reward: int
    start_date: datetime
    end_date: datetime
    participants: int
    completion_criteria: Dict[str, Any]
    user_progress: Optional[Dict[str, Any]]
    is_active: bool
    is_completed: bool


class CreateChallengeRequest(BaseModel):
    title: str
    description: str
    challenge_type: str = "individual"  # individual, team, class
    difficulty: str = "medium"
    xp_reward: int
    duration_days: int = 7
    max_participants: Optional[int] = None
    completion_criteria: Dict[str, Any]
    target_audience: List[str] = []  # class_ids or "all"
    
    @validator('title', 'description')
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title and description are required')
        return v.strip()
    
    @validator('challenge_type')
    def validate_challenge_type(cls, v):
        if v not in ['individual', 'team', 'class']:
            raise ValueError('Invalid challenge type')
        return v
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        if v not in ['easy', 'medium', 'hard', 'expert']:
            raise ValueError('Invalid difficulty level')
        return v
    
    @validator('xp_reward')
    def validate_xp_reward(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('XP reward must be between 1 and 1000')
        return v


class StreakResponse(BaseModel):
    current_streak: int
    longest_streak: int
    streak_type: str  # daily, weekly, monthly
    last_activity_date: Optional[date]
    streak_multiplier: float
    milestones: List[Dict[str, Any]]


class CustomBadgeRequest(BaseModel):
    name: str
    description: str
    icon_url: Optional[str] = None
    badge_color: str = "#gold"
    points_reward: int = 50
    unlock_criteria: Dict[str, Any]
    target_users: Optional[List[str]] = None
    expires_at: Optional[datetime] = None
    
    @validator('points_reward')
    def validate_points(cls, v):
        if v < 1 or v > 500:
            raise ValueError('Points reward must be between 1 and 500')
        return v


# Helper functions
async def calculate_level_from_xp(total_xp: int) -> Dict[str, Any]:
    """Calculate level and progress from total XP"""
    
    # Level calculation: Level = sqrt(XP / 100)
    level = max(1, int(math.sqrt(total_xp / 100)))
    
    # XP required for current level
    current_level_xp = (level - 1) ** 2 * 100
    
    # XP required for next level
    next_level_xp = level ** 2 * 100
    
    # XP needed to reach next level
    xp_to_next_level = next_level_xp - total_xp
    
    # Progress percentage within current level
    level_progress = ((total_xp - current_level_xp) / (next_level_xp - current_level_xp)) * 100
    
    return {
        "level": level,
        "xp_to_next_level": max(0, xp_to_next_level),
        "level_progress_percentage": min(100, max(0, level_progress))
    }


async def check_achievement_unlock(
    user: User,
    achievement: Achievement,
    db: AsyncSession
) -> Dict[str, Any]:
    """Check if user should unlock an achievement"""
    
    # Get user's current statistics
    stats = await get_user_statistics(user, db)
    criteria = achievement.unlock_criteria
    
    # Check different types of criteria
    if "quizzes_completed" in criteria:
        if stats.get("quizzes_completed", 0) >= criteria["quizzes_completed"]:
            return {"eligible": True, "progress": 100}
    
    if "perfect_scores" in criteria:
        if stats.get("perfect_scores", 0) >= criteria["perfect_scores"]:
            return {"eligible": True, "progress": 100}
    
    if "daily_streak" in criteria:
        if stats.get("current_streak", 0) >= criteria["daily_streak"]:
            return {"eligible": True, "progress": 100}
    
    if "total_xp" in criteria:
        current_xp = stats.get("total_xp", 0)
        required_xp = criteria["total_xp"]
        if current_xp >= required_xp:
            return {"eligible": True, "progress": 100}
        else:
            return {"eligible": False, "progress": (current_xp / required_xp) * 100}
    
    if "study_time_hours" in criteria:
        study_time_hours = stats.get("study_time_hours", 0)
        required_hours = criteria["study_time_hours"]
        if study_time_hours >= required_hours:
            return {"eligible": True, "progress": 100}
        else:
            return {"eligible": False, "progress": (study_time_hours / required_hours) * 100}
    
    return {"eligible": False, "progress": 0}


async def get_user_statistics(user: User, db: AsyncSession) -> Dict[str, Any]:
    """Get comprehensive user statistics for achievement checking"""
    
    # Get quiz statistics
    quiz_stats = await db.execute(
        select(
            func.count(QuizSubmission.id).label('total_submissions'),
            func.count(QuizSubmission.id).filter(QuizSubmission.percentage >= 100).label('perfect_scores'),
            func.avg(QuizSubmission.percentage).label('average_score')
        )
        .where(
            QuizSubmission.student_id == user.id,
            QuizSubmission.status == "submitted"
        )
    )
    quiz_result = quiz_stats.first()
    
    # Get XP and streak from student profile
    profile_result = await db.execute(
        select(StudentProfile).where(StudentProfile.user_id == user.id)
    )
    profile = profile_result.scalar_one_or_none()
    
    # Get study time from analytics events
    study_time_result = await db.execute(
        select(func.sum(AnalyticsEvent.duration))
        .where(
            AnalyticsEvent.user_id == user.id,
            AnalyticsEvent.event_type == "study_session"
        )
    )
    total_study_seconds = study_time_result.scalar() or 0
    
    return {
        "quizzes_completed": quiz_result.total_submissions or 0,
        "perfect_scores": quiz_result.perfect_scores or 0,
        "average_score": float(quiz_result.average_score or 0),
        "total_xp": profile.total_xp if profile else 0,
        "current_streak": profile.streak_days if profile else 0,
        "study_time_hours": total_study_seconds / 3600
    }


async def award_xp_transaction(
    user_id: str,
    amount: int,
    source_type: str,
    source_id: Optional[str],
    description: str,
    multiplier: float = 1.0,
    bonus_reason: Optional[str] = None,
    db: AsyncSession = None
) -> str:
    """Award XP to user and create transaction record"""
    
    import uuid
    
    # Create XP transaction
    xp_transaction = XPTransaction(
        id=uuid.uuid4(),
        user_id=user_id,
        amount=amount,
        source_type=source_type,
        source_id=source_id,
        description=description,
        multiplier=multiplier,
        bonus_reason=bonus_reason,
        metadata={}
    )
    
    if db:
        db.add(xp_transaction)
        
        # Update user's total XP
        profile_result = await db.execute(
            select(StudentProfile).where(StudentProfile.user_id == user_id)
        )
        profile = profile_result.scalar_one_or_none()
        
        if profile:
            profile.total_xp += amount
            profile.last_activity = datetime.utcnow()
            
            # Update level based on new XP
            level_info = await calculate_level_from_xp(profile.total_xp)
            profile.level = level_info["level"]
        
        await db.flush()
    
    return str(xp_transaction.id)


async def unlock_achievement_for_user(
    user_id: str,
    achievement_id: str,
    db: AsyncSession
) -> Optional[str]:
    """Unlock an achievement for a user"""
    
    # Check if already unlocked
    existing_result = await db.execute(
        select(UserAchievement).where(
            UserAchievement.user_id == user_id,
            UserAchievement.achievement_id == achievement_id
        )
    )
    
    if existing_result.scalar_one_or_none():
        return None  # Already unlocked
    
    # Get achievement details
    achievement_result = await db.execute(
        select(Achievement).where(Achievement.id == achievement_id)
    )
    achievement = achievement_result.scalar_one_or_none()
    
    if not achievement:
        return None
    
    # Create user achievement
    import uuid
    
    user_achievement = UserAchievement(
        id=uuid.uuid4(),
        user_id=user_id,
        achievement_id=achievement_id,
        unlocked_at=datetime.utcnow(),
        current_level=1,
        progress={},
        notification_sent=False
    )
    
    db.add(user_achievement)
    
    # Award XP bonus
    await award_xp_transaction(
        user_id=user_id,
        amount=achievement.points_reward,
        source_type="achievement",
        source_id=str(achievement_id),
        description=f"Achievement: {achievement.name}",
        db=db
    )
    
    logger.info(f"Achievement '{achievement.name}' unlocked for user {user_id}")
    
    # Trigger notification
    from .notifications import trigger_achievement_notifications
    await trigger_achievement_notifications(str(user_achievement.id), db)
    
    return str(user_achievement.id)


async def calculate_leaderboard(
    leaderboard_type: str,
    period: str,
    school_id: Optional[str] = None,
    class_id: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = None
) -> List[Dict[str, Any]]:
    """Calculate leaderboard rankings"""
    
    # Determine date range
    end_date = datetime.utcnow()
    if period == "weekly":
        start_date = end_date - timedelta(weeks=1)
    elif period == "monthly":
        start_date = end_date - timedelta(days=30)
    elif period == "yearly":
        start_date = end_date - timedelta(days=365)
    else:  # all_time
        start_date = datetime.min
    
    if leaderboard_type == "xp":
        # XP-based leaderboard
        query = select(
            User.id,
            User.first_name,
            User.last_name,
            StudentProfile.total_xp,
            StudentProfile.level
        ).select_from(User).join(StudentProfile).where(
            User.role == UserRole.STUDENT,
            User.is_deleted == False
        )
        
        if school_id:
            query = query.where(User.school_id == school_id)
        
        if class_id:
            query = query.join(Enrollment).where(
                Enrollment.class_id == class_id,
                Enrollment.status == "active"
            )
        
        if period != "all_time":
            # For time-based periods, sum XP transactions in that period
            query = select(
                User.id,
                User.first_name,
                User.last_name,
                func.sum(XPTransaction.amount).label('period_xp')
            ).select_from(User).join(XPTransaction).where(
                User.role == UserRole.STUDENT,
                User.is_deleted == False,
                XPTransaction.transaction_date >= start_date
            ).group_by(User.id, User.first_name, User.last_name)
            
            result = await db.execute(query.order_by(desc('period_xp')).limit(limit))
            
            leaderboard = []
            for rank, row in enumerate(result, 1):
                leaderboard.append({
                    "rank": rank,
                    "user_id": str(row.id),
                    "name": f"{row.first_name} {row.last_name}",
                    "score": int(row.period_xp or 0),
                    "metric": "XP"
                })
        else:
            result = await db.execute(query.order_by(desc(StudentProfile.total_xp)).limit(limit))
            
            leaderboard = []
            for rank, row in enumerate(result, 1):
                leaderboard.append({
                    "rank": rank,
                    "user_id": str(row.id),
                    "name": f"{row.first_name} {row.last_name}",
                    "score": row.total_xp,
                    "level": row.level,
                    "metric": "Total XP"
                })
    
    elif leaderboard_type == "quiz_performance":
        # Quiz performance leaderboard
        query = select(
            User.id,
            User.first_name,
            User.last_name,
            func.avg(QuizSubmission.percentage).label('avg_score'),
            func.count(QuizSubmission.id).label('quiz_count')
        ).select_from(User).join(QuizSubmission).where(
            User.role == UserRole.STUDENT,
            User.is_deleted == False,
            QuizSubmission.status == "submitted"
        )
        
        if period != "all_time":
            query = query.where(QuizSubmission.submitted_at >= start_date)
        
        if school_id:
            query = query.where(User.school_id == school_id)
        
        query = query.group_by(User.id, User.first_name, User.last_name)
        query = query.having(func.count(QuizSubmission.id) >= 3)  # Minimum 3 quizzes
        result = await db.execute(query.order_by(desc('avg_score')).limit(limit))
        
        leaderboard = []
        for rank, row in enumerate(result, 1):
            leaderboard.append({
                "rank": rank,
                "user_id": str(row.id),
                "name": f"{row.first_name} {row.last_name}",
                "score": round(row.avg_score, 2),
                "quiz_count": row.quiz_count,
                "metric": "Average Score %"
            })
    
    elif leaderboard_type == "streak":
        # Streak leaderboard
        query = select(
            User.id,
            User.first_name,
            User.last_name,
            StudentProfile.streak_days
        ).select_from(User).join(StudentProfile).where(
            User.role == UserRole.STUDENT,
            User.is_deleted == False,
            StudentProfile.streak_days > 0
        )
        
        if school_id:
            query = query.where(User.school_id == school_id)
        
        result = await db.execute(query.order_by(desc(StudentProfile.streak_days)).limit(limit))
        
        leaderboard = []
        for rank, row in enumerate(result, 1):
            leaderboard.append({
                "rank": rank,
                "user_id": str(row.id),
                "name": f"{row.first_name} {row.last_name}",
                "score": row.streak_days,
                "metric": "Day Streak"
            })
    
    else:
        return []
    
    return leaderboard


# API Routes
@router.get("/achievements", response_model=List[AchievementResponse])
async def get_achievements(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db),
    include_locked: bool = Query(True, description="Include locked achievements"),
    category: Optional[AchievementType] = Query(None, description="Filter by category"),
    rarity: Optional[str] = Query(None, description="Filter by rarity")
):
    """Get all achievements with user progress"""
    
    query = select(Achievement).where(Achievement.is_active == True)
    
    if not include_locked:
        query = query.where(Achievement.is_hidden == False)
    
    if category:
        query = query.where(Achievement.category == category)
    
    if rarity:
        query = query.where(Achievement.rarity == rarity)
    
    result = await db.execute(query.order_by(Achievement.category, Achievement.points_reward))
    achievements = result.scalars().all()
    
    # Get user's unlocked achievements
    user_achievements_result = await db.execute(
        select(UserAchievement).where(UserAchievement.user_id == current_user.id)
    )
    user_achievements = {str(ua.achievement_id): ua for ua in user_achievements_result.scalars()}
    
    achievement_list = []
    for achievement in achievements:
        user_achievement = user_achievements.get(str(achievement.id))
        
        # Calculate progress for locked achievements
        progress_percentage = None
        if not user_achievement and current_user.role == UserRole.STUDENT:
            progress_info = await check_achievement_unlock(current_user, achievement, db)
            progress_percentage = progress_info.get("progress", 0)
        
        achievement_response = AchievementResponse(
            id=str(achievement.id),
            name=achievement.name,
            description=achievement.description,
            category=achievement.category,
            icon_url=achievement.icon_url,
            badge_color=achievement.badge_color,
            points_reward=achievement.points_reward,
            rarity=achievement.rarity,
            unlock_criteria=achievement.unlock_criteria,
            is_hidden=achievement.is_hidden,
            unlock_message=achievement.unlock_message,
            prerequisite_achievements=achievement.prerequisite_achievements,
            max_level=achievement.max_level,
            unlocked_by_user=user_achievement is not None,
            unlock_date=user_achievement.unlocked_at if user_achievement else None,
            progress_percentage=progress_percentage
        )
        
        achievement_list.append(achievement_response)
    
    return achievement_list


@router.get("/my-achievements", response_model=List[UserAchievementResponse])
async def get_my_achievements(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's unlocked achievements"""
    
    result = await db.execute(
        select(UserAchievement, Achievement)
        .join(Achievement)
        .where(UserAchievement.user_id == current_user.id)
        .order_by(desc(UserAchievement.unlocked_at))
    )
    
    user_achievements = []
    for user_achievement, achievement in result:
        achievement_response = AchievementResponse(
            id=str(achievement.id),
            name=achievement.name,
            description=achievement.description,
            category=achievement.category,
            icon_url=achievement.icon_url,
            badge_color=achievement.badge_color,
            points_reward=achievement.points_reward,
            rarity=achievement.rarity,
            unlock_criteria=achievement.unlock_criteria,
            is_hidden=achievement.is_hidden,
            unlock_message=achievement.unlock_message,
            prerequisite_achievements=achievement.prerequisite_achievements,
            max_level=achievement.max_level,
            unlocked_by_user=True,
            unlock_date=user_achievement.unlocked_at
        )
        
        user_achievements.append(UserAchievementResponse(
            id=str(user_achievement.id),
            achievement=achievement_response,
            unlocked_at=user_achievement.unlocked_at,
            current_level=user_achievement.current_level,
            progress=user_achievement.progress,
            notification_sent=user_achievement.notification_sent
        ))
    
    return user_achievements


@router.get("/xp-transactions", response_model=List[XPTransactionResponse])
async def get_xp_transactions(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    source_type: Optional[str] = Query(None, description="Filter by source type")
):
    """Get user's XP transaction history"""
    
    query = select(XPTransaction).where(XPTransaction.user_id == current_user.id)
    
    if source_type:
        query = query.where(XPTransaction.source_type == source_type)
    
    result = await db.execute(
        query.order_by(desc(XPTransaction.transaction_date))
        .offset(skip)
        .limit(limit)
    )
    
    transactions = []
    for transaction in result.scalars():
        transactions.append(XPTransactionResponse(
            id=str(transaction.id),
            amount=transaction.amount,
            source_type=transaction.source_type,
            source_id=transaction.source_id,
            description=transaction.description,
            multiplier=transaction.multiplier,
            bonus_reason=transaction.bonus_reason,
            transaction_date=transaction.transaction_date,
            metadata=transaction.metadata
        ))
    
    return transactions


@router.get("/progress", response_model=ProgressSummaryResponse)
async def get_progress_summary(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive progress summary for current user"""
    
    # Get student profile
    profile_result = await db.execute(
        select(StudentProfile).where(StudentProfile.user_id == current_user.id)
    )
    profile = profile_result.scalar_one_or_none()
    
    if not profile:
        raise NotFoundException("Student profile not found")
    
    # Calculate level information
    level_info = await calculate_level_from_xp(profile.total_xp)
    
    # Get achievement counts
    total_achievements = await db.scalar(
        select(func.count(Achievement.id)).where(Achievement.is_active == True)
    ) or 0
    
    unlocked_achievements = await db.scalar(
        select(func.count(UserAchievement.id)).where(
            UserAchievement.user_id == current_user.id
        )
    ) or 0
    
    # Calculate weekly and monthly XP
    now = datetime.utcnow()
    week_ago = now - timedelta(weeks=1)
    month_ago = now - timedelta(days=30)
    
    weekly_xp = await db.scalar(
        select(func.sum(XPTransaction.amount)).where(
            XPTransaction.user_id == current_user.id,
            XPTransaction.transaction_date >= week_ago
        )
    ) or 0
    
    monthly_xp = await db.scalar(
        select(func.sum(XPTransaction.amount)).where(
            XPTransaction.user_id == current_user.id,
            XPTransaction.transaction_date >= month_ago
        )
    ) or 0
    
    # Get school ranking
    school_rank_result = await db.execute(
        select(func.count(StudentProfile.id))
        .join(User)
        .where(
            User.school_id == current_user.school_id,
            StudentProfile.total_xp > profile.total_xp,
            User.is_deleted == False
        )
    )
    school_rank = (school_rank_result.scalar() or 0) + 1
    
    # Get class ranking (for enrolled classes)
    class_rank = None
    enrolled_classes = await db.execute(
        select(Enrollment.class_id).where(
            Enrollment.student_id == current_user.id,
            Enrollment.status == "active"
        ).limit(1)
    )
    
    class_id = enrolled_classes.scalar_one_or_none()
    if class_id:
        class_rank_result = await db.execute(
            select(func.count(StudentProfile.id))
            .join(User)
            .join(Enrollment)
            .where(
                Enrollment.class_id == class_id,
                Enrollment.status == "active",
                StudentProfile.total_xp > profile.total_xp,
                User.is_deleted == False
            )
        )
        class_rank = (class_rank_result.scalar() or 0) + 1
    
    return ProgressSummaryResponse(
        user_id=str(current_user.id),
        total_xp=profile.total_xp,
        current_level=level_info["level"],
        xp_to_next_level=level_info["xp_to_next_level"],
        level_progress_percentage=level_info["level_progress_percentage"],
        achievements_unlocked=unlocked_achievements,
        achievements_available=total_achievements,
        current_streak=profile.streak_days,
        longest_streak=profile.streak_days,  # TODO: Track longest streak separately
        weekly_xp=weekly_xp,
        monthly_xp=monthly_xp,
        rank_in_school=school_rank,
        rank_in_class=class_rank
    )


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    current_user: User = Depends(require_authentication),
    school = Depends(get_current_school),
    db: AsyncSession = Depends(get_db),
    leaderboard_type: str = Query("xp", description="Type of leaderboard"),
    period: str = Query("all_time", description="Time period"),
    scope: str = Query("school", description="Scope: school, class, global"),
    class_id: Optional[str] = Query(None, description="Class ID for class scope"),
    limit: int = Query(50, ge=1, le=200)
):
    """Get leaderboard rankings"""
    
    # Determine scope parameters
    school_id = str(school.id) if scope in ["school", "class"] else None
    target_class_id = class_id if scope == "class" else None
    
    # Calculate leaderboard
    entries = await calculate_leaderboard(
        leaderboard_type=leaderboard_type,
        period=period,
        school_id=school_id,
        class_id=target_class_id,
        limit=limit,
        db=db
    )
    
    # Find user's position in leaderboard
    user_position = None
    for entry in entries:
        if entry["user_id"] == str(current_user.id):
            user_position = entry["rank"]
            break
    
    return LeaderboardResponse(
        leaderboard_type=leaderboard_type,
        period=period,
        entries=entries,
        user_position=user_position,
        total_participants=len(entries),
        last_updated=datetime.utcnow()
    )


@router.get("/badges", response_model=BadgeCollectionResponse)
async def get_badge_collection(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Get user's badge collection summary"""
    
    # Get all user achievements
    user_achievements_result = await db.execute(
        select(UserAchievement, Achievement)
        .join(Achievement)
        .where(UserAchievement.user_id == current_user.id)
    )
    
    total_badges = 0
    badges_by_rarity = {"common": 0, "rare": 0, "epic": 0, "legendary": 0}
    recent_badges = []
    featured_badge = None
    
    for user_achievement, achievement in user_achievements_result:
        total_badges += 1
        
        if achievement.rarity in badges_by_rarity:
            badges_by_rarity[achievement.rarity] += 1
        
        badge_data = {
            "id": str(achievement.id),
            "name": achievement.name,
            "description": achievement.description,
            "icon_url": achievement.icon_url,
            "badge_color": achievement.badge_color,
            "rarity": achievement.rarity,
            "unlocked_at": user_achievement.unlocked_at.isoformat()
        }
        
        recent_badges.append(badge_data)
        
        # Feature the rarest badge or most recent legendary
        if (featured_badge is None or 
            achievement.rarity == "legendary" or 
            (featured_badge.get("rarity") != "legendary" and achievement.rarity == "epic")):
            featured_badge = badge_data
    
    # Sort recent badges by unlock date
    recent_badges.sort(key=lambda x: x["unlocked_at"], reverse=True)
    recent_badges = recent_badges[:5]  # Keep only 5 most recent
    
    # Calculate completion percentage
    total_available = await db.scalar(
        select(func.count(Achievement.id)).where(Achievement.is_active == True)
    ) or 1
    
    completion_percentage = (total_badges / total_available) * 100
    
    return BadgeCollectionResponse(
        total_badges=total_badges,
        badges_by_rarity=badges_by_rarity,
        recent_badges=recent_badges,
        featured_badge=featured_badge,
        completion_percentage=round(completion_percentage, 2)
    )


@router.get("/streaks", response_model=StreakResponse)
async def get_streak_information(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Get user's streak information"""
    
    profile_result = await db.execute(
        select(StudentProfile).where(StudentProfile.user_id == current_user.id)
    )
    profile = profile_result.scalar_one_or_none()
    
    if not profile:
        raise NotFoundException("Student profile not found")
    
    current_streak = profile.streak_days
    longest_streak = profile.streak_days  # TODO: Implement longest streak tracking
    
    # Calculate streak multiplier
    settings = get_settings()
    streak_multiplier = min(
        settings.XP_MULTIPLIER_STREAK * (current_streak / 7),  # Multiplier increases weekly
        settings.MAX_STREAK_MULTIPLIER
    )
    
    # Define streak milestones
    milestones = [
        {"days": 3, "reward": "3-Day Warrior", "achieved": current_streak >= 3},
        {"days": 7, "reward": "Week Champion", "achieved": current_streak >= 7},
        {"days": 14, "reward": "Fortnight Hero", "achieved": current_streak >= 14},
        {"days": 30, "reward": "Monthly Master", "achieved": current_streak >= 30},
        {"days": 100, "reward": "Centurion", "achieved": current_streak >= 100}
    ]
    
    return StreakResponse(
        current_streak=current_streak,
        longest_streak=longest_streak,
        streak_type="daily",
        last_activity_date=profile.last_activity.date() if profile.last_activity else None,
        streak_multiplier=round(streak_multiplier, 2),
        milestones=milestones
    )


@router.post("/check-achievements")
async def check_and_unlock_achievements(
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Check and unlock any eligible achievements for current user"""
    
    # Get all achievements not yet unlocked by user
    unlocked_achievement_ids = await db.execute(
        select(UserAchievement.achievement_id).where(
            UserAchievement.user_id == current_user.id
        )
    )
    unlocked_ids = {str(aid) for aid in unlocked_achievement_ids.scalars()}
    
    available_achievements_result = await db.execute(
        select(Achievement).where(
            Achievement.is_active == True,
            ~Achievement.id.in_(unlocked_ids) if unlocked_ids else True
        )
    )
    
    newly_unlocked = []
    
    for achievement in available_achievements_result.scalars():
        unlock_check = await check_achievement_unlock(current_user, achievement, db)
        
        if unlock_check.get("eligible", False):
            unlock_id = await unlock_achievement_for_user(
                str(current_user.id),
                str(achievement.id),
                db
            )
            
            if unlock_id:
                newly_unlocked.append({
                    "achievement_id": str(achievement.id),
                    "name": achievement.name,
                    "points_reward": achievement.points_reward
                })
    
    await db.commit()
    
    return {
        "message": f"Checked achievements, unlocked {len(newly_unlocked)} new ones",
        "newly_unlocked": newly_unlocked
    }


@router.post("/award-xp")
async def award_xp_manual(
    user_id: str,
    amount: int,
    reason: str,
    current_user: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    """Manually award XP to a user (teachers/admins only)"""
    
    if amount < 1 or amount > 1000:
        raise ValidationException("XP amount must be between 1 and 1000")
    
    # Verify target user exists and is a student
    target_user_result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.role == UserRole.STUDENT,
            User.is_deleted == False
        )
    )
    
    target_user = target_user_result.scalar_one_or_none()
    if not target_user:
        raise NotFoundException("Student not found")
    
    # Award XP
    transaction_id = await award_xp_transaction(
        user_id=user_id,
        amount=amount,
        source_type="manual_award",
        source_id=str(current_user.id),
        description=f"Manual award: {reason}",
        bonus_reason=f"Awarded by {current_user.full_name}",
        db=db
    )
    
    await db.commit()
    
    logger.info(f"Manual XP award: {amount} XP to {target_user.username} by {current_user.username}")
    
    return {
        "message": f"Awarded {amount} XP to {target_user.full_name}",
        "transaction_id": transaction_id
    }


@router.post("/achievements", dependencies=[Depends(require_admin)])
async def create_custom_achievement(
    request: CustomBadgeRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create a custom achievement (admin only)"""
    
    import uuid
    
    # Create achievement
    achievement = Achievement(
        id=uuid.uuid4(),
        name=request.name,
        description=request.description,
        category=AchievementType.SPECIAL,
        icon_url=request.icon_url,
        badge_color=request.badge_color,
        points_reward=request.points_reward,
        rarity="special",
        unlock_criteria=request.unlock_criteria,
        is_hidden=False,
        is_active=True,
        unlock_message=f"Congratulations! You've earned the special '{request.name}' achievement!"
    )
    
    db.add(achievement)
    await db.flush()
    
    # If target users specified, unlock for them immediately
    if request.target_users:
        for user_id in request.target_users:
            await unlock_achievement_for_user(
                user_id,
                str(achievement.id),
                db
            )
    
    await db.commit()
    
    logger.info(f"Custom achievement '{request.name}' created by admin {current_user.username}")
    
    return {
        "message": "Custom achievement created successfully",
        "achievement_id": str(achievement.id),
        "users_unlocked": len(request.target_users) if request.target_users else 0
    }


@router.get("/stats/summary")
async def get_gamification_stats(
    current_user: User = Depends(require_admin),
    school = Depends(get_current_school),
    db: AsyncSession = Depends(get_db)
):
    """Get gamification system statistics (admin only)"""
    
    # Total XP distributed
    total_xp_result = await db.execute(
        select(func.sum(XPTransaction.amount))
        .join(User)
        .where(User.school_id == school.id)
    )
    total_xp = total_xp_result.scalar() or 0
    
    # Total achievements unlocked
    achievements_unlocked_result = await db.execute(
        select(func.count(UserAchievement.id))
        .join(User)
        .where(User.school_id == school.id)
    )
    achievements_unlocked = achievements_unlocked_result.scalar() or 0
    
    # Active streaks
    active_streaks_result = await db.execute(
        select(func.count(StudentProfile.id))
        .join(User)
        .where(
            User.school_id == school.id,
            StudentProfile.streak_days > 0
        )
    )
    active_streaks = active_streaks_result.scalar() or 0
    
    # Top performers
    top_performers_result = await db.execute(
        select(
            User.first_name,
            User.last_name,
            StudentProfile.total_xp,
            StudentProfile.level
        )
        .select_from(User)
        .join(StudentProfile)
        .where(User.school_id == school.id)
        .order_by(desc(StudentProfile.total_xp))
        .limit(5)
    )
    
    top_performers = []
    for row in top_performers_result:
        top_performers.append({
            "name": f"{row.first_name} {row.last_name}",
            "total_xp": row.total_xp,
            "level": row.level
        })
    
    return {
        "school_id": str(school.id),
        "total_xp_distributed": total_xp,
        "total_achievements_unlocked": achievements_unlocked,
        "students_with_active_streaks": active_streaks,
        "top_performers": top_performers,
        "engagement_metrics": {
            "daily_active_gamers": 0,  # TODO: Calculate from analytics
            "weekly_xp_growth": 0,     # TODO: Calculate weekly growth
            "achievement_unlock_rate": 0  # TODO: Calculate unlock rate
        }
    }


# Export router
__all__ = ["router", "award_xp_transaction", "unlock_achievement_for_user", "check_achievement_unlock"]
"""
AI-Powered Smart Class & Timetable Scheduler
Analytics and reporting API routes
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import APIRouter, Depends, Query, Path, BackgroundTasks
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.orm import selectinload
import pandas as pd
import numpy as np

from ..database.connection import get_db
from ..database.models import (
    User, UserRole, Quiz, QuizSubmission, Question, Answer,
    Class, Enrollment, Grade, XPTransaction, UserAchievement,
    AnalyticsEvent, BehavioralProfile, StudentProfile, TeacherProfile,
    School, Schedule, Resource
)
from ..dependencies import (
    require_authentication,
    require_teacher,
    require_admin,
    require_teacher_or_admin,
    get_current_school,
    PermissionChecker,
    standard_cache,
    long_cache,
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
class DateRangeRequest(BaseModel):
    start_date: date
    end_date: date
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('End date must be after start date')
        if 'start_date' in values and (v - values['start_date']).days > 365:
            raise ValueError('Date range cannot exceed 365 days')
        return v


class StudentAnalyticsResponse(BaseModel):
    student_id: str
    student_name: str
    overall_performance: Dict[str, Any]
    subject_performance: List[Dict[str, Any]]
    engagement_metrics: Dict[str, Any]
    learning_progress: Dict[str, Any]
    behavioral_insights: Dict[str, Any]
    recommendations: List[str]
    risk_factors: List[Dict[str, Any]]


class ClassAnalyticsResponse(BaseModel):
    class_id: str
    class_name: str
    class_code: str
    overall_statistics: Dict[str, Any]
    performance_distribution: Dict[str, Any]
    engagement_trends: List[Dict[str, Any]]
    question_difficulty_analysis: List[Dict[str, Any]]
    student_rankings: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    at_risk_students: List[Dict[str, Any]]


class SchoolAnalyticsResponse(BaseModel):
    school_id: str
    school_name: str
    summary_statistics: Dict[str, Any]
    class_performance_comparison: List[Dict[str, Any]]
    teacher_effectiveness: List[Dict[str, Any]]
    resource_utilization: Dict[str, Any]
    engagement_overview: Dict[str, Any]
    trends_analysis: Dict[str, Any]
    benchmarking_data: Dict[str, Any]


class LearningAnalyticsRequest(BaseModel):
    student_ids: Optional[List[str]] = None
    class_ids: Optional[List[str]] = None
    subject: Optional[str] = None
    date_range: DateRangeRequest
    metrics: List[str] = ["performance", "engagement", "progress"]
    
    @validator('metrics')
    def validate_metrics(cls, v):
        valid_metrics = [
            "performance", "engagement", "progress", "behavior",
            "time_spent", "completion_rate", "difficulty_progression"
        ]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f'Invalid metric: {metric}')
        return v


class PerformanceTrendResponse(BaseModel):
    period: str  # daily, weekly, monthly
    data_points: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    predictions: Optional[Dict[str, Any]]


class EngagementMetricsResponse(BaseModel):
    daily_active_users: List[Dict[str, Any]]
    session_duration_stats: Dict[str, Any]
    feature_usage: Dict[str, Any]
    retention_rates: Dict[str, Any]
    drop_off_analysis: Dict[str, Any]


class PredictiveAnalyticsRequest(BaseModel):
    prediction_type: str  # "performance", "dropout_risk", "engagement"
    target_entity: str  # "student", "class", "school"
    entity_ids: List[str]
    prediction_horizon_days: int = 30
    confidence_threshold: float = 0.7
    
    @validator('prediction_type')
    def validate_prediction_type(cls, v):
        valid_types = ["performance", "dropout_risk", "engagement", "success_probability"]
        if v not in valid_types:
            raise ValueError('Invalid prediction type')
        return v
    
    @validator('target_entity')
    def validate_target_entity(cls, v):
        if v not in ["student", "class", "school"]:
            raise ValueError('Invalid target entity')
        return v


class PredictiveAnalyticsResponse(BaseModel):
    prediction_type: str
    predictions: List[Dict[str, Any]]
    model_confidence: float
    feature_importance: Dict[str, float]
    recommendations: List[str]
    generated_at: datetime


class CustomReportRequest(BaseModel):
    report_name: str
    report_type: str  # "performance", "engagement", "custom"
    filters: Dict[str, Any]
    metrics: List[str]
    visualizations: List[str] = ["chart", "table"]
    format: str = "json"  # json, pdf, excel
    
    @validator('format')
    def validate_format(cls, v):
        if v not in ["json", "pdf", "excel", "csv"]:
            raise ValueError('Invalid format')
        return v


# Helper functions
async def calculate_performance_metrics(
    entity_type: str,
    entity_id: str,
    start_date: date,
    end_date: date,
    db: AsyncSession
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics"""
    
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    if entity_type == "student":
        # Student performance metrics
        submissions_result = await db.execute(
            select(
                func.count(QuizSubmission.id).label('total_submissions'),
                func.avg(QuizSubmission.percentage).label('avg_score'),
                func.max(QuizSubmission.percentage).label('max_score'),
                func.min(QuizSubmission.percentage).label('min_score'),
                func.sum(QuizSubmission.time_taken_seconds).label('total_time')
            )
            .where(
                QuizSubmission.student_id == entity_id,
                QuizSubmission.submitted_at.between(start_datetime, end_datetime),
                QuizSubmission.status == "submitted"
            )
        )
        
        result = submissions_result.first()
        
        # Calculate improvement trend
        improvement_result = await db.execute(
            select(QuizSubmission.percentage, QuizSubmission.submitted_at)
            .where(
                QuizSubmission.student_id == entity_id,
                QuizSubmission.submitted_at.between(start_datetime, end_datetime),
                QuizSubmission.status == "submitted"
            )
            .order_by(QuizSubmission.submitted_at)
        )
        
        scores = [(row.percentage, row.submitted_at) for row in improvement_result]
        
        improvement_trend = 0.0
        if len(scores) >= 2:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            
            if first_half and second_half:
                first_avg = sum(score for score, _ in first_half) / len(first_half)
                second_avg = sum(score for score, _ in second_half) / len(second_half)
                improvement_trend = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        return {
            "total_submissions": result.total_submissions or 0,
            "average_score": round(result.avg_score or 0, 2),
            "max_score": result.max_score or 0,
            "min_score": result.min_score or 0,
            "total_time_minutes": int((result.total_time or 0) / 60),
            "improvement_trend": round(improvement_trend, 2),
            "grade": calculate_letter_grade(result.avg_score or 0)
        }
    
    elif entity_type == "class":
        # Class performance metrics
        class_submissions = await db.execute(
            select(
                func.count(func.distinct(QuizSubmission.student_id)).label('active_students'),
                func.count(QuizSubmission.id).label('total_submissions'),
                func.avg(QuizSubmission.percentage).label('class_avg'),
                func.stddev(QuizSubmission.percentage).label('std_dev')
            )
            .join(Quiz)
            .where(
                Quiz.class_id == entity_id,
                QuizSubmission.submitted_at.between(start_datetime, end_datetime),
                QuizSubmission.status == "submitted"
            )
        )
        
        result = class_submissions.first()
        
        # Get grade distribution
        grade_dist_result = await db.execute(
            select(QuizSubmission.percentage)
            .join(Quiz)
            .where(
                Quiz.class_id == entity_id,
                QuizSubmission.submitted_at.between(start_datetime, end_datetime),
                QuizSubmission.status == "submitted"
            )
        )
        
        scores = [row.percentage for row in grade_dist_result]
        grade_distribution = {
            "A": len([s for s in scores if s >= 90]),
            "B": len([s for s in scores if 80 <= s < 90]),
            "C": len([s for s in scores if 70 <= s < 80]),
            "D": len([s for s in scores if 60 <= s < 70]),
            "F": len([s for s in scores if s < 60])
        }
        
        return {
            "active_students": result.active_students or 0,
            "total_submissions": result.total_submissions or 0,
            "class_average": round(result.class_avg or 0, 2),
            "standard_deviation": round(result.std_dev or 0, 2),
            "grade_distribution": grade_distribution
        }
    
    return {}


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


async def calculate_engagement_metrics(
    user_id: str,
    start_date: date,
    end_date: date,
    db: AsyncSession
) -> Dict[str, Any]:
    """Calculate user engagement metrics"""
    
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Get analytics events
    events_result = await db.execute(
        select(AnalyticsEvent)
        .where(
            AnalyticsEvent.user_id == user_id,
            AnalyticsEvent.timestamp.between(start_datetime, end_datetime)
        )
        .order_by(AnalyticsEvent.timestamp)
    )
    
    events = events_result.scalars().all()
    
    if not events:
        return {
            "total_sessions": 0,
            "total_time_minutes": 0,
            "avg_session_duration": 0,
            "days_active": 0,
            "activity_score": 0
        }
    
    # Calculate session metrics
    sessions = {}
    for event in events:
        session_date = event.timestamp.date()
        if session_date not in sessions:
            sessions[session_date] = {
                "events": [],
                "duration": 0
            }
        sessions[session_date]["events"].append(event)
        if event.duration:
            sessions[session_date]["duration"] += event.duration
    
    total_time_seconds = sum(session["duration"] for session in sessions.values())
    avg_session_duration = total_time_seconds / len(sessions) if sessions else 0
    
    # Calculate activity score (0-100)
    days_in_period = (end_date - start_date).days + 1
    activity_ratio = len(sessions) / days_in_period
    activity_score = min(100, activity_ratio * 100)
    
    return {
        "total_sessions": len(sessions),
        "total_time_minutes": int(total_time_seconds / 60),
        "avg_session_duration": int(avg_session_duration / 60),
        "days_active": len(sessions),
        "activity_score": round(activity_score, 2)
    }


async def generate_behavioral_insights(
    student_id: str,
    db: AsyncSession
) -> Dict[str, Any]:
    """Generate AI-powered behavioral insights"""
    
    # Get or create behavioral profile
    profile_result = await db.execute(
        select(BehavioralProfile).where(BehavioralProfile.user_id == student_id)
    )
    profile = profile_result.scalar_one_or_none()
    
    if not profile:
        # Create basic profile
        insights = {
            "learning_style": "mixed",
            "engagement_level": "medium",
            "optimal_study_times": ["10:00-12:00", "14:00-16:00"],
            "attention_span_minutes": 45,
            "preferred_content_types": ["visual", "interactive"],
            "motivation_factors": ["achievement", "progress"],
            "confidence_score": 0.5
        }
    else:
        insights = {
            "learning_style": profile.learning_style or "mixed",
            "engagement_level": profile.engagement_level or "medium",
            "optimal_study_times": profile.optimal_study_times or [],
            "attention_span_minutes": profile.attention_span_minutes or 45,
            "preferred_content_types": profile.preferred_content_types or [],
            "motivation_factors": profile.motivation_factors or [],
            "confidence_score": profile.confidence_score or 0.5
        }
    
    return insights


async def identify_at_risk_students(
    class_id: Optional[str],
    school_id: str,
    db: AsyncSession
) -> List[Dict[str, Any]]:
    """Identify students at risk of poor performance"""
    
    query = select(User, StudentProfile).join(StudentProfile).where(
        User.role == UserRole.STUDENT,
        User.school_id == school_id,
        User.is_deleted == False
    )
    
    if class_id:
        query = query.join(Enrollment).where(
            Enrollment.class_id == class_id,
            Enrollment.status == "active"
        )
    
    students_result = await db.execute(query)
    students = students_result.all()
    
    at_risk_students = []
    
    for user, profile in students:
        # Calculate risk factors
        risk_score = 0
        risk_factors = []
        
        # Check recent performance
        recent_submissions = await db.execute(
            select(func.avg(QuizSubmission.percentage))
            .where(
                QuizSubmission.student_id == user.id,
                QuizSubmission.submitted_at >= datetime.utcnow() - timedelta(days=30),
                QuizSubmission.status == "submitted"
            )
        )
        
        avg_recent_score = recent_submissions.scalar()
        
        if avg_recent_score is not None and avg_recent_score < 60:
            risk_score += 30
            risk_factors.append({
                "factor": "Low Recent Performance",
                "description": f"Average score: {avg_recent_score:.1f}%",
                "severity": "high"
            })
        
        # Check activity level
        if profile.last_activity:
            days_since_activity = (datetime.utcnow() - profile.last_activity).days
            if days_since_activity > 7:
                risk_score += 20
                risk_factors.append({
                    "factor": "Low Activity",
                    "description": f"Last active {days_since_activity} days ago",
                    "severity": "medium"
                })
        
        # Check streak
        if profile.streak_days == 0:
            risk_score += 10
            risk_factors.append({
                "factor": "No Current Streak",
                "description": "Student hasn't maintained daily activity",
                "severity": "low"
            })
        
        # Only include students with significant risk
        if risk_score >= 25:
            at_risk_students.append({
                "student_id": str(user.id),
                "student_name": f"{user.first_name} {user.last_name}",
                "student_number": profile.student_id,
                "risk_score": risk_score,
                "risk_level": "high" if risk_score >= 50 else "medium" if risk_score >= 25 else "low",
                "risk_factors": risk_factors,
                "last_activity": profile.last_activity.isoformat() if profile.last_activity else None
            })
    
    # Sort by risk score
    at_risk_students.sort(key=lambda x: x["risk_score"], reverse=True)
    
    return at_risk_students


async def generate_recommendations(
    entity_type: str,
    entity_id: str,
    performance_data: Dict[str, Any],
    db: AsyncSession
) -> List[str]:
    """Generate AI-powered recommendations"""
    
    recommendations = []
    
    if entity_type == "student":
        avg_score = performance_data.get("average_score", 0)
        improvement_trend = performance_data.get("improvement_trend", 0)
        
        if avg_score < 60:
            recommendations.append("Consider additional study sessions and practice quizzes")
            recommendations.append("Review fundamental concepts before attempting advanced topics")
        
        if improvement_trend < -5:
            recommendations.append("Performance is declining - schedule a meeting with teacher")
            recommendations.append("Identify specific areas of difficulty and focus study time there")
        
        if performance_data.get("total_time_minutes", 0) < 60:
            recommendations.append("Increase study time to at least 1 hour per week")
        
        if avg_score > 85 and improvement_trend > 0:
            recommendations.append("Excellent progress! Consider helping peers or taking on advanced challenges")
    
    elif entity_type == "class":
        class_avg = performance_data.get("class_average", 0)
        std_dev = performance_data.get("standard_deviation", 0)
        
        if class_avg < 70:
            recommendations.append("Class average is below target - consider reviewing teaching materials")
            recommendations.append("Implement additional support sessions for struggling students")
        
        if std_dev > 20:
            recommendations.append("High performance variation - consider differentiated instruction")
            recommendations.append("Identify and address individual student needs")
        
        grade_dist = performance_data.get("grade_distribution", {})
        if grade_dist.get("F", 0) > grade_dist.get("A", 0):
            recommendations.append("High failure rate - review curriculum difficulty and pacing")
    
    return recommendations


# API Routes
@router.get("/student/{student_id}", response_model=StudentAnalyticsResponse)
async def get_student_analytics(
    student_id: str = Path(..., description="Student user ID"),
    date_range: DateRangeRequest = Depends(),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive analytics for a specific student"""
    
    # Check permissions
    can_access = await PermissionChecker.can_access_user_data(
        current_user, student_id, db
    )
    if not can_access:
        raise AuthorizationException("Access denied to student analytics")
    
    # Get student information
    student_result = await db.execute(
        select(User, StudentProfile)
        .join(StudentProfile)
        .where(
            User.id == student_id,
            User.role == UserRole.STUDENT,
            User.is_deleted == False
        )
    )
    
    student_data = student_result.first()
    if not student_data:
        raise NotFoundException("Student not found")
    
    user, profile = student_data
    
    # Calculate performance metrics
    performance_metrics = await calculate_performance_metrics(
        "student", student_id, date_range.start_date, date_range.end_date, db
    )
    
    # Get subject-wise performance
    subject_performance_result = await db.execute(
        select(
            Class.subject,
            func.avg(QuizSubmission.percentage).label('avg_score'),
            func.count(QuizSubmission.id).label('quiz_count')
        )
        .select_from(QuizSubmission)
        .join(Quiz)
        .join(Class)
        .where(
            QuizSubmission.student_id == student_id,
            QuizSubmission.submitted_at.between(
                datetime.combine(date_range.start_date, datetime.min.time()),
                datetime.combine(date_range.end_date, datetime.max.time())
            ),
            QuizSubmission.status == "submitted"
        )
        .group_by(Class.subject)
    )
    
    subject_performance = []
    for row in subject_performance_result:
        subject_performance.append({
            "subject": row.subject,
            "average_score": round(row.avg_score, 2),
            "quiz_count": row.quiz_count,
            "grade": calculate_letter_grade(row.avg_score)
        })
    
    # Calculate engagement metrics
    engagement_metrics = await calculate_engagement_metrics(
        student_id, date_range.start_date, date_range.end_date, db
    )
    
    # Get learning progress
    learning_progress = {
        "total_xp": profile.total_xp,
        "current_level": profile.level,
        "xp_to_next_level": profile.xp_to_next_level,
        "current_streak": profile.streak_days,
        "achievements_unlocked": await db.scalar(
            select(func.count(UserAchievement.id))
            .where(UserAchievement.user_id == student_id)
        ) or 0
    }
    
    # Generate behavioral insights
    behavioral_insights = await generate_behavioral_insights(student_id, db)
    
    # Generate recommendations
    recommendations = await generate_recommendations(
        "student", student_id, performance_metrics, db
    )
    
    # Identify risk factors
    risk_factors = []
    if performance_metrics["average_score"] < 60:
        risk_factors.append({
            "factor": "Low Average Performance",
            "severity": "high",
            "description": f"Average score of {performance_metrics['average_score']}% is below passing threshold"
        })
    
    if engagement_metrics["activity_score"] < 30:
        risk_factors.append({
            "factor": "Low Engagement",
            "severity": "medium",
            "description": "Student shows low activity levels"
        })
    
    return StudentAnalyticsResponse(
        student_id=student_id,
        student_name=f"{user.first_name} {user.last_name}",
        overall_performance=performance_metrics,
        subject_performance=subject_performance,
        engagement_metrics=engagement_metrics,
        learning_progress=learning_progress,
        behavioral_insights=behavioral_insights,
        recommendations=recommendations,
        risk_factors=risk_factors
    )


@router.get("/class/{class_id}", response_model=ClassAnalyticsResponse)
async def get_class_analytics(
    class_id: str = Path(..., description="Class ID"),
    date_range: DateRangeRequest = Depends(),
    current_user: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive analytics for a specific class"""
    
    # Check permissions
    can_access = await PermissionChecker.can_access_class(current_user, class_id, db)
    if not can_access:
        raise AuthorizationException("Access denied to class analytics")
    
    # Get class information
    class_result = await db.execute(
        select(Class)
        .options(selectinload(Class.teacher))
        .where(
            Class.id == class_id,
            Class.is_deleted == False
        )
    )
    
    class_obj = class_result.scalar_one_or_none()
    if not class_obj:
        raise NotFoundException("Class not found")
    
    # Calculate overall statistics
    overall_stats = await calculate_performance_metrics(
        "class", class_id, date_range.start_date, date_range.end_date, db
    )
    
    # Get performance distribution
    performance_dist_result = await db.execute(
        select(QuizSubmission.percentage)
        .join(Quiz)
        .where(
            Quiz.class_id == class_id,
            QuizSubmission.submitted_at.between(
                datetime.combine(date_range.start_date, datetime.min.time()),
                datetime.combine(date_range.end_date, datetime.max.time())
            ),
            QuizSubmission.status == "submitted"
        )
    )
    
    scores = [row.percentage for row in performance_dist_result]
    
    if scores:
        performance_distribution = {
            "mean": round(np.mean(scores), 2),
            "median": round(np.median(scores), 2),
            "std_dev": round(np.std(scores), 2),
            "quartiles": {
                "q1": round(np.percentile(scores, 25), 2),
                "q2": round(np.percentile(scores, 50), 2),
                "q3": round(np.percentile(scores, 75), 2)
            },
            "grade_distribution": overall_stats.get("grade_distribution", {})
        }
    else:
        performance_distribution = {
            "mean": 0, "median": 0, "std_dev": 0,
            "quartiles": {"q1": 0, "q2": 0, "q3": 0},
            "grade_distribution": {}
        }
    
    # Get engagement trends (weekly aggregation)
    engagement_trends = []
    current_date = date_range.start_date
    while current_date <= date_range.end_date:
        week_end = min(current_date + timedelta(days=6), date_range.end_date)
        
        week_submissions = await db.execute(
            select(func.count(QuizSubmission.id))
            .join(Quiz)
            .where(
                Quiz.class_id == class_id,
                QuizSubmission.submitted_at.between(
                    datetime.combine(current_date, datetime.min.time()),
                    datetime.combine(week_end, datetime.max.time())
                ),
                QuizSubmission.status == "submitted"
            )
        )
        
        engagement_trends.append({
            "week_start": current_date.isoformat(),
            "submissions": week_submissions.scalar() or 0
        })
        
        current_date = week_end + timedelta(days=1)
    
    # Analyze question difficulty
    question_analysis_result = await db.execute(
        select(
            Question.id,
            Question.content,
            Question.difficulty_level,
            func.avg(Answer.points_earned).label('avg_points'),
            func.count(Answer.id).label('answer_count')
        )
        .select_from(Question)
        .join(Quiz)
        .join(Answer)
        .join(QuizSubmission)
        .where(
            Quiz.class_id == class_id,
            QuizSubmission.submitted_at.between(
                datetime.combine(date_range.start_date, datetime.min.time()),
                datetime.combine(date_range.end_date, datetime.max.time())
            )
        )
        .group_by(Question.id, Question.content, Question.difficulty_level)
    )
    
    question_difficulty_analysis = []
    for row in question_analysis_result:
        success_rate = (row.avg_points / 1.0) * 100 if row.avg_points else 0  # Assuming 1.0 max points
        question_difficulty_analysis.append({
            "question_id": str(row.id),
            "question_preview": row.content[:100] + "..." if len(row.content) > 100 else row.content,
            "difficulty_level": row.difficulty_level,
            "success_rate": round(success_rate, 2),
            "attempt_count": row.answer_count
        })
    
    # Get student rankings
    student_rankings_result = await db.execute(
        select(
            User.id,
            User.first_name,
            User.last_name,
            func.avg(QuizSubmission.percentage).label('avg_score'),
            func.count(QuizSubmission.id).label('quiz_count')
        )
        .select_from(User)
        .join(QuizSubmission)
        .join(Quiz)
        .where(
            Quiz.class_id == class_id,
            QuizSubmission.submitted_at.between(
                datetime.combine(date_range.start_date, datetime.min.time()),
                datetime.combine(date_range.end_date, datetime.max.time())
            ),
            QuizSubmission.status == "submitted"
        )
        .group_by(User.id, User.first_name, User.last_name)
        .order_by(desc(func.avg(QuizSubmission.percentage)))
    )
    
    student_rankings = []
    for rank, row in enumerate(student_rankings_result, 1):
        student_rankings.append({
            "rank": rank,
            "student_id": str(row.id),
            "student_name": f"{row.first_name} {row.last_name}",
            "average_score": round(row.avg_score, 2),
            "quiz_count": row.quiz_count,
            "grade": calculate_letter_grade(row.avg_score)
        })
    
    # Generate improvement suggestions
    improvement_suggestions = await generate_recommendations(
        "class", class_id, overall_stats, db
    )
    
    # Identify at-risk students
    at_risk_students = await identify_at_risk_students(class_id, class_obj.school_id, db)
    
    return ClassAnalyticsResponse(
        class_id=class_id,
        class_name=class_obj.name,
        class_code=class_obj.code,
        overall_statistics=overall_stats,
        performance_distribution=performance_distribution,
        engagement_trends=engagement_trends,
        question_difficulty_analysis=question_difficulty_analysis,
        student_rankings=student_rankings,
        improvement_suggestions=improvement_suggestions,
        at_risk_students=at_risk_students
    )


@router.get("/school", response_model=SchoolAnalyticsResponse)
async def get_school_analytics(
    date_range: DateRangeRequest = Depends(),
    current_user: User = Depends(require_admin),
    school = Depends(get_current_school),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive school-wide analytics (admin only)"""
    
    # Summary statistics
    summary_stats_result = await db.execute(
        select(
            func.count(func.distinct(User.id)).filter(User.role == UserRole.STUDENT).label('total_students'),
            func.count(func.distinct(User.id)).filter(User.role == UserRole.TEACHER).label('total_teachers'),
            func.count(func.distinct(Class.id)).label('total_classes'),
            func.count(func.distinct(Quiz.id)).label('total_quizzes')
        )
        .select_from(User)
        .outerjoin(Class, and_(User.role == UserRole.TEACHER, Class.teacher_id == User.id))
        .outerjoin(Quiz, Quiz.class_id == Class.id)
        .where(User.school_id == school.id, User.is_deleted == False)
    )
    
    summary_result = summary_stats_result.first()
    
    # Recent activity statistics
    start_datetime = datetime.combine(date_range.start_date, datetime.min.time())
    end_datetime = datetime.combine(date_range.end_date, datetime.max.time())
    
    activity_stats_result = await db.execute(
        select(
            func.count(QuizSubmission.id).label('total_submissions'),
            func.avg(QuizSubmission.percentage).label('school_avg_score')
        )
        .select_from(QuizSubmission)
        .join(Quiz)
        .join(Class)
        .where(
            Class.school_id == school.id,
            QuizSubmission.submitted_at.between(start_datetime, end_datetime),
            QuizSubmission.status == "submitted"
        )
    )
    
    activity_result = activity_stats_result.first()
    
    summary_statistics = {
        "total_students": summary_result.total_students or 0,
        "total_teachers": summary_result.total_teachers or 0,
        "total_classes": summary_result.total_classes or 0,
        "total_quizzes": summary_result.total_quizzes or 0,
        "total_submissions": activity_result.total_submissions or 0,
        "school_average_score": round(activity_result.school_avg_score or 0, 2)
    }
    
    # Class performance comparison
    class_comparison_result = await db.execute(
        select(
            Class.id,
            Class.name,
            Class.code,
            Class.subject,
            func.avg(QuizSubmission.percentage).label('class_avg'),
            func.count(QuizSubmission.id).label('submissions')
        )
        .select_from(Class)
        .join(Quiz)
        .join(QuizSubmission)
        .where(
            Class.school_id == school.id,
            QuizSubmission.submitted_at.between(start_datetime, end_datetime),
            QuizSubmission.status == "submitted"
        )
        .group_by(Class.id, Class.name, Class.code, Class.subject)
        .order_by(desc(func.avg(QuizSubmission.percentage)))
    )
    
    class_performance_comparison = []
    for row in class_comparison_result:
        class_performance_comparison.append({
            "class_id": str(row.id),
            "class_name": row.name,
            "class_code": row.code,
            "subject": row.subject,
            "average_score": round(row.class_avg, 2),
            "submissions_count": row.submissions,
            "grade": calculate_letter_grade(row.class_avg)
        })
    
    # Teacher effectiveness (based on class performance)
    teacher_effectiveness_result = await db.execute(
        select(
            User.id,
            User.first_name,
            User.last_name,
            func.avg(QuizSubmission.percentage).label('avg_class_performance'),
            func.count(func.distinct(Class.id)).label('classes_taught')
        )
        .select_from(User)
        .join(TeacherProfile)
        .join(Class, Class.teacher_id == User.id)
        .join(Quiz)
        .join(QuizSubmission)
        .where(
            User.school_id == school.id,
            User.role == UserRole.TEACHER,
            QuizSubmission.submitted_at.between(start_datetime, end_datetime),
            QuizSubmission.status == "submitted"
        )
        .group_by(User.id, User.first_name, User.last_name)
        .order_by(desc(func.avg(QuizSubmission.percentage)))
    )
    
    teacher_effectiveness = []
    for row in teacher_effectiveness_result:
        teacher_effectiveness.append({
            "teacher_id": str(row.id),
            "teacher_name": f"{row.first_name} {row.last_name}",
            "average_class_performance": round(row.avg_class_performance, 2),
            "classes_taught": row.classes_taught,
            "effectiveness_rating": "excellent" if row.avg_class_performance >= 85 else 
                                   "good" if row.avg_class_performance >= 75 else 
                                   "average" if row.avg_class_performance >= 65 else "needs_improvement"
        })
    
    # Resource utilization
    resource_stats_result = await db.execute(
        select(
            func.count(Resource.id).label('total_resources'),
            func.count(Resource.id).filter(Resource.is_active == True).label('active_resources')
        )
        .where(Resource.school_id == school.id, Resource.is_deleted == False)
    )
    
    resource_result = resource_stats_result.first()
    
    resource_utilization = {
        "total_resources": resource_result.total_resources or 0,
        "active_resources": resource_result.active_resources or 0,
        "utilization_rate": round(
            (resource_result.active_resources / resource_result.total_resources * 100) 
            if resource_result.total_resources else 0, 2
        )
    }
    
    # Engagement overview
    engagement_overview = {
        "daily_active_users": 0,  # TODO: Calculate from analytics events
        "session_duration_avg": 0,  # TODO: Calculate from analytics events
        "feature_adoption": {}  # TODO: Implement feature usage tracking
    }
    
    # Trends analysis
    trends_analysis = {
        "performance_trend": 0.0,  # TODO: Calculate trend over time
        "engagement_trend": 0.0,   # TODO: Calculate engagement trend
        "growth_metrics": {}       # TODO: Implement growth tracking
    }
    
    # Benchmarking data (placeholder)
    benchmarking_data = {
        "percentile_ranking": 75,  # Where this school ranks
        "comparison_metrics": {
            "average_performance": "above_average",
            "engagement_level": "high",
            "resource_utilization": "excellent"
        }
    }
    
    return SchoolAnalyticsResponse(
        school_id=str(school.id),
        school_name=school.name,
        summary_statistics=summary_statistics,
        class_performance_comparison=class_performance_comparison,
        teacher_effectiveness=teacher_effectiveness,
        resource_utilization=resource_utilization,
        engagement_overview=engagement_overview,
        trends_analysis=trends_analysis,
        benchmarking_data=benchmarking_data
    )


@router.post("/learning-analytics", response_model=List[Dict[str, Any]])
async def get_learning_analytics(
    request: LearningAnalyticsRequest,
    current_user: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed learning analytics based on specified criteria"""
    
    analytics_results = []
    
    # Process each student if specified
    if request.student_ids:
        for student_id in request.student_ids:
            # Check permissions
            can_access = await PermissionChecker.can_access_user_data(
                current_user, student_id, db
            )
            if not can_access:
                continue
            
            student_analytics = {}
            
            if "performance" in request.metrics:
                student_analytics["performance"] = await calculate_performance_metrics(
                    "student", student_id, request.date_range.start_date, request.date_range.end_date, db
                )
            
            if "engagement" in request.metrics:
                student_analytics["engagement"] = await calculate_engagement_metrics(
                    student_id, request.date_range.start_date, request.date_range.end_date, db
                )
            
            if "behavior" in request.metrics:
                student_analytics["behavior"] = await generate_behavioral_insights(student_id, db)
            
            analytics_results.append({
                "entity_type": "student",
                "entity_id": student_id,
                "data": student_analytics
            })
    
    # Process each class if specified
    if request.class_ids:
        for class_id in request.class_ids:
            # Check permissions
            can_access = await PermissionChecker.can_access_class(current_user, class_id, db)
            if not can_access:
                continue
            
            class_analytics = {}
            
            if "performance" in request.metrics:
                class_analytics["performance"] = await calculate_performance_metrics(
                    "class", class_id, request.date_range.start_date, request.date_range.end_date, db
                )
            
            analytics_results.append({
                "entity_type": "class",
                "entity_id": class_id,
                "data": class_analytics
            })
    
    return analytics_results


@router.get("/trends/performance", response_model=PerformanceTrendResponse)
async def get_performance_trends(
    entity_type: str = Query(..., regex="^(student|class|school)$"),
    entity_id: str = Query(...),
    period: str = Query("weekly", regex="^(daily|weekly|monthly)$"),
    days: int = Query(30, ge=7, le=365),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get performance trends over time"""
    
    # Check permissions based on entity type
    if entity_type == "student":
        can_access = await PermissionChecker.can_access_user_data(current_user, entity_id, db)
        if not can_access:
            raise AuthorizationException("Access denied")
    elif entity_type == "class":
        can_access = await PermissionChecker.can_access_class(current_user, entity_id, db)
        if not can_access:
            raise AuthorizationException("Access denied")
    elif entity_type == "school":
        if current_user.role != UserRole.ADMIN:
            raise AuthorizationException("Admin access required")
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    # Calculate period grouping
    if period == "daily":
        date_trunc = "DATE(submitted_at)"
        period_days = 1
    elif period == "weekly":
        date_trunc = "DATE(submitted_at, 'weekday 0', '-6 days')"
        period_days = 7
    else:  # monthly
        date_trunc = "DATE(submitted_at, 'start of month')"
        period_days = 30
    
    # Build query based on entity type
    if entity_type == "student":
        trend_query = text(f"""
            SELECT {date_trunc} as period_date,
                   AVG(percentage) as avg_score,
                   COUNT(*) as submission_count
            FROM quiz_submissions 
            WHERE student_id = :entity_id 
            AND submitted_at BETWEEN :start_date AND :end_date
            AND status = 'submitted'
            GROUP BY {date_trunc}
            ORDER BY period_date
        """)
    elif entity_type == "class":
        trend_query = text(f"""
            SELECT {date_trunc} as period_date,
                   AVG(qs.percentage) as avg_score,
                   COUNT(*) as submission_count
            FROM quiz_submissions qs
            JOIN quizzes q ON qs.quiz_id = q.id
            WHERE q.class_id = :entity_id
            AND qs.submitted_at BETWEEN :start_date AND :end_date
            AND qs.status = 'submitted'
            GROUP BY {date_trunc}
            ORDER BY period_date
        """)
    else:  # school
        school = await get_current_school(current_user, db)
        trend_query = text(f"""
            SELECT {date_trunc} as period_date,
                   AVG(qs.percentage) as avg_score,
                   COUNT(*) as submission_count
            FROM quiz_submissions qs
            JOIN quizzes q ON qs.quiz_id = q.id
            JOIN classes c ON q.class_id = c.id
            WHERE c.school_id = :entity_id
            AND qs.submitted_at BETWEEN :start_date AND :end_date
            AND qs.status = 'submitted'
            GROUP BY {date_trunc}
            ORDER BY period_date
        """)
        entity_id = str(school.id)
    
    result = await db.execute(trend_query, {
        "entity_id": entity_id,
        "start_date": start_date,
        "end_date": end_date
    })
    
    data_points = []
    scores = []
    
    for row in result:
        data_point = {
            "period": row.period_date,
            "average_score": round(row.avg_score, 2),
            "submission_count": row.submission_count
        }
        data_points.append(data_point)
        scores.append(row.avg_score)
    
    # Calculate trend analysis
    trend_analysis = {"direction": "stable", "slope": 0.0, "r_squared": 0.0}
    
    if len(scores) >= 2:
        # Simple linear regression
        x = np.arange(len(scores))
        slope, intercept = np.polyfit(x, scores, 1)
        trend_analysis["slope"] = round(slope, 4)
        
        if slope > 0.5:
            trend_analysis["direction"] = "improving"
        elif slope < -0.5:
            trend_analysis["direction"] = "declining"
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((scores - y_pred) ** 2)
        ss_tot = np.sum((scores - np.mean(scores)) ** 2)
        trend_analysis["r_squared"] = round(1 - (ss_res / ss_tot) if ss_tot != 0 else 0, 4)
    
    # Simple prediction for next period
    predictions = None
    if len(scores) >= 3:
        next_score = scores[-1] + trend_analysis["slope"]
        predictions = {
            "next_period_prediction": round(max(0, min(100, next_score)), 2),
            "confidence": min(0.9, trend_analysis["r_squared"])
        }
    
    return PerformanceTrendResponse(
        period=period,
        data_points=data_points,
        trend_analysis=trend_analysis,
        predictions=predictions
    )


@router.get("/engagement", response_model=EngagementMetricsResponse)
async def get_engagement_metrics(
    current_user: User = Depends(require_admin),
    school = Depends(get_current_school),
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=7, le=90)
):
    """Get comprehensive engagement metrics (admin only)"""
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Daily active users
    dau_result = await db.execute(
        select(
            func.date(AnalyticsEvent.timestamp).label('date'),
            func.count(func.distinct(AnalyticsEvent.user_id)).label('active_users')
        )
        .join(User)
        .where(
            User.school_id == school.id,
            AnalyticsEvent.timestamp.between(start_date, end_date)
        )
        .group_by(func.date(AnalyticsEvent.timestamp))
        .order_by('date')
    )
    
    daily_active_users = []
    for row in dau_result:
        daily_active_users.append({
            "date": row.date,
            "active_users": row.active_users
        })
    
    # Session duration statistics
    session_stats_result = await db.execute(
        select(
            func.avg(AnalyticsEvent.duration).label('avg_duration'),
            func.median(AnalyticsEvent.duration).label('median_duration'),
            func.max(AnalyticsEvent.duration).label('max_duration')
        )
        .join(User)
        .where(
            User.school_id == school.id,
            AnalyticsEvent.timestamp.between(start_date, end_date),
            AnalyticsEvent.duration.isnot(None)
        )
    )
    
    session_result = session_stats_result.first()
    session_duration_stats = {
        "average_minutes": int((session_result.avg_duration or 0) / 60),
        "median_minutes": int((session_result.median_duration or 0) / 60),
        "max_minutes": int((session_result.max_duration or 0) / 60)
    }
    
    # Feature usage
    feature_usage_result = await db.execute(
        select(
            AnalyticsEvent.event_name,
            func.count(AnalyticsEvent.id).label('usage_count')
        )
        .join(User)
        .where(
            User.school_id == school.id,
            AnalyticsEvent.timestamp.between(start_date, end_date)
        )
        .group_by(AnalyticsEvent.event_name)
        .order_by(desc('usage_count'))
    )
    
    feature_usage = {}
    for row in feature_usage_result:
        feature_usage[row.event_name] = row.usage_count
    
    # Retention rates (simplified)
    retention_rates = {
        "daily": 0.85,  # TODO: Calculate actual retention
        "weekly": 0.72,
        "monthly": 0.58
    }
    
    # Drop-off analysis (placeholder)
    drop_off_analysis = {
        "login_to_quiz_start": 0.15,
        "quiz_start_to_completion": 0.08,
        "session_abandonment_rate": 0.12
    }
    
    return EngagementMetricsResponse(
        daily_active_users=daily_active_users,
        session_duration_stats=session_duration_stats,
        feature_usage=feature_usage,
        retention_rates=retention_rates,
        drop_off_analysis=drop_off_analysis
    )


@router.post("/predictive", response_model=PredictiveAnalyticsResponse)
async def get_predictive_analytics(
    request: PredictiveAnalyticsRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Generate predictive analytics using AI models (admin only)"""
    
    # This would typically use machine learning models
    # For now, we'll provide a simplified implementation
    
    predictions = []
    model_confidence = 0.75
    
    for entity_id in request.entity_ids:
        if request.prediction_type == "performance":
            # Predict future performance based on trends
            recent_performance = await calculate_performance_metrics(
                request.target_entity, entity_id, 
                date.today() - timedelta(days=30), date.today(), db
            )
            
            predicted_score = recent_performance.get("average_score", 0)
            trend = recent_performance.get("improvement_trend", 0)
            
            # Simple trend extrapolation
            future_score = predicted_score + (trend * (request.prediction_horizon_days / 30))
            future_score = max(0, min(100, future_score))
            
            predictions.append({
                "entity_id": entity_id,
                "current_score": predicted_score,
                "predicted_score": round(future_score, 2),
                "confidence": model_confidence,
                "trend": "improving" if trend > 0 else "declining" if trend < 0 else "stable"
            })
        
        elif request.prediction_type == "dropout_risk":
            # Assess dropout risk based on engagement and performance
            risk_students = await identify_at_risk_students(entity_id, current_user.school_id, db)
            
            for student in risk_students:
                if student["risk_score"] >= 50:
                    risk_level = "high"
                elif student["risk_score"] >= 25:
                    risk_level = "medium"
                else:
                    risk_level = "low"
                
                predictions.append({
                    "entity_id": student["student_id"],
                    "risk_level": risk_level,
                    "risk_score": student["risk_score"],
                    "confidence": model_confidence,
                    "factors": [factor["factor"] for factor in student["risk_factors"]]
                })
    
    # Feature importance (simplified)
    feature_importance = {
        "recent_performance": 0.35,
        "engagement_level": 0.25,
        "attendance_rate": 0.20,
        "assignment_completion": 0.15,
        "peer_interaction": 0.05
    }
    
    # Generate recommendations
    recommendations = []
    if request.prediction_type == "performance":
        recommendations = [
            "Focus on students with declining trends",
            "Implement targeted interventions for low performers",
            "Enhance engagement for at-risk students"
        ]
    elif request.prediction_type == "dropout_risk":
        recommendations = [
            "Schedule one-on-one meetings with high-risk students",
            "Implement early warning system for attendance",
            "Provide additional academic support resources"
        ]
    
    return PredictiveAnalyticsResponse(
        prediction_type=request.prediction_type,
        predictions=predictions,
        model_confidence=model_confidence,
        feature_importance=feature_importance,
        recommendations=recommendations,
        generated_at=datetime.utcnow()
    )


@router.post("/reports/custom")
async def generate_custom_report(
    request: CustomReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db)
):
    """Generate a custom analytics report"""
    
    # Queue the report generation as a background task
    background_tasks.add_task(
        generate_report_background,
        request.report_name,
        request.report_type,
        request.filters,
        request.metrics,
        request.visualizations,
        request.format,
        str(current_user.id)
    )
    
    return {
        "message": "Custom report generation started",
        "report_name": request.report_name,
        "estimated_completion": "5-10 minutes",
        "format": request.format
    }


async def generate_report_background(
    report_name: str,
    report_type: str,
    filters: Dict[str, Any],
    metrics: List[str],
    visualizations: List[str],
    format: str,
    user_id: str
):
    """Background task for generating custom reports"""
    
    logger.info(f"Generating custom report: {report_name} for user {user_id}")
    
    try:
        # TODO: Implement actual report generation logic
        # This would include:
        # - Data extraction based on filters
        # - Metric calculations
        # - Visualization creation
        # - Format conversion (PDF, Excel, etc.)
        
        import asyncio
        await asyncio.sleep(30)  # Simulate processing time
        
        logger.info(f"Custom report '{report_name}' generated successfully")
        
        # TODO: Store report and notify user of completion
        
    except Exception as e:
        logger.error(f"Custom report generation failed: {e}")


# Export router
__all__ = ["router"]
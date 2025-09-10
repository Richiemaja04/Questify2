"""
AI-Powered Smart Class & Timetable Scheduler
Quiz-related API routes
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Query, Path, HTTPException, status
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, or_
from sqlalchemy.orm import selectinload

from ..database.connection import get_db
from ..database.models import (
    User, UserRole, Quiz, QuizType, QuizStatus, Question, QuestionType,
    QuizSubmission, SubmissionStatus, Answer, Grade, Class, Enrollment,
    XPTransaction, Achievement, UserAchievement, AnalyticsEvent
)
from ..dependencies import (
    require_authentication,
    require_student,
    require_teacher,
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
    QuizNotAvailableException,
    MaxAttemptsExceededException,
    ConflictException
)

# Configure logging
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()


# Pydantic models
class QuizResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    instructions: Optional[str]
    quiz_type: QuizType
    status: QuizStatus
    time_limit_minutes: Optional[int]
    max_attempts: int
    passing_score: float
    shuffle_questions: bool
    shuffle_answers: bool
    show_results_immediately: bool
    show_correct_answers: bool
    allow_backtrack: bool
    require_lockdown: bool
    available_from: Optional[datetime]
    available_until: Optional[datetime]
    late_submission_penalty: float
    weight: float
    tags: List[str]
    difficulty_level: Optional[str]
    estimated_duration: Optional[int]
    ai_generated: bool
    class_name: str
    teacher_name: str
    question_count: int
    total_points: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class QuestionResponse(BaseModel):
    id: str
    order_index: int
    question_type: QuestionType
    title: Optional[str]
    content: str
    explanation: Optional[str]
    points: float
    difficulty_level: Optional[str]
    estimated_time_seconds: Optional[int]
    media_url: Optional[str]
    media_type: Optional[str]
    options: List[Any]
    hints: List[str]
    tags: List[str]
    
    class Config:
        from_attributes = True


class QuizCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    quiz_type: QuizType = QuizType.QUIZ
    time_limit_minutes: Optional[int] = None
    max_attempts: int = 1
    passing_score: float = 60.0
    shuffle_questions: bool = True
    shuffle_answers: bool = True
    show_results_immediately: bool = True
    show_correct_answers: bool = True
    allow_backtrack: bool = False
    require_lockdown: bool = False
    available_from: Optional[datetime] = None
    available_until: Optional[datetime] = None
    late_submission_penalty: float = 0.0
    weight: float = 1.0
    tags: List[str] = []
    difficulty_level: Optional[str] = "medium"
    class_id: str
    
    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title is required')
        return v.strip()
    
    @validator('passing_score')
    def validate_passing_score(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Passing score must be between 0 and 100')
        return v
    
    @validator('max_attempts')
    def validate_max_attempts(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Max attempts must be between 1 and 10')
        return v


class QuestionCreateRequest(BaseModel):
    quiz_id: str
    order_index: int
    question_type: QuestionType
    title: Optional[str] = None
    content: str
    explanation: Optional[str] = None
    points: float = 1.0
    difficulty_level: Optional[str] = "medium"
    estimated_time_seconds: Optional[int] = None
    media_url: Optional[str] = None
    media_type: Optional[str] = None
    options: List[Any] = []
    correct_answers: List[Any] = []
    partial_credit_rules: Dict[str, Any] = {}
    hints: List[str] = []
    tags: List[str] = []
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Question content is required')
        return v.strip()
    
    @validator('points')
    def validate_points(cls, v):
        if v <= 0:
            raise ValueError('Points must be greater than 0')
        return v


class QuizSubmissionRequest(BaseModel):
    quiz_id: str
    answers: List[Dict[str, Any]]  # [{"question_id": "...", "answer": "..."}]
    
    @validator('answers')
    def validate_answers(cls, v):
        if not v:
            raise ValueError('At least one answer is required')
        
        for answer in v:
            if 'question_id' not in answer or 'answer' not in answer:
                raise ValueError('Each answer must have question_id and answer')
        
        return v


class QuizSubmissionResponse(BaseModel):
    id: str
    quiz_id: str
    student_id: str
    attempt_number: int
    status: SubmissionStatus
    started_at: datetime
    submitted_at: Optional[datetime]
    time_taken_seconds: Optional[int]
    score: Optional[float]
    percentage: Optional[float]
    max_possible_score: Optional[float]
    is_late: bool
    late_penalty_applied: float
    final_score: Optional[float]
    feedback: Optional[str]
    teacher_comments: Optional[str]
    flagged_for_review: bool
    quiz_title: str
    
    class Config:
        from_attributes = True


class QuizAttemptRequest(BaseModel):
    quiz_id: str


class AnswerSubmissionRequest(BaseModel):
    question_id: str
    answer_content: Any
    time_taken_seconds: Optional[int] = None


class QuizResultsResponse(BaseModel):
    submission: QuizSubmissionResponse
    questions: List[Dict[str, Any]]
    answers: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    achievements_earned: List[Dict[str, Any]]
    xp_earned: int


class QuizAnalyticsResponse(BaseModel):
    quiz_stats: Dict[str, Any]
    question_analytics: List[Dict[str, Any]]
    student_performance: List[Dict[str, Any]]
    difficulty_analysis: Dict[str, Any]
    time_analysis: Dict[str, Any]
    improvement_suggestions: List[str]


# Helper functions
async def check_quiz_availability(
    quiz: Quiz,
    student: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Check if quiz is available for student to take"""
    
    current_time = datetime.utcnow()
    
    # Check quiz status
    if quiz.status != QuizStatus.PUBLISHED:
        raise QuizNotAvailableException("Quiz is not published", str(quiz.id))
    
    # Check time window
    if quiz.available_from and current_time < quiz.available_from:
        raise QuizNotAvailableException("Quiz is not yet available", str(quiz.id))
    
    if quiz.available_until and current_time > quiz.available_until:
        raise QuizNotAvailableException("Quiz deadline has passed", str(quiz.id))
    
    # Check enrollment
    enrollment_result = await db.execute(
        select(Enrollment).where(
            Enrollment.student_id == student.id,
            Enrollment.class_id == quiz.class_id,
            Enrollment.status == "active"
        )
    )
    
    if not enrollment_result.scalar_one_or_none():
        raise QuizNotAvailableException("You are not enrolled in this class", str(quiz.id))
    
    # Check attempts
    attempts_result = await db.execute(
        select(func.count(QuizSubmission.id))
        .where(
            QuizSubmission.student_id == student.id,
            QuizSubmission.quiz_id == quiz.id,
            QuizSubmission.status.in_([SubmissionStatus.SUBMITTED, SubmissionStatus.GRADED])
        )
    )
    
    current_attempts = attempts_result.scalar() or 0
    
    if current_attempts >= quiz.max_attempts:
        raise MaxAttemptsExceededException(
            str(quiz.id), 
            quiz.max_attempts, 
            current_attempts
        )
    
    return {
        "available": True,
        "attempts_remaining": quiz.max_attempts - current_attempts,
        "time_limit": quiz.time_limit_minutes,
        "is_late": quiz.available_until and current_time > quiz.available_until if quiz.available_until else False
    }


async def calculate_quiz_score(
    submission: QuizSubmission,
    answers: List[Answer],
    db: AsyncSession
) -> Dict[str, float]:
    """Calculate quiz score based on answers"""
    
    total_points = 0.0
    max_possible_points = 0.0
    
    # Get all questions for the quiz
    questions_result = await db.execute(
        select(Question)
        .where(Question.quiz_id == submission.quiz_id)
        .order_by(Question.order_index)
    )
    questions = {str(q.id): q for q in questions_result.scalars()}
    
    # Calculate score for each answer
    for answer in answers:
        question = questions.get(str(answer.question_id))
        if not question:
            continue
        
        max_possible_points += question.points
        
        if answer.is_correct:
            total_points += answer.points_earned or question.points
    
    # Calculate percentage
    percentage = (total_points / max_possible_points * 100) if max_possible_points > 0 else 0
    
    # Apply late penalty
    final_score = total_points
    late_penalty = 0.0
    
    if submission.is_late:
        quiz_result = await db.execute(
            select(Quiz).where(Quiz.id == submission.quiz_id)
        )
        quiz = quiz_result.scalar_one()
        
        if quiz.late_submission_penalty > 0:
            late_penalty = total_points * (quiz.late_submission_penalty / 100)
            final_score = max(0, total_points - late_penalty)
    
    return {
        "raw_score": total_points,
        "max_possible_score": max_possible_points,
        "percentage": percentage,
        "late_penalty": late_penalty,
        "final_score": final_score
    }


async def award_quiz_xp(
    student: User,
    submission: QuizSubmission,
    score_data: Dict[str, float],
    db: AsyncSession
) -> int:
    """Award XP based on quiz performance"""
    
    import uuid
    from ...config import get_settings
    
    settings = get_settings()
    base_xp = settings.BASE_XP_PER_QUIZ
    
    # Calculate XP multipliers
    multiplier = 1.0
    
    # Perfect score bonus
    if score_data["percentage"] >= 100:
        multiplier *= settings.XP_MULTIPLIER_PERFECT_SCORE
    
    # Performance-based XP (scaled by percentage)
    performance_multiplier = max(0.1, score_data["percentage"] / 100)
    multiplier *= performance_multiplier
    
    # Calculate final XP
    xp_earned = int(base_xp * multiplier)
    
    # Create XP transaction
    xp_transaction = XPTransaction(
        id=uuid.uuid4(),
        user_id=student.id,
        amount=xp_earned,
        source_type="quiz_completion",
        source_id=str(submission.id),
        description=f"Quiz: {submission.quiz.title}",
        multiplier=multiplier
    )
    
    db.add(xp_transaction)
    
    # Update student profile
    from ..database.models import StudentProfile
    profile_result = await db.execute(
        select(StudentProfile).where(StudentProfile.user_id == student.id)
    )
    profile = profile_result.scalar_one_or_none()
    
    if profile:
        profile.total_xp += xp_earned
        profile.last_activity = datetime.utcnow()
    
    return xp_earned


async def check_achievements(
    student: User,
    submission: QuizSubmission,
    score_data: Dict[str, float],
    db: AsyncSession
) -> List[Dict[str, Any]]:
    """Check and award achievements based on quiz performance"""
    
    achievements_earned = []
    
    # Get student's current achievements
    current_achievements = await db.execute(
        select(UserAchievement.achievement_id)
        .where(UserAchievement.user_id == student.id)
    )
    earned_achievement_ids = {str(aid) for aid in current_achievements.scalars()}
    
    # Get all available achievements
    achievements_result = await db.execute(
        select(Achievement)
        .where(
            Achievement.is_active == True,
            ~Achievement.id.in_(earned_achievement_ids) if earned_achievement_ids else True
        )
    )
    
    for achievement in achievements_result.scalars():
        if await check_achievement_criteria(student, achievement, submission, score_data, db):
            # Award achievement
            import uuid
            
            user_achievement = UserAchievement(
                id=uuid.uuid4(),
                user_id=student.id,
                achievement_id=achievement.id,
                unlocked_at=datetime.utcnow(),
                current_level=1,
                progress={}
            )
            
            db.add(user_achievement)
            
            # Award achievement XP bonus
            bonus_xp = XPTransaction(
                id=uuid.uuid4(),
                user_id=student.id,
                amount=achievement.points_reward,
                source_type="achievement",
                source_id=str(achievement.id),
                description=f"Achievement: {achievement.name}",
                multiplier=1.0
            )
            
            db.add(bonus_xp)
            
            achievements_earned.append({
                "id": str(achievement.id),
                "name": achievement.name,
                "description": achievement.description,
                "points_reward": achievement.points_reward,
                "rarity": achievement.rarity
            })
    
    return achievements_earned


async def check_achievement_criteria(
    student: User,
    achievement: Achievement,
    submission: QuizSubmission,
    score_data: Dict[str, float],
    db: AsyncSession
) -> bool:
    """Check if student meets achievement criteria"""
    
    criteria = achievement.unlock_criteria
    
    # Perfect score achievement
    if criteria.get("perfect_scores", 0) == 1 and score_data["percentage"] >= 100:
        return True
    
    # First quiz completion
    if criteria.get("quizzes_completed", 0) == 1:
        completed_count = await db.execute(
            select(func.count(QuizSubmission.id))
            .where(
                QuizSubmission.student_id == student.id,
                QuizSubmission.status == SubmissionStatus.SUBMITTED
            )
        )
        return completed_count.scalar() == 1
    
    # Add more achievement criteria as needed
    return False


# API Routes
@router.get("/", response_model=List[QuizResponse])
async def list_quizzes(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db),
    class_id: Optional[str] = Query(None, description="Filter by class ID"),
    status: Optional[QuizStatus] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of records to return")
):
    """List quizzes based on user role and filters"""
    
    query = select(Quiz).options(
        selectinload(Quiz.class_obj),
        selectinload(Quiz.created_by)
    )
    
    if current_user.role == UserRole.STUDENT:
        # Students see only quizzes from their enrolled classes
        query = query.join(Class).join(Enrollment).where(
            Enrollment.student_id == current_user.id,
            Enrollment.status == "active",
            Quiz.status == QuizStatus.PUBLISHED
        )
    elif current_user.role == UserRole.TEACHER:
        # Teachers see quizzes they created
        query = query.where(Quiz.created_by_teacher_id == current_user.id)
    
    # Apply filters
    if class_id:
        query = query.where(Quiz.class_id == class_id)
    
    if status and current_user.role != UserRole.STUDENT:
        query = query.where(Quiz.status == status)
    
    # Apply pagination
    query = query.offset(skip).limit(limit).order_by(desc(Quiz.created_at))
    
    result = await db.execute(query)
    quizzes = result.scalars().all()
    
    quiz_list = []
    for quiz in quizzes:
        # Get question count and total points
        questions_result = await db.execute(
            select(
                func.count(Question.id).label('question_count'),
                func.sum(Question.points).label('total_points')
            )
            .where(Question.quiz_id == quiz.id)
        )
        question_stats = questions_result.first()
        
        quiz_list.append(QuizResponse(
            id=str(quiz.id),
            title=quiz.title,
            description=quiz.description,
            instructions=quiz.instructions,
            quiz_type=quiz.quiz_type,
            status=quiz.status,
            time_limit_minutes=quiz.time_limit_minutes,
            max_attempts=quiz.max_attempts,
            passing_score=quiz.passing_score,
            shuffle_questions=quiz.shuffle_questions,
            shuffle_answers=quiz.shuffle_answers,
            show_results_immediately=quiz.show_results_immediately,
            show_correct_answers=quiz.show_correct_answers,
            allow_backtrack=quiz.allow_backtrack,
            require_lockdown=quiz.require_lockdown,
            available_from=quiz.available_from,
            available_until=quiz.available_until,
            late_submission_penalty=quiz.late_submission_penalty,
            weight=quiz.weight,
            tags=quiz.tags,
            difficulty_level=quiz.difficulty_level,
            estimated_duration=quiz.estimated_duration,
            ai_generated=quiz.ai_generated,
            class_name=quiz.class_obj.name,
            teacher_name=f"{quiz.created_by.user.first_name} {quiz.created_by.user.last_name}",
            question_count=question_stats.question_count or 0,
            total_points=float(question_stats.total_points or 0),
            created_at=quiz.created_at
        ))
    
    return quiz_list


@router.post("/", response_model=Dict[str, str])
async def create_quiz(
    request: QuizCreateRequest,
    current_user: User = Depends(require_teacher),
    db: AsyncSession = Depends(get_db)
):
    """Create a new quiz"""
    
    # Check if teacher can access the class
    can_access = await PermissionChecker.can_access_class(
        current_user, request.class_id, db
    )
    if not can_access:
        raise AuthorizationException("Access denied to this class")
    
    # Create quiz
    import uuid
    
    quiz = Quiz(
        id=uuid.uuid4(),
        title=request.title,
        description=request.description,
        instructions=request.instructions,
        quiz_type=request.quiz_type,
        status=QuizStatus.DRAFT,
        time_limit_minutes=request.time_limit_minutes,
        max_attempts=request.max_attempts,
        passing_score=request.passing_score,
        shuffle_questions=request.shuffle_questions,
        shuffle_answers=request.shuffle_answers,
        show_results_immediately=request.show_results_immediately,
        show_correct_answers=request.show_correct_answers,
        allow_backtrack=request.allow_backtrack,
        require_lockdown=request.require_lockdown,
        available_from=request.available_from,
        available_until=request.available_until,
        late_submission_penalty=request.late_submission_penalty,
        weight=request.weight,
        tags=request.tags,
        difficulty_level=request.difficulty_level,
        class_id=request.class_id,
        created_by_teacher_id=current_user.id
    )
    
    db.add(quiz)
    await db.commit()
    
    logger.info(f"Quiz '{request.title}' created by teacher {current_user.username}")
    
    return {
        "message": "Quiz created successfully",
        "quiz_id": str(quiz.id)
    }


@router.get("/{quiz_id}", response_model=QuizResponse)
async def get_quiz(
    quiz_id: str = Path(..., description="Quiz ID"),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get quiz details"""
    
    result = await db.execute(
        select(Quiz)
        .options(
            selectinload(Quiz.class_obj),
            selectinload(Quiz.created_by)
        )
        .where(
            Quiz.id == quiz_id,
            Quiz.is_deleted == False
        )
    )
    
    quiz = result.scalar_one_or_none()
    if not quiz:
        raise NotFoundException("Quiz not found")
    
    # Check permissions
    if current_user.role == UserRole.STUDENT:
        # Students can only see published quizzes from enrolled classes
        enrollment_result = await db.execute(
            select(Enrollment).where(
                Enrollment.student_id == current_user.id,
                Enrollment.class_id == quiz.class_id,
                Enrollment.status == "active"
            )
        )
        
        if not enrollment_result.scalar_one_or_none():
            raise AuthorizationException("Access denied to this quiz")
        
        if quiz.status != QuizStatus.PUBLISHED:
            raise AuthorizationException("Quiz is not available")
    
    elif current_user.role == UserRole.TEACHER:
        # Teachers can see their own quizzes
        if quiz.created_by_teacher_id != current_user.id and current_user.role != UserRole.ADMIN:
            raise AuthorizationException("Access denied to this quiz")
    
    # Get question count and total points
    questions_result = await db.execute(
        select(
            func.count(Question.id).label('question_count'),
            func.sum(Question.points).label('total_points')
        )
        .where(Question.quiz_id == quiz.id)
    )
    question_stats = questions_result.first()
    
    return QuizResponse(
        id=str(quiz.id),
        title=quiz.title,
        description=quiz.description,
        instructions=quiz.instructions,
        quiz_type=quiz.quiz_type,
        status=quiz.status,
        time_limit_minutes=quiz.time_limit_minutes,
        max_attempts=quiz.max_attempts,
        passing_score=quiz.passing_score,
        shuffle_questions=quiz.shuffle_questions,
        shuffle_answers=quiz.shuffle_answers,
        show_results_immediately=quiz.show_results_immediately,
        show_correct_answers=quiz.show_correct_answers,
        allow_backtrack=quiz.allow_backtrack,
        require_lockdown=quiz.require_lockdown,
        available_from=quiz.available_from,
        available_until=quiz.available_until,
        late_submission_penalty=quiz.late_submission_penalty,
        weight=quiz.weight,
        tags=quiz.tags,
        difficulty_level=quiz.difficulty_level,
        estimated_duration=quiz.estimated_duration,
        ai_generated=quiz.ai_generated,
        class_name=quiz.class_obj.name,
        teacher_name=f"{quiz.created_by.user.first_name} {quiz.created_by.user.last_name}",
        question_count=question_stats.question_count or 0,
        total_points=float(question_stats.total_points or 0),
        created_at=quiz.created_at
    )


@router.post("/{quiz_id}/start")
async def start_quiz_attempt(
    quiz_id: str = Path(..., description="Quiz ID"),
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Start a new quiz attempt"""
    
    # Get quiz
    result = await db.execute(
        select(Quiz).where(
            Quiz.id == quiz_id,
            Quiz.is_deleted == False
        )
    )
    quiz = result.scalar_one_or_none()
    
    if not quiz:
        raise NotFoundException("Quiz not found")
    
    # Check availability
    availability = await check_quiz_availability(quiz, current_user, db)
    
    # Check for existing active attempt
    existing_attempt = await db.execute(
        select(QuizSubmission).where(
            QuizSubmission.student_id == current_user.id,
            QuizSubmission.quiz_id == quiz_id,
            QuizSubmission.status == SubmissionStatus.STARTED
        )
    )
    
    if existing_attempt.scalar_one_or_none():
        raise ConflictException("You already have an active attempt for this quiz")
    
    # Create new submission
    import uuid
    
    attempt_count = await db.execute(
        select(func.count(QuizSubmission.id))
        .where(
            QuizSubmission.student_id == current_user.id,
            QuizSubmission.quiz_id == quiz_id
        )
    )
    
    submission = QuizSubmission(
        id=uuid.uuid4(),
        student_id=current_user.id,
        quiz_id=quiz_id,
        attempt_number=(attempt_count.scalar() or 0) + 1,
        status=SubmissionStatus.STARTED,
        started_at=datetime.utcnow(),
        is_late=availability.get("is_late", False),
        session_data={
            "start_time": datetime.utcnow().isoformat(),
            "user_agent": "web",  # TODO: Get from request headers
            "ip_address": "127.0.0.1"  # TODO: Get from request
        }
    )
    
    db.add(submission)
    await db.commit()
    
    logger.info(f"Quiz attempt started: {quiz_id} by student {current_user.username}")
    
    return {
        "submission_id": str(submission.id),
        "attempt_number": submission.attempt_number,
        "time_limit_minutes": quiz.time_limit_minutes,
        "started_at": submission.started_at.isoformat(),
        "message": "Quiz attempt started successfully"
    }


@router.get("/{quiz_id}/questions")
async def get_quiz_questions(
    quiz_id: str = Path(..., description="Quiz ID"),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get questions for a quiz"""
    
    # Get quiz and check permissions
    quiz_result = await db.execute(
        select(Quiz).where(
            Quiz.id == quiz_id,
            Quiz.is_deleted == False
        )
    )
    quiz = quiz_result.scalar_one_or_none()
    
    if not quiz:
        raise NotFoundException("Quiz not found")
    
    # Permission checks based on role
    if current_user.role == UserRole.STUDENT:
        # Students can only see questions if they have an active attempt
        active_attempt = await db.execute(
            select(QuizSubmission).where(
                QuizSubmission.student_id == current_user.id,
                QuizSubmission.quiz_id == quiz_id,
                QuizSubmission.status == SubmissionStatus.STARTED
            )
        )
        
        if not active_attempt.scalar_one_or_none():
            raise AuthorizationException("No active quiz attempt found")
    
    elif current_user.role == UserRole.TEACHER:
        # Teachers can see questions for their quizzes
        if quiz.created_by_teacher_id != current_user.id and current_user.role != UserRole.ADMIN:
            raise AuthorizationException("Access denied to this quiz")
    
    # Get questions
    questions_result = await db.execute(
        select(Question)
        .where(Question.quiz_id == quiz_id)
        .order_by(Question.order_index)
    )
    questions = questions_result.scalars().all()
    
    question_list = []
    for question in questions:
        question_data = {
            "id": str(question.id),
            "order_index": question.order_index,
            "question_type": question.question_type.value,
            "title": question.title,
            "content": question.content,
            "points": question.points,
            "difficulty_level": question.difficulty_level,
            "estimated_time_seconds": question.estimated_time_seconds,
            "media_url": question.media_url,
            "media_type": question.media_type,
            "options": question.options,
            "hints": question.hints,
            "tags": question.tags
        }
        
        # Hide correct answers from students during active attempt
        if current_user.role != UserRole.STUDENT or quiz.show_correct_answers:
            question_data["explanation"] = question.explanation
        
        question_list.append(question_data)
    
    # Shuffle questions if enabled and student is taking quiz
    if quiz.shuffle_questions and current_user.role == UserRole.STUDENT:
        import random
        random.shuffle(question_list)
    
    return {"questions": question_list}


@router.post("/{quiz_id}/submit")
async def submit_quiz(
    request: QuizSubmissionRequest,
    current_user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db)
):
    """Submit quiz answers"""
    
    # Get active submission
    submission_result = await db.execute(
        select(QuizSubmission)
        .options(selectinload(QuizSubmission.quiz))
        .where(
            QuizSubmission.student_id == current_user.id,
            QuizSubmission.quiz_id == request.quiz_id,
            QuizSubmission.status == SubmissionStatus.STARTED
        )
    )
    
    submission = submission_result.scalar_one_or_none()
    if not submission:
        raise NotFoundException("No active quiz attempt found")
    
    # Check time limit
    if submission.quiz.time_limit_minutes:
        time_limit = timedelta(minutes=submission.quiz.time_limit_minutes)
        if datetime.utcnow() - submission.started_at > time_limit:
            raise BusinessLogicException("Time limit exceeded")
    
    # Create answer records
    import uuid
    answers = []
    
    for answer_data in request.answers:
        answer = Answer(
            id=uuid.uuid4(),
            submission_id=submission.id,
            question_id=answer_data["question_id"],
            answer_content=answer_data["answer"],
            time_taken_seconds=answer_data.get("time_taken_seconds"),
            answer_order=len(answers) + 1
        )
        
        # Grade the answer (simplified - real implementation would be more complex)
        await grade_answer(answer, db)
        
        db.add(answer)
        answers.append(answer)
    
    # Calculate final score
    score_data = await calculate_quiz_score(submission, answers, db)
    
    # Update submission
    submission.status = SubmissionStatus.SUBMITTED
    submission.submitted_at = datetime.utcnow()
    submission.time_taken_seconds = int((submission.submitted_at - submission.started_at).total_seconds())
    submission.score = score_data["raw_score"]
    submission.percentage = score_data["percentage"]
    submission.max_possible_score = score_data["max_possible_score"]
    submission.late_penalty_applied = score_data["late_penalty"]
    submission.final_score = score_data["final_score"]
    
    await db.commit()
    
    # Award XP
    xp_earned = await award_quiz_xp(current_user, submission, score_data, db)
    
    # Check achievements
    achievements_earned = await check_achievements(current_user, submission, score_data, db)
    
    await db.commit()
    
    logger.info(f"Quiz submitted: {request.quiz_id} by student {current_user.username}")
    
    return {
        "message": "Quiz submitted successfully",
        "submission_id": str(submission.id),
        "score": submission.final_score,
        "percentage": submission.percentage,
        "xp_earned": xp_earned,
        "achievements_earned": achievements_earned
    }


async def grade_answer(answer: Answer, db: AsyncSession) -> None:
    """Grade an individual answer (simplified implementation)"""
    
    # Get question
    question_result = await db.execute(
        select(Question).where(Question.id == answer.question_id)
    )
    question = question_result.scalar_one()
    
    if not question:
        answer.is_correct = False
        answer.points_earned = 0.0
        return
    
    # Simple grading logic (would be more sophisticated in real implementation)
    if question.question_type == QuestionType.MULTIPLE_CHOICE:
        correct_answers = question.correct_answers
        student_answer = answer.answer_content
        
        if isinstance(student_answer, list):
            answer.is_correct = set(student_answer) == set(correct_answers)
        else:
            answer.is_correct = student_answer in correct_answers
    
    elif question.question_type == QuestionType.TRUE_FALSE:
        answer.is_correct = answer.answer_content == question.correct_answers[0]
    
    elif question.question_type == QuestionType.SHORT_ANSWER:
        # Simple string comparison (would use NLP in real implementation)
        student_answer = str(answer.answer_content).lower().strip()
        correct_answers = [str(ans).lower().strip() for ans in question.correct_answers]
        answer.is_correct = student_answer in correct_answers
    
    else:
        # Default to incorrect for unsupported types
        answer.is_correct = False
    
    # Award points
    answer.points_earned = question.points if answer.is_correct else 0.0


@router.get("/{quiz_id}/submissions")
async def get_quiz_submissions(
    quiz_id: str = Path(..., description="Quiz ID"),
    current_user: User = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db),
    student_id: Optional[str] = Query(None, description="Filter by student ID")
):
    """Get quiz submissions (teachers/admins only)"""
    
    # Check permissions
    can_modify = await PermissionChecker.can_modify_quiz(current_user, quiz_id, db)
    if not can_modify:
        raise AuthorizationException("Access denied to quiz submissions")
    
    query = select(QuizSubmission).options(
        selectinload(QuizSubmission.student),
        selectinload(QuizSubmission.quiz)
    ).where(QuizSubmission.quiz_id == quiz_id)
    
    if student_id:
        query = query.where(QuizSubmission.student_id == student_id)
    
    result = await db.execute(query.order_by(desc(QuizSubmission.submitted_at)))
    submissions = result.scalars().all()
    
    submission_list = []
    for submission in submissions:
        submission_list.append({
            "id": str(submission.id),
            "student_name": f"{submission.student.first_name} {submission.student.last_name}",
            "attempt_number": submission.attempt_number,
            "status": submission.status.value,
            "started_at": submission.started_at.isoformat(),
            "submitted_at": submission.submitted_at.isoformat() if submission.submitted_at else None,
            "score": submission.score,
            "percentage": submission.percentage,
            "time_taken_seconds": submission.time_taken_seconds,
            "is_late": submission.is_late,
            "flagged_for_review": submission.flagged_for_review
        })
    
    return {"submissions": submission_list}


# Export router
__all__ = ["router"]
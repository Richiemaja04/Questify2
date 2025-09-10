"""
AI-Powered Smart Class & Timetable Scheduler
SQLAlchemy Database Models
"""

import enum
import uuid
from datetime import datetime, date, time as dt_time
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    Boolean, Column, DateTime, Date, Time, Integer, String, Text, Float, 
    ForeignKey, JSON, Enum, UniqueConstraint, Index, CheckConstraint,
    event, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import expression
from sqlalchemy.types import TypeDecorator, CHAR

# Base class for all models
Base = declarative_base()


class GUID(TypeDecorator):
    """Platform-independent GUID type"""
    impl = CHAR
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                return "%.32x" % value.int

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


# Enums
class UserRole(enum.Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    PARENT = "parent"


class UserStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class QuizType(enum.Enum):
    PRACTICE = "practice"
    ASSIGNMENT = "assignment"
    EXAM = "exam"
    QUIZ = "quiz"


class QuizStatus(enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class QuestionType(enum.Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    FILL_BLANK = "fill_blank"
    MATCHING = "matching"
    DRAG_DROP = "drag_drop"


class SubmissionStatus(enum.Enum):
    STARTED = "started"
    SUBMITTED = "submitted"
    GRADED = "graded"
    REVIEWED = "reviewed"


class AchievementType(enum.Enum):
    PROGRESS = "progress"
    STREAK = "streak"
    PERFORMANCE = "performance"
    PARTICIPATION = "participation"
    SPECIAL = "special"


class EventType(enum.Enum):
    CLASS = "class"
    EXAM = "exam"
    BREAK = "break"
    LUNCH = "lunch"
    ASSEMBLY = "assembly"
    HOLIDAY = "holiday"


class NotificationType(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    SUCCESS = "success"
    ERROR = "error"
    ACHIEVEMENT = "achievement"


class SyncStatus(enum.Enum):
    PENDING = "pending"
    SYNCED = "synced"
    CONFLICT = "conflict"
    FAILED = "failed"


# Base model with common fields
class BaseModel(Base):
    __abstract__ = True
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by_id = Column(GUID(), ForeignKey('users.id'), nullable=True)
    updated_by_id = Column(GUID(), ForeignKey('users.id'), nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)
    
    # Relationships (will be defined in child classes)
    created_by = relationship("User", foreign_keys=[created_by_id], lazy="select")
    updated_by = relationship("User", foreign_keys=[updated_by_id], lazy="select")


# User Management Models
class School(BaseModel):
    __tablename__ = "schools"
    
    name = Column(String(255), nullable=False, index=True)
    code = Column(String(50), unique=True, nullable=False)
    address = Column(Text)
    phone = Column(String(20))
    email = Column(String(255))
    website = Column(String(255))
    logo_url = Column(String(500))
    timezone = Column(String(50), default="UTC")
    academic_year_start = Column(Date)
    academic_year_end = Column(Date)
    settings = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    users = relationship("User", back_populates="school")
    classes = relationship("Class", back_populates="school")
    resources = relationship("Resource", back_populates="school")


class User(BaseModel):
    __tablename__ = "users"
    
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role = Column(Enum(UserRole), nullable=False, index=True)
    status = Column(Enum(UserStatus), default=UserStatus.ACTIVE, nullable=False)
    avatar_url = Column(String(500))
    phone = Column(String(20))
    date_of_birth = Column(Date)
    address = Column(Text)
    emergency_contact = Column(JSON)
    preferences = Column(JSON, default=dict)
    last_login = Column(DateTime)
    login_count = Column(Integer, default=0)
    is_verified = Column(Boolean, default=False)
    verification_token = Column(String(255))
    password_reset_token = Column(String(255))
    password_reset_expires = Column(DateTime)
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(32))
    
    # Foreign Keys
    school_id = Column(GUID(), ForeignKey('schools.id'), nullable=False)
    
    # Relationships
    school = relationship("School", back_populates="users")
    student_profile = relationship("StudentProfile", back_populates="user", uselist=False)
    teacher_profile = relationship("TeacherProfile", back_populates="user", uselist=False)
    enrollments = relationship("Enrollment", back_populates="student")
    quiz_submissions = relationship("QuizSubmission", back_populates="student")
    grades = relationship("Grade", back_populates="student")
    achievements = relationship("UserAchievement", back_populates="user")
    xp_transactions = relationship("XPTransaction", back_populates="user")
    analytics_events = relationship("AnalyticsEvent", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    @property
    def display_name(self) -> str:
        return self.full_name or self.username
    
    @validates('email')
    def validate_email(self, key, email):
        assert '@' in email, "Invalid email format"
        return email.lower()


class StudentProfile(BaseModel):
    __tablename__ = "student_profiles"
    
    user_id = Column(GUID(), ForeignKey('users.id'), unique=True, nullable=False)
    student_id = Column(String(50), unique=True, nullable=False, index=True)
    admission_date = Column(Date)
    graduation_date = Column(Date)
    grade_level = Column(String(20))
    section = Column(String(10))
    guardian_name = Column(String(255))
    guardian_email = Column(String(255))
    guardian_phone = Column(String(20))
    learning_style = Column(String(50))  # visual, auditory, kinesthetic, reading
    academic_interests = Column(JSON, default=list)
    goals = Column(JSON, default=list)
    accessibility_needs = Column(JSON, default=dict)
    total_xp = Column(Integer, default=0, nullable=False)
    level = Column(Integer, default=1, nullable=False)
    streak_days = Column(Integer, default=0, nullable=False)
    last_activity = Column(DateTime)
    performance_metrics = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="student_profile")
    
    @property
    def xp_to_next_level(self) -> int:
        """Calculate XP needed for next level"""
        next_level_xp = self.level * 100  # Simple formula: level * 100
        return max(0, next_level_xp - self.total_xp)


class TeacherProfile(BaseModel):
    __tablename__ = "teacher_profiles"
    
    user_id = Column(GUID(), ForeignKey('users.id'), unique=True, nullable=False)
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    department = Column(String(100))
    subjects = Column(JSON, default=list)
    qualifications = Column(JSON, default=list)
    experience_years = Column(Integer, default=0)
    hire_date = Column(Date)
    office_location = Column(String(100))
    office_hours = Column(JSON, default=dict)  # {"monday": {"start": "09:00", "end": "17:00"}}
    teaching_load = Column(Float, default=1.0)  # Full-time = 1.0
    specializations = Column(JSON, default=list)
    certifications = Column(JSON, default=list)
    performance_rating = Column(Float, default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="teacher_profile")
    classes_taught = relationship("Class", back_populates="teacher")
    quizzes_created = relationship("Quiz", back_populates="created_by")


# Academic Models
class Class(BaseModel):
    __tablename__ = "classes"
    
    name = Column(String(255), nullable=False)
    code = Column(String(50), nullable=False)
    description = Column(Text)
    subject = Column(String(100), nullable=False)
    grade_level = Column(String(20))
    section = Column(String(10))
    academic_year = Column(String(20))
    semester = Column(String(20))
    max_students = Column(Integer, default=30)
    credits = Column(Float, default=1.0)
    syllabus_url = Column(String(500))
    meeting_schedule = Column(JSON, default=dict)
    grading_scheme = Column(JSON, default=dict)
    policies = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    
    # Foreign Keys
    school_id = Column(GUID(), ForeignKey('schools.id'), nullable=False)
    teacher_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    
    # Relationships
    school = relationship("School", back_populates="classes")
    teacher = relationship("TeacherProfile", back_populates="classes_taught")
    enrollments = relationship("Enrollment", back_populates="class_obj")
    quizzes = relationship("Quiz", back_populates="class_obj")
    schedules = relationship("Schedule", back_populates="class_obj")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('school_id', 'code', 'academic_year', name='_school_class_year_uc'),
        Index('idx_class_teacher_active', 'teacher_id', 'is_active'),
    )


class Enrollment(BaseModel):
    __tablename__ = "enrollments"
    
    student_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    class_id = Column(GUID(), ForeignKey('classes.id'), nullable=False)
    enrollment_date = Column(Date, default=date.today)
    status = Column(String(20), default="active")  # active, dropped, completed
    final_grade = Column(String(5))  # A, B, C, D, F
    final_percentage = Column(Float)
    credits_earned = Column(Float)
    attendance_percentage = Column(Float)
    notes = Column(Text)
    
    # Relationships
    student = relationship("User", back_populates="enrollments")
    class_obj = relationship("Class", back_populates="enrollments")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('student_id', 'class_id', name='_student_class_uc'),
        Index('idx_enrollment_status', 'status', 'enrollment_date'),
    )


# Quiz and Assessment Models
class Quiz(BaseModel):
    __tablename__ = "quizzes"
    
    title = Column(String(255), nullable=False)
    description = Column(Text)
    instructions = Column(Text)
    quiz_type = Column(Enum(QuizType), default=QuizType.QUIZ, nullable=False)
    status = Column(Enum(QuizStatus), default=QuizStatus.DRAFT, nullable=False)
    time_limit_minutes = Column(Integer)
    max_attempts = Column(Integer, default=1)
    passing_score = Column(Float, default=60.0)
    shuffle_questions = Column(Boolean, default=True)
    shuffle_answers = Column(Boolean, default=True)
    show_results_immediately = Column(Boolean, default=True)
    show_correct_answers = Column(Boolean, default=True)
    allow_backtrack = Column(Boolean, default=False)
    require_lockdown = Column(Boolean, default=False)
    available_from = Column(DateTime)
    available_until = Column(DateTime)
    late_submission_penalty = Column(Float, default=0.0)
    weight = Column(Float, default=1.0)  # Weight in final grade calculation
    tags = Column(JSON, default=list)
    difficulty_level = Column(String(20))  # easy, medium, hard
    estimated_duration = Column(Integer)  # in minutes
    ai_generated = Column(Boolean, default=False)
    generation_prompt = Column(Text)  # Original prompt used for AI generation
    
    # Foreign Keys
    class_id = Column(GUID(), ForeignKey('classes.id'), nullable=False)
    created_by_teacher_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    
    # Relationships
    class_obj = relationship("Class", back_populates="quizzes")
    created_by = relationship("TeacherProfile", back_populates="quizzes_created")
    questions = relationship("Question", back_populates="quiz", cascade="all, delete-orphan")
    submissions = relationship("QuizSubmission", back_populates="quiz")
    
    # Constraints
    __table_args__ = (
        Index('idx_quiz_status_available', 'status', 'available_from', 'available_until'),
        CheckConstraint('passing_score >= 0 AND passing_score <= 100', name='valid_passing_score'),
    )


class Question(BaseModel):
    __tablename__ = "questions"
    
    quiz_id = Column(GUID(), ForeignKey('quizzes.id'), nullable=False)
    order_index = Column(Integer, nullable=False)
    question_type = Column(Enum(QuestionType), nullable=False)
    title = Column(String(500))
    content = Column(Text, nullable=False)
    explanation = Column(Text)  # Explanation shown after answering
    points = Column(Float, default=1.0)
    difficulty_level = Column(String(20))
    estimated_time_seconds = Column(Integer)
    media_url = Column(String(500))  # Image, video, or audio
    media_type = Column(String(50))  # image, video, audio
    options = Column(JSON, default=list)  # For multiple choice, matching, etc.
    correct_answers = Column(JSON, default=list)  # Correct answer(s)
    partial_credit_rules = Column(JSON, default=dict)
    hints = Column(JSON, default=list)
    tags = Column(JSON, default=list)
    ai_generated = Column(Boolean, default=False)
    generation_metadata = Column(JSON, default=dict)
    
    # Relationships
    quiz = relationship("Quiz", back_populates="questions")
    answers = relationship("Answer", back_populates="question", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('quiz_id', 'order_index', name='_quiz_question_order_uc'),
        Index('idx_question_type_difficulty', 'question_type', 'difficulty_level'),
        CheckConstraint('points > 0', name='positive_points'),
    )


class QuizSubmission(BaseModel):
    __tablename__ = "quiz_submissions"
    
    student_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    quiz_id = Column(GUID(), ForeignKey('quizzes.id'), nullable=False)
    attempt_number = Column(Integer, default=1, nullable=False)
    status = Column(Enum(SubmissionStatus), default=SubmissionStatus.STARTED, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    submitted_at = Column(DateTime)
    time_taken_seconds = Column(Integer)
    score = Column(Float)  # Raw score
    percentage = Column(Float)  # Percentage score
    max_possible_score = Column(Float)
    is_late = Column(Boolean, default=False)
    late_penalty_applied = Column(Float, default=0.0)
    final_score = Column(Float)  # After penalties
    feedback = Column(Text)
    teacher_comments = Column(Text)
    flagged_for_review = Column(Boolean, default=False)
    review_reason = Column(String(255))
    browser_info = Column(JSON, default=dict)
    session_data = Column(JSON, default=dict)  # For integrity monitoring
    ip_address = Column(String(45))
    
    # Relationships
    student = relationship("User", back_populates="quiz_submissions")
    quiz = relationship("Quiz", back_populates="submissions")
    answers = relationship("Answer", back_populates="submission", cascade="all, delete-orphan")
    grade = relationship("Grade", back_populates="quiz_submission", uselist=False)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('student_id', 'quiz_id', 'attempt_number', name='_student_quiz_attempt_uc'),
        Index('idx_submission_status_date', 'status', 'submitted_at'),
    )


class Answer(BaseModel):
    __tablename__ = "answers"
    
    submission_id = Column(GUID(), ForeignKey('quiz_submissions.id'), nullable=False)
    question_id = Column(GUID(), ForeignKey('questions.id'), nullable=False)
    answer_content = Column(JSON, nullable=False)  # Student's answer(s)
    is_correct = Column(Boolean)
    points_earned = Column(Float, default=0.0)
    time_taken_seconds = Column(Integer)
    answer_order = Column(Integer)  # Order in which questions were answered
    flagged = Column(Boolean, default=False)
    flag_reason = Column(String(255))
    ai_feedback = Column(Text)  # AI-generated feedback for the answer
    confidence_score = Column(Float)  # AI confidence in grading
    
    # Relationships
    submission = relationship("QuizSubmission", back_populates="answers")
    question = relationship("Question", back_populates="answers")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('submission_id', 'question_id', name='_submission_question_uc'),
        Index('idx_answer_correct_points', 'is_correct', 'points_earned'),
    )


class Grade(BaseModel):
    __tablename__ = "grades"
    
    student_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    class_id = Column(GUID(), ForeignKey('classes.id'), nullable=False)
    quiz_submission_id = Column(GUID(), ForeignKey('quiz_submissions.id'), nullable=True)
    grade_type = Column(String(50), nullable=False)  # quiz, assignment, exam, participation
    title = Column(String(255), nullable=False)
    points_earned = Column(Float, nullable=False)
    points_possible = Column(Float, nullable=False)
    percentage = Column(Float, nullable=False)
    letter_grade = Column(String(5))  # A+, A, A-, B+, etc.
    weight = Column(Float, default=1.0)
    due_date = Column(DateTime)
    graded_at = Column(DateTime)
    graded_by_id = Column(GUID(), ForeignKey('users.id'))
    feedback = Column(Text)
    rubric_scores = Column(JSON, default=dict)
    late_penalty = Column(Float, default=0.0)
    extra_credit = Column(Float, default=0.0)
    
    # Relationships
    student = relationship("User", back_populates="grades")
    quiz_submission = relationship("QuizSubmission", back_populates="grade")
    graded_by = relationship("User", foreign_keys=[graded_by_id])
    
    # Constraints
    __table_args__ = (
        Index('idx_grade_student_class_type', 'student_id', 'class_id', 'grade_type'),
        CheckConstraint('points_earned >= 0', name='non_negative_points_earned'),
        CheckConstraint('points_possible > 0', name='positive_points_possible'),
        CheckConstraint('percentage >= 0 AND percentage <= 100', name='valid_percentage'),
    )


# Scheduling Models
class Resource(BaseModel):
    __tablename__ = "resources"
    
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # room, equipment, lab, etc.
    capacity = Column(Integer)
    location = Column(String(255))
    description = Column(Text)
    features = Column(JSON, default=list)  # projector, computers, etc.
    availability = Column(JSON, default=dict)  # Available time slots
    booking_rules = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    cost_per_hour = Column(Float, default=0.0)
    maintenance_schedule = Column(JSON, default=dict)
    
    # Foreign Keys
    school_id = Column(GUID(), ForeignKey('schools.id'), nullable=False)
    
    # Relationships
    school = relationship("School", back_populates="resources")
    schedules = relationship("Schedule", back_populates="resource")
    
    # Constraints
    __table_args__ = (
        Index('idx_resource_type_active', 'type', 'is_active'),
    )


class Schedule(BaseModel):
    __tablename__ = "schedules"
    
    class_id = Column(GUID(), ForeignKey('classes.id'), nullable=False)
    resource_id = Column(GUID(), ForeignKey('resources.id'), nullable=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    event_type = Column(Enum(EventType), default=EventType.CLASS, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 6=Sunday
    recurrence_rule = Column(String(255))  # RRULE for recurring events
    is_recurring = Column(Boolean, default=False)
    parent_schedule_id = Column(GUID(), ForeignKey('schedules.id'), nullable=True)
    exceptions = Column(JSON, default=list)  # Dates when recurring event is skipped
    overrides = Column(JSON, default=dict)  # Modified instances of recurring events
    status = Column(String(20), default="active")  # active, cancelled, postponed
    attendance_required = Column(Boolean, default=True)
    max_attendees = Column(Integer)
    booking_deadline = Column(DateTime)
    cancellation_policy = Column(Text)
    special_instructions = Column(Text)
    
    # AI Optimization fields
    optimization_score = Column(Float)  # AI-calculated optimal scheduling score
    constraints_satisfied = Column(JSON, default=dict)
    optimization_metadata = Column(JSON, default=dict)
    
    # Relationships
    class_obj = relationship("Class", back_populates="schedules")
    resource = relationship("Resource", back_populates="schedules")
    parent_schedule = relationship("Schedule", remote_side=[id])
    child_schedules = relationship("Schedule", remote_side=[parent_schedule_id])
    conflicts = relationship("ScheduleConflict", back_populates="schedule")
    
    # Constraints
    __table_args__ = (
        Index('idx_schedule_time_range', 'start_time', 'end_time'),
        Index('idx_schedule_resource_time', 'resource_id', 'start_time', 'end_time'),
        CheckConstraint('end_time > start_time', name='valid_time_range'),
        CheckConstraint('day_of_week >= 0 AND day_of_week <= 6', name='valid_day_of_week'),
    )


class ScheduleConflict(BaseModel):
    __tablename__ = "schedule_conflicts"
    
    schedule_id = Column(GUID(), ForeignKey('schedules.id'), nullable=False)
    conflict_type = Column(String(50), nullable=False)  # resource, teacher, student
    description = Column(Text, nullable=False)
    severity = Column(String(20), default="medium")  # low, medium, high, critical
    conflicting_schedule_id = Column(GUID(), ForeignKey('schedules.id'), nullable=True)
    resolution_status = Column(String(20), default="pending")  # pending, resolved, ignored
    resolution_notes = Column(Text)
    auto_resolvable = Column(Boolean, default=False)
    suggested_resolution = Column(JSON, default=dict)
    
    # Relationships
    schedule = relationship("Schedule", back_populates="conflicts")
    conflicting_schedule = relationship("Schedule", foreign_keys=[conflicting_schedule_id])


# Gamification Models
class Achievement(BaseModel):
    __tablename__ = "achievements"
    
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=False)
    category = Column(Enum(AchievementType), nullable=False)
    icon_url = Column(String(500))
    badge_color = Column(String(20), default="#gold")
    points_reward = Column(Integer, default=50)
    rarity = Column(String(20), default="common")  # common, rare, epic, legendary
    unlock_criteria = Column(JSON, nullable=False)
    is_hidden = Column(Boolean, default=False)  # Hidden until unlocked
    is_active = Column(Boolean, default=True)
    unlock_message = Column(String(500))
    prerequisite_achievements = Column(JSON, default=list)
    max_level = Column(Integer, default=1)  # For progressive achievements
    
    # Relationships
    user_achievements = relationship("UserAchievement", back_populates="achievement")
    
    # Constraints
    __table_args__ = (
        Index('idx_achievement_category_rarity', 'category', 'rarity'),
    )


class UserAchievement(BaseModel):
    __tablename__ = "user_achievements"
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    achievement_id = Column(GUID(), ForeignKey('achievements.id'), nullable=False)
    unlocked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    current_level = Column(Integer, default=1)
    progress = Column(JSON, default=dict)  # Progress towards next level
    notification_sent = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="achievements")
    achievement = relationship("Achievement", back_populates="user_achievements")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'achievement_id', name='_user_achievement_uc'),
        Index('idx_user_achievement_unlocked', 'user_id', 'unlocked_at'),
    )


class XPTransaction(BaseModel):
    __tablename__ = "xp_transactions"
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    amount = Column(Integer, nullable=False)
    source_type = Column(String(50), nullable=False)  # quiz, login, achievement, etc.
    source_id = Column(String(255))  # ID of the source (quiz_id, achievement_id, etc.)
    description = Column(String(255), nullable=False)
    multiplier = Column(Float, default=1.0)
    bonus_reason = Column(String(255))
    transaction_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="xp_transactions")
    
    # Constraints
    __table_args__ = (
        Index('idx_xp_user_date', 'user_id', 'transaction_date'),
        Index('idx_xp_source', 'source_type', 'source_id'),
    )


# Analytics and Tracking Models
class AnalyticsEvent(BaseModel):
    __tablename__ = "analytics_events"
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=True)
    session_id = Column(String(255), nullable=False)
    event_type = Column(String(100), nullable=False)
    event_name = Column(String(100), nullable=False)
    page_url = Column(String(500))
    referrer_url = Column(String(500))
    user_agent = Column(String(500))
    ip_address = Column(String(45))
    device_type = Column(String(50))  # desktop, mobile, tablet
    browser = Column(String(100))
    operating_system = Column(String(100))
    screen_resolution = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    properties = Column(JSON, default=dict)
    duration = Column(Integer)  # For timed events
    
    # Relationships
    user = relationship("User", back_populates="analytics_events")
    
    # Constraints
    __table_args__ = (
        Index('idx_analytics_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_analytics_event_type', 'event_type', 'event_name'),
        Index('idx_analytics_session', 'session_id', 'timestamp'),
    )


class BehavioralProfile(BaseModel):
    __tablename__ = "behavioral_profiles"
    
    user_id = Column(GUID(), ForeignKey('users.id'), unique=True, nullable=False)
    learning_style = Column(String(50))  # Determined by AI
    engagement_level = Column(String(20))  # high, medium, low
    optimal_study_times = Column(JSON, default=list)  # Best hours for learning
    attention_span_minutes = Column(Integer)
    preferred_content_types = Column(JSON, default=list)
    difficulty_preference = Column(String(20))  # challenging, moderate, easy
    collaboration_preference = Column(String(20))  # individual, group, mixed
    feedback_style = Column(String(20))  # immediate, delayed, summary
    motivation_factors = Column(JSON, default=list)
    procrastination_tendency = Column(String(20))  # low, medium, high
    stress_indicators = Column(JSON, default=dict)
    performance_patterns = Column(JSON, default=dict)
    last_analysis = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float, default=0.0)
    analysis_version = Column(String(10), default="1.0")
    
    # Relationships
    user = relationship("User")
    
    # Constraints
    __table_args__ = (
        Index('idx_behavioral_engagement', 'engagement_level', 'last_analysis'),
    )


# Communication Models
class Notification(BaseModel):
    __tablename__ = "notifications"
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(Enum(NotificationType), default=NotificationType.INFO, nullable=False)
    priority = Column(String(20), default="normal")  # low, normal, high, urgent
    is_read = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)
    read_at = Column(DateTime)
    action_url = Column(String(500))
    action_text = Column(String(100))
    expires_at = Column(DateTime)
    channel = Column(String(50), default="web")  # web, email, push, sms
    metadata = Column(JSON, default=dict)
    delivery_status = Column(String(20), default="pending")  # pending, sent, delivered, failed
    delivery_attempts = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    
    # Constraints
    __table_args__ = (
        Index('idx_notification_user_read', 'user_id', 'is_read', 'created_at'),
        Index('idx_notification_type_priority', 'notification_type', 'priority'),
    )


# Offline Sync Models
class SyncQueue(BaseModel):
    __tablename__ = "sync_queue"
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    entity_type = Column(String(100), nullable=False)  # quiz_submission, grade, etc.
    entity_id = Column(String(255), nullable=False)
    operation = Column(String(20), nullable=False)  # create, update, delete
    data = Column(JSON, nullable=False)
    priority = Column(Integer, default=1)  # Higher number = higher priority
    status = Column(Enum(SyncStatus), default=SyncStatus.PENDING, nullable=False)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    last_attempt = Column(DateTime)
    error_message = Column(Text)
    client_timestamp = Column(DateTime, nullable=False)
    server_processed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User")
    
    # Constraints
    __table_args__ = (
        Index('idx_sync_queue_user_status', 'user_id', 'status', 'priority'),
        Index('idx_sync_queue_entity', 'entity_type', 'entity_id'),
    )


# AI Model Management
class AIModel(BaseModel):
    __tablename__ = "ai_models"
    
    name = Column(String(255), nullable=False, unique=True)
    model_type = Column(String(100), nullable=False)  # text_generation, classification, etc.
    version = Column(String(50), nullable=False)
    description = Column(Text)
    model_path = Column(String(500))
    config = Column(JSON, default=dict)
    performance_metrics = Column(JSON, default=dict)
    training_data_info = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    last_trained = Column(DateTime)
    next_training_scheduled = Column(DateTime)
    usage_statistics = Column(JSON, default=dict)
    
    # Constraints
    __table_args__ = (
        Index('idx_ai_model_type_active', 'model_type', 'is_active'),
    )


# Content Analysis Cache
class ContentAnalysis(BaseModel):
    __tablename__ = "content_analysis"
    
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    content_type = Column(String(50), nullable=False)  # quiz, essay, document
    analysis_type = Column(String(50), nullable=False)  # difficulty, plagiarism, ai_detection
    results = Column(JSON, nullable=False)
    confidence_score = Column(Float)
    model_version = Column(String(50))
    processing_time_ms = Column(Integer)
    cache_expires_at = Column(DateTime)
    
    # Constraints
    __table_args__ = (
        Index('idx_content_analysis_hash_type', 'content_hash', 'analysis_type'),
    )


# Event listeners for automatic updates
@event.listens_for(User, 'before_update')
def update_user_timestamps(mapper, connection, target):
    target.updated_at = datetime.utcnow()


@event.listens_for(StudentProfile, 'before_update')
def update_student_level(mapper, connection, target):
    """Auto-update student level based on total XP"""
    if target.total_xp:
        new_level = max(1, target.total_xp // 100)  # Simple level calculation
        if new_level != target.level:
            target.level = new_level


# Export all models
__all__ = [
    'Base', 'BaseModel',
    'School', 'User', 'StudentProfile', 'TeacherProfile',
    'Class', 'Enrollment',
    'Quiz', 'Question', 'QuizSubmission', 'Answer', 'Grade',
    'Resource', 'Schedule', 'ScheduleConflict',
    'Achievement', 'UserAchievement', 'XPTransaction',
    'AnalyticsEvent', 'BehavioralProfile',
    'Notification', 'SyncQueue',
    'AIModel', 'ContentAnalysis',
    # Enums
    'UserRole', 'UserStatus', 'QuizType', 'QuizStatus', 'QuestionType',
    'SubmissionStatus', 'AchievementType', 'EventType', 'NotificationType',
    'SyncStatus'
]
"""
AI-Powered Smart Class & Timetable Scheduler
Real-time notification system API routes
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import APIRouter, Depends, Query, Path, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, validator, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, update
from sqlalchemy.orm import selectinload
import json
import asyncio

from ..database.connection import get_db
from ..database.models import (
    User, UserRole, Notification, NotificationType,
    Quiz, QuizSubmission, Grade, Achievement, UserAchievement,
    Class, Enrollment, Schedule, XPTransaction
)
from ..dependencies import (
    require_authentication,
    require_teacher,
    require_admin,
    get_current_user,
    get_redis_client,
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

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(connection_id)
        
        logger.info(f"WebSocket connected: user={user_id}, connection={connection_id}")
    
    def disconnect(self, user_id: str, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            if connection_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(connection_id)
            
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: user={user_id}, connection={connection_id}")
    
    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.user_connections:
            dead_connections = []
            
            for connection_id in self.user_connections[user_id]:
                if connection_id in self.active_connections:
                    try:
                        await self.active_connections[connection_id].send_text(json.dumps(message))
                    except:
                        dead_connections.append(connection_id)
            
            # Clean up dead connections
            for connection_id in dead_connections:
                self.disconnect(user_id, connection_id)
    
    async def send_to_multiple_users(self, user_ids: List[str], message: dict):
        for user_id in user_ids:
            await self.send_to_user(user_id, message)
    
    async def broadcast(self, message: dict):
        dead_connections = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                dead_connections.append(connection_id)
        
        # Clean up dead connections
        for connection_id in dead_connections:
            # Find user for this connection and disconnect
            for user_id, connections in self.user_connections.items():
                if connection_id in connections:
                    self.disconnect(user_id, connection_id)
                    break

manager = ConnectionManager()


# Pydantic models
class NotificationResponse(BaseModel):
    id: str
    title: str
    message: str
    notification_type: NotificationType
    priority: str
    is_read: bool
    is_archived: bool
    action_url: Optional[str]
    action_text: Optional[str]
    expires_at: Optional[datetime]
    created_at: datetime
    read_at: Optional[datetime]
    metadata: Dict[str, Any]
    
    class Config:
        from_attributes = True


class NotificationCreateRequest(BaseModel):
    title: str
    message: str
    notification_type: NotificationType = NotificationType.INFO
    priority: str = "normal"
    target_users: Optional[List[str]] = None  # User IDs
    target_roles: Optional[List[UserRole]] = None
    target_classes: Optional[List[str]] = None  # Class IDs
    action_url: Optional[str] = None
    action_text: Optional[str] = None
    expires_at: Optional[datetime] = None
    schedule_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    @validator('title', 'message')
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title and message are required')
        return v.strip()
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['low', 'normal', 'high', 'urgent']:
            raise ValueError('Invalid priority level')
        return v


class BulkNotificationRequest(BaseModel):
    notifications: List[NotificationCreateRequest]
    
    @validator('notifications')
    def validate_notifications(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one notification is required')
        if len(v) > 100:
            raise ValueError('Cannot send more than 100 notifications at once')
        return v


class NotificationSettingsRequest(BaseModel):
    email_notifications: bool = True
    push_notifications: bool = True
    achievement_notifications: bool = True
    quiz_reminders: bool = True
    grade_notifications: bool = True
    schedule_updates: bool = True
    system_announcements: bool = True
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    digest_frequency: str = "daily"  # "immediate", "hourly", "daily", "weekly"
    
    @validator('digest_frequency')
    def validate_digest_frequency(cls, v):
        if v not in ['immediate', 'hourly', 'daily', 'weekly']:
            raise ValueError('Invalid digest frequency')
        return v


class NotificationTemplateRequest(BaseModel):
    name: str
    template_type: str  # "quiz_due", "grade_posted", "achievement_unlocked", etc.
    title_template: str
    message_template: str
    default_priority: str = "normal"
    action_url_template: Optional[str] = None
    variables: List[str] = []  # Variables that can be replaced in templates
    
    @validator('template_type')
    def validate_template_type(cls, v):
        valid_types = [
            "quiz_due", "grade_posted", "achievement_unlocked", "class_reminder",
            "schedule_change", "system_announcement", "welcome", "password_reset"
        ]
        if v not in valid_types:
            raise ValueError('Invalid template type')
        return v


class NotificationStatsResponse(BaseModel):
    total_sent: int
    total_read: int
    total_unread: int
    by_type: Dict[str, int]
    by_priority: Dict[str, int]
    engagement_rate: float
    recent_activity: List[Dict[str, Any]]


# Helper functions
async def create_notification(
    user_id: str,
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.INFO,
    priority: str = "normal",
    action_url: Optional[str] = None,
    action_text: Optional[str] = None,
    expires_at: Optional[datetime] = None,
    metadata: Dict[str, Any] = None,
    db: AsyncSession = None
) -> str:
    """Create a notification for a user"""
    
    import uuid
    
    notification = Notification(
        id=uuid.uuid4(),
        user_id=user_id,
        title=title,
        message=message,
        notification_type=notification_type,
        priority=priority,
        action_url=action_url,
        action_text=action_text,
        expires_at=expires_at,
        metadata=metadata or {},
        is_read=False,
        is_archived=False,
        delivery_status="pending"
    )
    
    if db:
        db.add(notification)
        await db.flush()
        return str(notification.id)
    
    return str(notification.id)


async def send_real_time_notification(user_id: str, notification_data: Dict[str, Any]):
    """Send real-time notification via WebSocket"""
    
    message = {
        "type": "notification",
        "data": notification_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.send_to_user(user_id, message)


async def get_users_by_criteria(
    target_users: Optional[List[str]] = None,
    target_roles: Optional[List[UserRole]] = None,
    target_classes: Optional[List[str]] = None,
    db: AsyncSession = None
) -> List[str]:
    """Get user IDs based on targeting criteria"""
    
    user_ids = set()
    
    # Direct user targeting
    if target_users:
        user_ids.update(target_users)
    
    # Role-based targeting
    if target_roles:
        role_users_result = await db.execute(
            select(User.id).where(
                User.role.in_(target_roles),
                User.is_deleted == False,
                User.status == "active"
            )
        )
        user_ids.update(str(user_id) for user_id in role_users_result.scalars())
    
    # Class-based targeting
    if target_classes:
        class_users_result = await db.execute(
            select(Enrollment.student_id).where(
                Enrollment.class_id.in_(target_classes),
                Enrollment.status == "active"
            )
        )
        user_ids.update(str(user_id) for user_id in class_users_result.scalars())
        
        # Also include teachers of these classes
        teacher_result = await db.execute(
            select(Class.teacher_id).where(
                Class.id.in_(target_classes),
                Class.is_active == True
            )
        )
        user_ids.update(str(user_id) for user_id in teacher_result.scalars())
    
    return list(user_ids)


async def process_notification_templates(
    template_type: str,
    variables: Dict[str, Any],
    db: AsyncSession
) -> Dict[str, str]:
    """Process notification templates with variable substitution"""
    
    # This would typically load from a template database
    # For now, we'll use predefined templates
    
    templates = {
        "quiz_due": {
            "title": "Quiz Due: {quiz_title}",
            "message": "Your quiz '{quiz_title}' is due {due_date}. Don't forget to submit it!",
            "action_url": "/quiz/{quiz_id}",
            "action_text": "Take Quiz"
        },
        "grade_posted": {
            "title": "Grade Posted for {quiz_title}",
            "message": "Your grade for '{quiz_title}' has been posted. Score: {score}%",
            "action_url": "/grades/{grade_id}",
            "action_text": "View Grade"
        },
        "achievement_unlocked": {
            "title": "Achievement Unlocked: {achievement_name}",
            "message": "Congratulations! You've earned the '{achievement_name}' achievement. {points} XP awarded!",
            "action_url": "/achievements",
            "action_text": "View Achievements"
        },
        "class_reminder": {
            "title": "Class Reminder: {class_name}",
            "message": "Your class '{class_name}' starts in {time_until}.",
            "action_url": "/schedule",
            "action_text": "View Schedule"
        },
        "schedule_change": {
            "title": "Schedule Update: {class_name}",
            "message": "The schedule for '{class_name}' has been updated. New time: {new_time}",
            "action_url": "/schedule",
            "action_text": "View Schedule"
        }
    }
    
    if template_type not in templates:
        return {
            "title": "Notification",
            "message": "You have a new notification.",
            "action_url": None,
            "action_text": None
        }
    
    template = templates[template_type]
    processed = {}
    
    for key, value in template.items():
        if value and isinstance(value, str):
            # Replace variables in template
            processed_value = value
            for var_name, var_value in variables.items():
                processed_value = processed_value.replace(f"{{{var_name}}}", str(var_value))
            processed[key] = processed_value
        else:
            processed[key] = value
    
    return processed


# Automatic notification triggers
async def trigger_quiz_due_notifications(db: AsyncSession):
    """Send notifications for quizzes due soon"""
    
    # Get quizzes due within 24 hours
    tomorrow = datetime.utcnow() + timedelta(days=1)
    
    due_quizzes_result = await db.execute(
        select(Quiz, Class).join(Class).where(
            Quiz.available_until.between(datetime.utcnow(), tomorrow),
            Quiz.status == "published"
        )
    )
    
    for quiz, class_obj in due_quizzes_result:
        # Get enrolled students
        students_result = await db.execute(
            select(Enrollment.student_id).where(
                Enrollment.class_id == class_obj.id,
                Enrollment.status == "active"
            )
        )
        
        student_ids = [str(student_id) for student_id in students_result.scalars()]
        
        for student_id in student_ids:
            # Check if student hasn't submitted yet
            submission_result = await db.execute(
                select(QuizSubmission).where(
                    QuizSubmission.quiz_id == quiz.id,
                    QuizSubmission.student_id == student_id,
                    QuizSubmission.status.in_(["submitted", "graded"])
                )
            )
            
            if not submission_result.scalar_one_or_none():
                # Student hasn't submitted, send reminder
                time_left = quiz.available_until - datetime.utcnow()
                hours_left = int(time_left.total_seconds() / 3600)
                
                template_vars = {
                    "quiz_title": quiz.title,
                    "due_date": f"{hours_left} hours",
                    "quiz_id": str(quiz.id)
                }
                
                processed = await process_notification_templates("quiz_due", template_vars, db)
                
                notification_id = await create_notification(
                    user_id=student_id,
                    title=processed["title"],
                    message=processed["message"],
                    notification_type=NotificationType.WARNING,
                    priority="high",
                    action_url=processed["action_url"],
                    action_text=processed["action_text"],
                    db=db
                )
                
                # Send real-time notification
                await send_real_time_notification(student_id, {
                    "id": notification_id,
                    "title": processed["title"],
                    "message": processed["message"],
                    "type": "warning",
                    "priority": "high"
                })


async def trigger_grade_posted_notifications(grade_id: str, db: AsyncSession):
    """Send notification when a grade is posted"""
    
    grade_result = await db.execute(
        select(Grade, User, Quiz).select_from(Grade)
        .join(User, Grade.student_id == User.id)
        .outerjoin(Quiz, Grade.quiz_submission_id == QuizSubmission.id)
        .outerjoin(QuizSubmission, QuizSubmission.quiz_id == Quiz.id)
        .where(Grade.id == grade_id)
    )
    
    grade_data = grade_result.first()
    if not grade_data:
        return
    
    grade, student, quiz = grade_data
    
    template_vars = {
        "quiz_title": quiz.title if quiz else "Assignment",
        "score": int(grade.percentage),
        "grade_id": str(grade.id)
    }
    
    processed = await process_notification_templates("grade_posted", template_vars, db)
    
    notification_id = await create_notification(
        user_id=str(student.id),
        title=processed["title"],
        message=processed["message"],
        notification_type=NotificationType.SUCCESS,
        priority="normal",
        action_url=processed["action_url"],
        action_text=processed["action_text"],
        db=db
    )
    
    await send_real_time_notification(str(student.id), {
        "id": notification_id,
        "title": processed["title"],
        "message": processed["message"],
        "type": "success",
        "priority": "normal"
    })


async def trigger_achievement_notifications(user_achievement_id: str, db: AsyncSession):
    """Send notification when achievement is unlocked"""
    
    achievement_result = await db.execute(
        select(UserAchievement, Achievement).join(Achievement).where(
            UserAchievement.id == user_achievement_id
        )
    )
    
    achievement_data = achievement_result.first()
    if not achievement_data:
        return
    
    user_achievement, achievement = achievement_data
    
    template_vars = {
        "achievement_name": achievement.name,
        "points": achievement.points_reward
    }
    
    processed = await process_notification_templates("achievement_unlocked", template_vars, db)
    
    notification_id = await create_notification(
        user_id=str(user_achievement.user_id),
        title=processed["title"],
        message=processed["message"],
        notification_type=NotificationType.ACHIEVEMENT,
        priority="normal",
        action_url=processed["action_url"],
        action_text=processed["action_text"],
        db=db
    )
    
    await send_real_time_notification(str(user_achievement.user_id), {
        "id": notification_id,
        "title": processed["title"],
        "message": processed["message"],
        "type": "achievement",
        "priority": "normal",
        "special_effects": ["confetti", "celebration"]  # For UI effects
    })


# WebSocket endpoint
@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    token: str = Query(..., description="JWT token for authentication")
):
    """WebSocket endpoint for real-time notifications"""
    
    try:
        # Validate token and user
        from ..dependencies import verify_jwt_token
        payload = await verify_jwt_token(token)
        
        if payload.get("sub") != user_id:
            await websocket.close(code=4001, reason="Invalid user token")
            return
        
        import uuid
        connection_id = str(uuid.uuid4())
        
        await manager.connect(websocket, user_id, connection_id)
        
        try:
            while True:
                # Keep connection alive and handle any client messages
                data = await websocket.receive_text()
                
                # Handle ping/pong for connection health
                if data == "ping":
                    await websocket.send_text("pong")
                
        except WebSocketDisconnect:
            manager.disconnect(user_id, connection_id)
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        try:
            await websocket.close(code=4000, reason="Authentication failed")
        except:
            pass


# API Routes
@router.get("/", response_model=List[NotificationResponse])
async def get_user_notifications(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db),
    unread_only: bool = Query(False, description="Get only unread notifications"),
    notification_type: Optional[NotificationType] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=200, description="Number of notifications to return"),
    skip: int = Query(0, ge=0, description="Number of notifications to skip")
):
    """Get notifications for current user"""
    
    query = select(Notification).where(
        Notification.user_id == current_user.id,
        Notification.is_deleted == False,
        or_(
            Notification.expires_at.is_(None),
            Notification.expires_at > datetime.utcnow()
        )
    )
    
    if unread_only:
        query = query.where(Notification.is_read == False)
    
    if notification_type:
        query = query.where(Notification.notification_type == notification_type)
    
    query = query.order_by(desc(Notification.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    notifications = result.scalars().all()
    
    notification_list = []
    for notification in notifications:
        notification_list.append(NotificationResponse(
            id=str(notification.id),
            title=notification.title,
            message=notification.message,
            notification_type=notification.notification_type,
            priority=notification.priority,
            is_read=notification.is_read,
            is_archived=notification.is_archived,
            action_url=notification.action_url,
            action_text=notification.action_text,
            expires_at=notification.expires_at,
            created_at=notification.created_at,
            read_at=notification.read_at,
            metadata=notification.metadata
        ))
    
    return notification_list


@router.post("/", dependencies=[Depends(require_teacher)])
async def create_notification_endpoint(
    request: NotificationCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Create and send notifications"""
    
    # Get target users
    target_user_ids = await get_users_by_criteria(
        target_users=request.target_users,
        target_roles=request.target_roles,
        target_classes=request.target_classes,
        db=db
    )
    
    if not target_user_ids:
        raise ValidationException("No target users found for notification")
    
    created_notifications = []
    
    for user_id in target_user_ids:
        if request.schedule_at and request.schedule_at > datetime.utcnow():
            # Schedule notification for later
            background_tasks.add_task(
                schedule_notification,
                user_id,
                request.title,
                request.message,
                request.notification_type,
                request.priority,
                request.action_url,
                request.action_text,
                request.expires_at,
                request.metadata,
                request.schedule_at
            )
        else:
            # Send immediately
            notification_id = await create_notification(
                user_id=user_id,
                title=request.title,
                message=request.message,
                notification_type=request.notification_type,
                priority=request.priority,
                action_url=request.action_url,
                action_text=request.action_text,
                expires_at=request.expires_at,
                metadata=request.metadata,
                db=db
            )
            
            created_notifications.append(notification_id)
            
            # Send real-time notification
            await send_real_time_notification(user_id, {
                "id": notification_id,
                "title": request.title,
                "message": request.message,
                "type": request.notification_type.value,
                "priority": request.priority
            })
    
    await db.commit()
    
    logger.info(f"Notifications created by {current_user.username} for {len(target_user_ids)} users")
    
    return {
        "message": f"Notifications sent to {len(target_user_ids)} users",
        "notification_ids": created_notifications,
        "scheduled": request.schedule_at is not None and request.schedule_at > datetime.utcnow()
    }


async def schedule_notification(
    user_id: str,
    title: str,
    message: str,
    notification_type: NotificationType,
    priority: str,
    action_url: Optional[str],
    action_text: Optional[str],
    expires_at: Optional[datetime],
    metadata: Dict[str, Any],
    schedule_time: datetime
):
    """Schedule notification for later delivery"""
    
    # Wait until scheduled time
    wait_seconds = (schedule_time - datetime.utcnow()).total_seconds()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
    
    # Create and send notification
    from ..database.connection import get_async_session
    
    async with get_async_session() as db:
        notification_id = await create_notification(
            user_id=user_id,
            title=title,
            message=message,
            notification_type=notification_type,
            priority=priority,
            action_url=action_url,
            action_text=action_text,
            expires_at=expires_at,
            metadata=metadata,
            db=db
        )
        
        await db.commit()
        
        # Send real-time notification
        await send_real_time_notification(user_id, {
            "id": notification_id,
            "title": title,
            "message": message,
            "type": notification_type.value,
            "priority": priority
        })


@router.put("/{notification_id}/read")
async def mark_notification_read(
    notification_id: str = Path(..., description="Notification ID"),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Mark notification as read"""
    
    result = await db.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == current_user.id
        )
    )
    
    notification = result.scalar_one_or_none()
    if not notification:
        raise NotFoundException("Notification not found")
    
    if not notification.is_read:
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        await db.commit()
    
    return {"message": "Notification marked as read"}


@router.put("/mark-all-read")
async def mark_all_notifications_read(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Mark all notifications as read for current user"""
    
    await db.execute(
        update(Notification)
        .where(
            Notification.user_id == current_user.id,
            Notification.is_read == False
        )
        .values(
            is_read=True,
            read_at=datetime.utcnow()
        )
    )
    
    await db.commit()
    
    return {"message": "All notifications marked as read"}


@router.put("/{notification_id}/archive")
async def archive_notification(
    notification_id: str = Path(..., description="Notification ID"),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Archive a notification"""
    
    result = await db.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == current_user.id
        )
    )
    
    notification = result.scalar_one_or_none()
    if not notification:
        raise NotFoundException("Notification not found")
    
    notification.is_archived = True
    await db.commit()
    
    return {"message": "Notification archived"}


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: str = Path(..., description="Notification ID"),
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Delete a notification"""
    
    result = await db.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == current_user.id
        )
    )
    
    notification = result.scalar_one_or_none()
    if not notification:
        raise NotFoundException("Notification not found")
    
    notification.is_deleted = True
    notification.deleted_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Notification deleted"}


@router.get("/unread-count")
async def get_unread_count(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get count of unread notifications"""
    
    count = await db.scalar(
        select(func.count(Notification.id)).where(
            Notification.user_id == current_user.id,
            Notification.is_read == False,
            Notification.is_deleted == False,
            or_(
                Notification.expires_at.is_(None),
                Notification.expires_at > datetime.utcnow()
            )
        )
    )
    
    return {"unread_count": count or 0}


@router.post("/bulk", dependencies=[Depends(require_teacher)])
async def send_bulk_notifications(
    request: BulkNotificationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Send multiple notifications in bulk"""
    
    total_notifications = 0
    
    for notification_request in request.notifications:
        target_user_ids = await get_users_by_criteria(
            target_users=notification_request.target_users,
            target_roles=notification_request.target_roles,
            target_classes=notification_request.target_classes,
            db=db
        )
        
        for user_id in target_user_ids:
            await create_notification(
                user_id=user_id,
                title=notification_request.title,
                message=notification_request.message,
                notification_type=notification_request.notification_type,
                priority=notification_request.priority,
                action_url=notification_request.action_url,
                action_text=notification_request.action_text,
                expires_at=notification_request.expires_at,
                metadata=notification_request.metadata,
                db=db
            )
            
            total_notifications += 1
    
    await db.commit()
    
    logger.info(f"Bulk notifications sent by {current_user.username}: {total_notifications} total")
    
    return {
        "message": f"Sent {total_notifications} notifications successfully",
        "notification_batches": len(request.notifications)
    }


@router.get("/settings")
async def get_notification_settings(
    current_user: User = Depends(require_authentication)
):
    """Get user's notification settings"""
    
    # Get from user preferences
    settings = current_user.preferences.get("notifications", {
        "email_notifications": True,
        "push_notifications": True,
        "achievement_notifications": True,
        "quiz_reminders": True,
        "grade_notifications": True,
        "schedule_updates": True,
        "system_announcements": True,
        "digest_frequency": "daily"
    })
    
    return {"settings": settings}


@router.put("/settings")
async def update_notification_settings(
    request: NotificationSettingsRequest,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Update user's notification settings"""
    
    # Update user preferences
    if not current_user.preferences:
        current_user.preferences = {}
    
    current_user.preferences["notifications"] = {
        "email_notifications": request.email_notifications,
        "push_notifications": request.push_notifications,
        "achievement_notifications": request.achievement_notifications,
        "quiz_reminders": request.quiz_reminders,
        "grade_notifications": request.grade_notifications,
        "schedule_updates": request.schedule_updates,
        "system_announcements": request.system_announcements,
        "quiet_hours_start": request.quiet_hours_start,
        "quiet_hours_end": request.quiet_hours_end,
        "digest_frequency": request.digest_frequency
    }
    
    current_user.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Notification settings updated successfully"}


@router.get("/stats", response_model=NotificationStatsResponse)
async def get_notification_statistics(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get notification statistics (admin only)"""
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Total counts
    total_sent = await db.scalar(
        select(func.count(Notification.id)).where(
            Notification.created_at >= start_date
        )
    ) or 0
    
    total_read = await db.scalar(
        select(func.count(Notification.id)).where(
            Notification.created_at >= start_date,
            Notification.is_read == True
        )
    ) or 0
    
    total_unread = total_sent - total_read
    
    # By type
    type_stats = await db.execute(
        select(
            Notification.notification_type,
            func.count(Notification.id).label('count')
        )
        .where(Notification.created_at >= start_date)
        .group_by(Notification.notification_type)
    )
    
    by_type = {}
    for row in type_stats:
        by_type[row.notification_type.value] = row.count
    
    # By priority
    priority_stats = await db.execute(
        select(
            Notification.priority,
            func.count(Notification.id).label('count')
        )
        .where(Notification.created_at >= start_date)
        .group_by(Notification.priority)
    )
    
    by_priority = {}
    for row in priority_stats:
        by_priority[row.priority] = row.count
    
    # Engagement rate
    engagement_rate = (total_read / total_sent * 100) if total_sent > 0 else 0
    
    # Recent activity (last 7 days, by day)
    recent_activity = []
    for i in range(7):
        day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        
        day_count = await db.scalar(
            select(func.count(Notification.id)).where(
                Notification.created_at.between(day_start, day_end)
            )
        ) or 0
        
        recent_activity.append({
            "date": day_start.date().isoformat(),
            "count": day_count
        })
    
    recent_activity.reverse()  # Show oldest to newest
    
    return NotificationStatsResponse(
        total_sent=total_sent,
        total_read=total_read,
        total_unread=total_unread,
        by_type=by_type,
        by_priority=by_priority,
        engagement_rate=round(engagement_rate, 2),
        recent_activity=recent_activity
    )


@router.post("/test-connection")
async def test_websocket_connection(
    current_user: User = Depends(require_authentication)
):
    """Test WebSocket connection by sending a test notification"""
    
    test_message = {
        "type": "test",
        "data": {
            "title": "Connection Test",
            "message": "WebSocket connection is working!",
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    await manager.send_to_user(str(current_user.id), test_message)
    
    return {"message": "Test notification sent via WebSocket"}


# Background task for automatic notifications
async def run_notification_triggers():
    """Background task to check for and send automatic notifications"""
    
    from ..database.connection import get_async_session
    
    while True:
        try:
            async with get_async_session() as db:
                await trigger_quiz_due_notifications(db)
                await db.commit()
        
        except Exception as e:
            logger.error(f"Notification trigger error: {e}")
        
        # Wait 15 minutes before next check
        await asyncio.sleep(900)


# Export router
__all__ = ["router", "manager", "trigger_grade_posted_notifications", "trigger_achievement_notifications"]
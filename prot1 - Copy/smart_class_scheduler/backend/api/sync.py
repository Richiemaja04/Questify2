"""
AI-Powered Smart Class & Timetable Scheduler
Offline synchronization API routes for managing data sync between client and server
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import APIRouter, Depends, Query, Path, HTTPException, status
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.orm import selectinload
import json

from ..database.connection import get_db
from ..database.models import (
    User, Quiz, QuizSubmission, Answer, Grade, XPTransaction,
    UserAchievement, AnalyticsEvent, SyncQueue, SyncStatus,
    StudentProfile, Notification
)
from ..dependencies import (
    require_authentication,
    get_current_user,
    get_redis_client,
    SessionManager,
    user_rate_limit
)
from ..exceptions import (
    NotFoundException,
    AuthorizationException,
    ValidationException,
    SyncException,
    SyncConflictException,
    BusinessLogicException
)

# Configure logging
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()


# Pydantic models
class SyncDataRequest(BaseModel):
    entity_type: str
    entity_id: str
    operation: str  # create, update, delete
    data: Dict[str, Any]
    client_timestamp: datetime
    client_version: str = "1.0"
    device_id: Optional[str] = None
    
    @validator('operation')
    def validate_operation(cls, v):
        if v not in ['create', 'update', 'delete']:
            raise ValueError('Invalid operation')
        return v
    
    @validator('entity_type')
    def validate_entity_type(cls, v):
        valid_types = [
            'quiz_submission', 'answer', 'analytics_event', 
            'xp_transaction', 'user_profile', 'notification_read'
        ]
        if v not in valid_types:
            raise ValueError(f'Invalid entity type. Must be one of: {valid_types}')
        return v


class SyncBatchRequest(BaseModel):
    sync_items: List[SyncDataRequest]
    batch_id: str
    client_last_sync: Optional[datetime] = None
    
    @validator('sync_items')
    def validate_sync_items(cls, v):
        if not v:
            raise ValueError('At least one sync item is required')
        if len(v) > 100:  # Limit batch size
            raise ValueError('Batch size cannot exceed 100 items')
        return v


class SyncResponse(BaseModel):
    sync_id: str
    status: SyncStatus
    server_timestamp: datetime
    conflicts: List[Dict[str, Any]] = []
    error_message: Optional[str] = None


class SyncBatchResponse(BaseModel):
    batch_id: str
    processed_count: int
    successful_count: int
    failed_count: int
    conflict_count: int
    results: List[SyncResponse]
    server_timestamp: datetime


class SyncStatusResponse(BaseModel):
    user_id: str
    last_sync: Optional[datetime]
    pending_items: int
    failed_items: int
    next_sync_recommended: datetime
    sync_statistics: Dict[str, Any]


class PullSyncRequest(BaseModel):
    last_sync_timestamp: Optional[datetime] = None
    entity_types: List[str] = []
    max_items: int = 1000
    
    @validator('max_items')
    def validate_max_items(cls, v):
        if v < 1 or v > 5000:
            raise ValueError('Max items must be between 1 and 5000')
        return v


class PullSyncResponse(BaseModel):
    updates: List[Dict[str, Any]]
    deletions: List[Dict[str, Any]]
    server_timestamp: datetime
    has_more: bool
    continuation_token: Optional[str] = None


class ConflictResolutionRequest(BaseModel):
    sync_id: str
    resolution_strategy: str  # 'client_wins', 'server_wins', 'merge', 'manual'
    resolved_data: Optional[Dict[str, Any]] = None
    
    @validator('resolution_strategy')
    def validate_resolution_strategy(cls, v):
        if v not in ['client_wins', 'server_wins', 'merge', 'manual']:
            raise ValueError('Invalid resolution strategy')
        return v


class OfflineCacheRequest(BaseModel):
    cache_type: str  # 'quiz_data', 'class_info', 'user_profile'
    entity_ids: List[str] = []
    cache_duration_hours: int = 24
    
    @validator('cache_duration_hours')
    def validate_cache_duration(cls, v):
        if v < 1 or v > 168:  # Max 1 week
            raise ValueError('Cache duration must be between 1 and 168 hours')
        return v


# Helper functions
async def create_sync_queue_item(
    user_id: str,
    sync_request: SyncDataRequest,
    db: AsyncSession
) -> str:
    """Create a sync queue item for processing"""
    
    import uuid
    
    sync_item = SyncQueue(
        id=uuid.uuid4(),
        user_id=user_id,
        entity_type=sync_request.entity_type,
        entity_id=sync_request.entity_id,
        operation=sync_request.operation,
        data=sync_request.data,
        priority=1,  # Default priority
        status=SyncStatus.PENDING,
        retry_count=0,
        max_retries=3,
        client_timestamp=sync_request.client_timestamp
    )
    
    db.add(sync_item)
    await db.flush()
    
    return str(sync_item.id)


async def process_sync_item(
    sync_item: SyncQueue,
    user: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Process an individual sync item"""
    
    try:
        if sync_item.entity_type == 'quiz_submission':
            return await sync_quiz_submission(sync_item, user, db)
        elif sync_item.entity_type == 'answer':
            return await sync_answer(sync_item, user, db)
        elif sync_item.entity_type == 'analytics_event':
            return await sync_analytics_event(sync_item, user, db)
        elif sync_item.entity_type == 'xp_transaction':
            return await sync_xp_transaction(sync_item, user, db)
        elif sync_item.entity_type == 'user_profile':
            return await sync_user_profile(sync_item, user, db)
        elif sync_item.entity_type == 'notification_read':
            return await sync_notification_read(sync_item, user, db)
        else:
            raise SyncException(f"Unsupported entity type: {sync_item.entity_type}")
    
    except Exception as e:
        logger.error(f"Sync processing failed for item {sync_item.id}: {e}")
        raise SyncException(f"Sync processing failed: {str(e)}")


async def sync_quiz_submission(
    sync_item: SyncQueue,
    user: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Sync quiz submission data"""
    
    if sync_item.operation == 'create':
        # Check if submission already exists
        existing_submission = await db.execute(
            select(QuizSubmission).where(
                QuizSubmission.id == sync_item.entity_id
            )
        )
        
        if existing_submission.scalar_one_or_none():
            # Conflict - submission already exists
            return {
                "status": "conflict",
                "conflict_type": "duplicate_submission",
                "server_data": "Submission already exists on server"
            }
        
        # Create new submission
        submission_data = sync_item.data
        import uuid
        
        submission = QuizSubmission(
            id=uuid.UUID(sync_item.entity_id) if sync_item.entity_id else uuid.uuid4(),
            student_id=user.id,
            quiz_id=submission_data['quiz_id'],
            attempt_number=submission_data.get('attempt_number', 1),
            status=submission_data.get('status', 'started'),
            started_at=datetime.fromisoformat(submission_data['started_at']),
            submitted_at=datetime.fromisoformat(submission_data['submitted_at']) if submission_data.get('submitted_at') else None,
            time_taken_seconds=submission_data.get('time_taken_seconds'),
            score=submission_data.get('score'),
            percentage=submission_data.get('percentage'),
            max_possible_score=submission_data.get('max_possible_score'),
            is_late=submission_data.get('is_late', False),
            session_data=submission_data.get('session_data', {})
        )
        
        db.add(submission)
        return {"status": "success", "action": "created"}
    
    elif sync_item.operation == 'update':
        # Update existing submission
        result = await db.execute(
            select(QuizSubmission).where(
                QuizSubmission.id == sync_item.entity_id,
                QuizSubmission.student_id == user.id
            )
        )
        
        submission = result.scalar_one_or_none()
        if not submission:
            return {
                "status": "conflict",
                "conflict_type": "submission_not_found",
                "message": "Submission not found on server"
            }
        
        # Check for conflicts (server timestamp vs client timestamp)
        if submission.updated_at > sync_item.client_timestamp:
            return {
                "status": "conflict",
                "conflict_type": "concurrent_modification",
                "server_data": {
                    "updated_at": submission.updated_at.isoformat(),
                    "status": submission.status.value,
                    "score": submission.score
                }
            }
        
        # Apply updates
        submission_data = sync_item.data
        for field, value in submission_data.items():
            if hasattr(submission, field) and field not in ['id', 'student_id', 'created_at']:
                if field in ['started_at', 'submitted_at'] and value:
                    setattr(submission, field, datetime.fromisoformat(value))
                else:
                    setattr(submission, field, value)
        
        submission.updated_at = datetime.utcnow()
        return {"status": "success", "action": "updated"}
    
    return {"status": "error", "message": "Unsupported operation"}


async def sync_answer(
    sync_item: SyncQueue,
    user: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Sync answer data"""
    
    if sync_item.operation == 'create':
        answer_data = sync_item.data
        import uuid
        
        # Verify the submission belongs to the user
        submission_result = await db.execute(
            select(QuizSubmission).where(
                QuizSubmission.id == answer_data['submission_id'],
                QuizSubmission.student_id == user.id
            )
        )
        
        submission = submission_result.scalar_one_or_none()
        if not submission:
            return {
                "status": "error",
                "message": "Submission not found or access denied"
            }
        
        answer = Answer(
            id=uuid.UUID(sync_item.entity_id) if sync_item.entity_id else uuid.uuid4(),
            submission_id=answer_data['submission_id'],
            question_id=answer_data['question_id'],
            answer_content=answer_data['answer_content'],
            is_correct=answer_data.get('is_correct'),
            points_earned=answer_data.get('points_earned', 0.0),
            time_taken_seconds=answer_data.get('time_taken_seconds'),
            answer_order=answer_data.get('answer_order')
        )
        
        db.add(answer)
        return {"status": "success", "action": "created"}
    
    return {"status": "error", "message": "Unsupported operation for answers"}


async def sync_analytics_event(
    sync_item: SyncQueue,
    user: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Sync analytics event data"""
    
    if sync_item.operation == 'create':
        event_data = sync_item.data
        import uuid
        
        event = AnalyticsEvent(
            id=uuid.UUID(sync_item.entity_id) if sync_item.entity_id else uuid.uuid4(),
            user_id=user.id,
            session_id=event_data['session_id'],
            event_type=event_data['event_type'],
            event_name=event_data['event_name'],
            page_url=event_data.get('page_url'),
            timestamp=datetime.fromisoformat(event_data['timestamp']),
            properties=event_data.get('properties', {}),
            duration=event_data.get('duration')
        )
        
        db.add(event)
        return {"status": "success", "action": "created"}
    
    return {"status": "success", "action": "ignored"}  # Analytics events are typically create-only


async def sync_xp_transaction(
    sync_item: SyncQueue,
    user: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Sync XP transaction data"""
    
    if sync_item.operation == 'create':
        # XP transactions are typically server-generated, so this might be a conflict
        return {
            "status": "conflict",
            "conflict_type": "server_authoritative",
            "message": "XP transactions are managed by the server"
        }
    
    return {"status": "success", "action": "ignored"}


async def sync_user_profile(
    sync_item: SyncQueue,
    user: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Sync user profile updates"""
    
    if sync_item.operation == 'update':
        profile_result = await db.execute(
            select(StudentProfile).where(StudentProfile.user_id == user.id)
        )
        
        profile = profile_result.scalar_one_or_none()
        if not profile:
            return {"status": "error", "message": "Profile not found"}
        
        # Apply updates from client
        profile_data = sync_item.data
        updatable_fields = ['learning_style', 'academic_interests', 'goals', 'accessibility_needs']
        
        for field in updatable_fields:
            if field in profile_data:
                setattr(profile, field, profile_data[field])
        
        profile.updated_at = datetime.utcnow()
        return {"status": "success", "action": "updated"}
    
    return {"status": "error", "message": "Unsupported operation"}


async def sync_notification_read(
    sync_item: SyncQueue,
    user: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Sync notification read status"""
    
    if sync_item.operation == 'update':
        notification_data = sync_item.data
        
        result = await db.execute(
            select(Notification).where(
                Notification.id == sync_item.entity_id,
                Notification.user_id == user.id
            )
        )
        
        notification = result.scalar_one_or_none()
        if not notification:
            return {"status": "error", "message": "Notification not found"}
        
        notification.is_read = notification_data.get('is_read', True)
        notification.read_at = datetime.utcnow() if notification.is_read else None
        
        return {"status": "success", "action": "updated"}
    
    return {"status": "error", "message": "Unsupported operation"}


async def get_server_updates_for_user(
    user_id: str,
    last_sync_timestamp: Optional[datetime],
    entity_types: List[str],
    db: AsyncSession
) -> Dict[str, List[Dict[str, Any]]]:
    """Get updates from server for user since last sync"""
    
    updates = []
    deletions = []
    
    sync_timestamp = last_sync_timestamp or datetime.utcnow() - timedelta(days=30)
    
    # Get quiz updates
    if not entity_types or 'quiz' in entity_types:
        quiz_updates = await db.execute(
            select(Quiz)
            .join(Class)
            .join(Enrollment)
            .where(
                Enrollment.student_id == user_id,
                Quiz.updated_at > sync_timestamp,
                Quiz.is_deleted == False
            )
        )
        
        for quiz in quiz_updates.scalars():
            updates.append({
                "entity_type": "quiz",
                "entity_id": str(quiz.id),
                "operation": "update",
                "data": {
                    "title": quiz.title,
                    "description": quiz.description,
                    "status": quiz.status.value,
                    "available_from": quiz.available_from.isoformat() if quiz.available_from else None,
                    "available_until": quiz.available_until.isoformat() if quiz.available_until else None,
                    "time_limit_minutes": quiz.time_limit_minutes,
                    "max_attempts": quiz.max_attempts,
                    "updated_at": quiz.updated_at.isoformat()
                }
            })
    
    # Get grade updates
    if not entity_types or 'grade' in entity_types:
        grade_updates = await db.execute(
            select(Grade).where(
                Grade.student_id == user_id,
                Grade.updated_at > sync_timestamp,
                Grade.is_deleted == False
            )
        )
        
        for grade in grade_updates.scalars():
            updates.append({
                "entity_type": "grade",
                "entity_id": str(grade.id),
                "operation": "update",
                "data": {
                    "points_earned": grade.points_earned,
                    "points_possible": grade.points_possible,
                    "percentage": grade.percentage,
                    "letter_grade": grade.letter_grade,
                    "feedback": grade.feedback,
                    "graded_at": grade.graded_at.isoformat() if grade.graded_at else None,
                    "updated_at": grade.updated_at.isoformat()
                }
            })
    
    # Get achievement updates
    if not entity_types or 'achievement' in entity_types:
        achievement_updates = await db.execute(
            select(UserAchievement)
            .where(
                UserAchievement.user_id == user_id,
                UserAchievement.unlocked_at > sync_timestamp
            )
        )
        
        for achievement in achievement_updates.scalars():
            updates.append({
                "entity_type": "user_achievement",
                "entity_id": str(achievement.id),
                "operation": "create",
                "data": {
                    "achievement_id": str(achievement.achievement_id),
                    "unlocked_at": achievement.unlocked_at.isoformat(),
                    "current_level": achievement.current_level,
                    "progress": achievement.progress
                }
            })
    
    # Get notifications
    if not entity_types or 'notification' in entity_types:
        notification_updates = await db.execute(
            select(Notification).where(
                Notification.user_id == user_id,
                Notification.created_at > sync_timestamp,
                Notification.is_deleted == False
            )
        )
        
        for notification in notification_updates.scalars():
            updates.append({
                "entity_type": "notification",
                "entity_id": str(notification.id),
                "operation": "create",
                "data": {
                    "title": notification.title,
                    "message": notification.message,
                    "notification_type": notification.notification_type.value,
                    "priority": notification.priority,
                    "is_read": notification.is_read,
                    "action_url": notification.action_url,
                    "expires_at": notification.expires_at.isoformat() if notification.expires_at else None,
                    "created_at": notification.created_at.isoformat()
                }
            })
    
    return {"updates": updates, "deletions": deletions}


# API Routes
@router.post("/push", response_model=SyncResponse)
async def push_sync_data(
    request: SyncDataRequest,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Push single sync item to server"""
    
    try:
        # Create sync queue item
        sync_id = await create_sync_queue_item(str(current_user.id), request, db)
        
        # Get the sync item for processing
        sync_item_result = await db.execute(
            select(SyncQueue).where(SyncQueue.id == sync_id)
        )
        sync_item = sync_item_result.scalar_one()
        
        # Process the sync item
        result = await process_sync_item(sync_item, current_user, db)
        
        if result["status"] == "success":
            sync_item.status = SyncStatus.SYNCED
            sync_item.server_processed_at = datetime.utcnow()
        elif result["status"] == "conflict":
            sync_item.status = SyncStatus.CONFLICT
            sync_item.error_message = result.get("message", "Sync conflict occurred")
        else:
            sync_item.status = SyncStatus.FAILED
            sync_item.error_message = result.get("message", "Sync failed")
            sync_item.retry_count += 1
        
        await db.commit()
        
        conflicts = []
        if result["status"] == "conflict":
            conflicts.append({
                "entity_type": request.entity_type,
                "entity_id": request.entity_id,
                "conflict_type": result.get("conflict_type"),
                "server_data": result.get("server_data"),
                "message": result.get("message")
            })
        
        return SyncResponse(
            sync_id=sync_id,
            status=sync_item.status,
            server_timestamp=datetime.utcnow(),
            conflicts=conflicts,
            error_message=sync_item.error_message
        )
    
    except Exception as e:
        logger.error(f"Push sync failed: {e}")
        raise SyncException(f"Push sync failed: {str(e)}")


@router.post("/push-batch", response_model=SyncBatchResponse)
async def push_sync_batch(
    request: SyncBatchRequest,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Push batch of sync items to server"""
    
    results = []
    successful_count = 0
    failed_count = 0
    conflict_count = 0
    
    try:
        for sync_request in request.sync_items:
            try:
                # Create and process sync item
                sync_id = await create_sync_queue_item(str(current_user.id), sync_request, db)
                
                sync_item_result = await db.execute(
                    select(SyncQueue).where(SyncQueue.id == sync_id)
                )
                sync_item = sync_item_result.scalar_one()
                
                result = await process_sync_item(sync_item, current_user, db)
                
                conflicts = []
                if result["status"] == "success":
                    sync_item.status = SyncStatus.SYNCED
                    sync_item.server_processed_at = datetime.utcnow()
                    successful_count += 1
                elif result["status"] == "conflict":
                    sync_item.status = SyncStatus.CONFLICT
                    sync_item.error_message = result.get("message", "Sync conflict occurred")
                    conflict_count += 1
                    conflicts.append({
                        "entity_type": sync_request.entity_type,
                        "entity_id": sync_request.entity_id,
                        "conflict_type": result.get("conflict_type"),
                        "server_data": result.get("server_data")
                    })
                else:
                    sync_item.status = SyncStatus.FAILED
                    sync_item.error_message = result.get("message", "Sync failed")
                    sync_item.retry_count += 1
                    failed_count += 1
                
                results.append(SyncResponse(
                    sync_id=sync_id,
                    status=sync_item.status,
                    server_timestamp=datetime.utcnow(),
                    conflicts=conflicts,
                    error_message=sync_item.error_message
                ))
            
            except Exception as e:
                logger.error(f"Batch sync item failed: {e}")
                failed_count += 1
                results.append(SyncResponse(
                    sync_id="",
                    status=SyncStatus.FAILED,
                    server_timestamp=datetime.utcnow(),
                    error_message=str(e)
                ))
        
        await db.commit()
        
        return SyncBatchResponse(
            batch_id=request.batch_id,
            processed_count=len(request.sync_items),
            successful_count=successful_count,
            failed_count=failed_count,
            conflict_count=conflict_count,
            results=results,
            server_timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"Batch sync failed: {e}")
        raise SyncException(f"Batch sync failed: {str(e)}")


@router.post("/pull", response_model=PullSyncResponse)
async def pull_sync_data(
    request: PullSyncRequest,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Pull updates from server since last sync"""
    
    try:
        # Get server updates
        server_data = await get_server_updates_for_user(
            str(current_user.id),
            request.last_sync_timestamp,
            request.entity_types,
            db
        )
        
        updates = server_data["updates"]
        deletions = server_data["deletions"]
        
        # Apply max items limit
        has_more = len(updates) > request.max_items
        if has_more:
            updates = updates[:request.max_items]
        
        # Generate continuation token if needed
        continuation_token = None
        if has_more:
            import base64
            continuation_data = {
                "last_timestamp": updates[-1]["data"]["updated_at"] if updates else None,
                "entity_types": request.entity_types
            }
            continuation_token = base64.b64encode(
                json.dumps(continuation_data).encode()
            ).decode()
        
        return PullSyncResponse(
            updates=updates,
            deletions=deletions,
            server_timestamp=datetime.utcnow(),
            has_more=has_more,
            continuation_token=continuation_token
        )
    
    except Exception as e:
        logger.error(f"Pull sync failed: {e}")
        raise SyncException(f"Pull sync failed: {str(e)}")


@router.get("/status", response_model=SyncStatusResponse)
async def get_sync_status(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Get sync status for current user"""
    
    # Get sync statistics
    pending_count = await db.scalar(
        select(func.count(SyncQueue.id))
        .where(
            SyncQueue.user_id == current_user.id,
            SyncQueue.status == SyncStatus.PENDING
        )
    ) or 0
    
    failed_count = await db.scalar(
        select(func.count(SyncQueue.id))
        .where(
            SyncQueue.user_id == current_user.id,
            SyncQueue.status == SyncStatus.FAILED
        )
    ) or 0
    
    # Get last successful sync
    last_sync_result = await db.execute(
        select(SyncQueue.server_processed_at)
        .where(
            SyncQueue.user_id == current_user.id,
            SyncQueue.status == SyncStatus.SYNCED
        )
        .order_by(desc(SyncQueue.server_processed_at))
        .limit(1)
    )
    
    last_sync = last_sync_result.scalar_one_or_none()
    
    # Calculate next recommended sync
    next_sync_recommended = datetime.utcnow() + timedelta(minutes=15)  # Default 15 minutes
    
    if pending_count > 0 or failed_count > 0:
        next_sync_recommended = datetime.utcnow() + timedelta(minutes=1)  # Immediate if issues
    
    # Sync statistics
    sync_statistics = {
        "total_synced_today": await db.scalar(
            select(func.count(SyncQueue.id))
            .where(
                SyncQueue.user_id == current_user.id,
                SyncQueue.status == SyncStatus.SYNCED,
                SyncQueue.server_processed_at >= datetime.utcnow().date()
            )
        ) or 0,
        "average_sync_time_ms": 150,  # Mock average
        "sync_efficiency": 0.95  # Mock efficiency rate
    }
    
    return SyncStatusResponse(
        user_id=str(current_user.id),
        last_sync=last_sync,
        pending_items=pending_count,
        failed_items=failed_count,
        next_sync_recommended=next_sync_recommended,
        sync_statistics=sync_statistics
    )


@router.post("/resolve-conflict")
async def resolve_sync_conflict(
    request: ConflictResolutionRequest,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Resolve a sync conflict"""
    
    # Get sync item
    sync_result = await db.execute(
        select(SyncQueue).where(
            SyncQueue.id == request.sync_id,
            SyncQueue.user_id == current_user.id,
            SyncQueue.status == SyncStatus.CONFLICT
        )
    )
    
    sync_item = sync_result.scalar_one_or_none()
    if not sync_item:
        raise NotFoundException("Conflict not found or already resolved")
    
    try:
        if request.resolution_strategy == "client_wins":
            # Reprocess with client data winning
            result = await process_sync_item(sync_item, current_user, db)
            sync_item.status = SyncStatus.SYNCED if result["status"] == "success" else SyncStatus.FAILED
        
        elif request.resolution_strategy == "server_wins":
            # Mark as resolved, keeping server data
            sync_item.status = SyncStatus.SYNCED
            sync_item.resolution_notes = "Server data preserved"
        
        elif request.resolution_strategy == "merge":
            # Merge data (simplified - would need entity-specific logic)
            if request.resolved_data:
                sync_item.data = request.resolved_data
                result = await process_sync_item(sync_item, current_user, db)
                sync_item.status = SyncStatus.SYNCED if result["status"] == "success" else SyncStatus.FAILED
        
        elif request.resolution_strategy == "manual":
            # User provided manual resolution
            if not request.resolved_data:
                raise ValidationException("Resolved data required for manual resolution")
            
            sync_item.data = request.resolved_data
            result = await process_sync_item(sync_item, current_user, db)
            sync_item.status = SyncStatus.SYNCED if result["status"] == "success" else SyncStatus.FAILED
        
        sync_item.server_processed_at = datetime.utcnow()
        await db.commit()
        
        logger.info(f"Conflict {request.sync_id} resolved using {request.resolution_strategy}")
        
        return {
            "message": "Conflict resolved successfully",
            "sync_id": request.sync_id,
            "resolution_strategy": request.resolution_strategy,
            "status": sync_item.status.value
        }
    
    except Exception as e:
        logger.error(f"Conflict resolution failed: {e}")
        raise SyncException(f"Conflict resolution failed: {str(e)}")


@router.post("/cache-for-offline")
async def cache_data_for_offline(
    request: OfflineCacheRequest,
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Cache data for offline access"""
    
    redis_client = await get_redis_client()
    if not redis_client:
        raise SyncException("Caching service unavailable")
    
    try:
        cache_data = {}
        cache_key = f"offline_cache:{current_user.id}:{request.cache_type}"
        
        if request.cache_type == "quiz_data":
            # Cache quiz and question data
            for quiz_id in request.entity_ids:
                quiz_result = await db.execute(
                    select(Quiz)
                    .options(selectinload(Quiz.questions))
                    .where(Quiz.id == quiz_id)
                )
                quiz = quiz_result.scalar_one_or_none()
                
                if quiz:
                    cache_data[quiz_id] = {
                        "title": quiz.title,
                        "description": quiz.description,
                        "time_limit_minutes": quiz.time_limit_minutes,
                        "questions": [
                            {
                                "id": str(q.id),
                                "content": q.content,
                                "question_type": q.question_type.value,
                                "options": q.options,
                                "points": q.points
                            }
                            for q in quiz.questions
                        ]
                    }
        
        elif request.cache_type == "user_profile":
            # Cache user profile data
            profile_result = await db.execute(
                select(StudentProfile).where(StudentProfile.user_id == current_user.id)
            )
            profile = profile_result.scalar_one_or_none()
            
            if profile:
                cache_data["profile"] = {
                    "total_xp": profile.total_xp,
                    "level": profile.level,
                    "streak_days": profile.streak_days,
                    "learning_style": profile.learning_style,
                    "goals": profile.goals
                }
        
        # Store in Redis with expiration
        await redis_client.setex(
            cache_key,
            request.cache_duration_hours * 3600,
            json.dumps(cache_data)
        )
        
        return {
            "message": "Data cached for offline access",
            "cache_key": cache_key,
            "expires_in_hours": request.cache_duration_hours,
            "cached_items": len(cache_data)
        }
    
    except Exception as e:
        logger.error(f"Offline caching failed: {e}")
        raise SyncException(f"Offline caching failed: {str(e)}")


@router.delete("/clear-cache")
async def clear_offline_cache(
    cache_type: Optional[str] = Query(None, description="Specific cache type to clear"),
    current_user: User = Depends(require_authentication)
):
    """Clear offline cache data"""
    
    redis_client = await get_redis_client()
    if not redis_client:
        return {"message": "Caching service unavailable"}
    
    try:
        if cache_type:
            # Clear specific cache type
            cache_key = f"offline_cache:{current_user.id}:{cache_type}"
            await redis_client.delete(cache_key)
            cleared_keys = 1
        else:
            # Clear all cache for user
            pattern = f"offline_cache:{current_user.id}:*"
            keys = await redis_client.keys(pattern)
            if keys:
                await redis_client.delete(*keys)
            cleared_keys = len(keys)
        
        return {
            "message": "Cache cleared successfully",
            "cleared_keys": cleared_keys
        }
    
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        return {"message": "Cache clearing failed", "error": str(e)}


@router.get("/pending-items")
async def get_pending_sync_items(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=200)
):
    """Get pending sync items for user"""
    
    result = await db.execute(
        select(SyncQueue)
        .where(
            SyncQueue.user_id == current_user.id,
            SyncQueue.status.in_([SyncStatus.PENDING, SyncStatus.FAILED])
        )
        .order_by(desc(SyncQueue.priority), SyncQueue.created_at)
        .limit(limit)
    )
    
    pending_items = []
    for item in result.scalars():
        pending_items.append({
            "sync_id": str(item.id),
            "entity_type": item.entity_type,
            "entity_id": item.entity_id,
            "operation": item.operation,
            "status": item.status.value,
            "retry_count": item.retry_count,
            "created_at": item.created_at.isoformat(),
            "error_message": item.error_message
        })
    
    return {
        "pending_items": pending_items,
        "total_count": len(pending_items)
    }


@router.post("/retry-failed")
async def retry_failed_sync_items(
    current_user: User = Depends(require_authentication),
    db: AsyncSession = Depends(get_db)
):
    """Retry all failed sync items for user"""
    
    # Get failed items
    failed_items_result = await db.execute(
        select(SyncQueue).where(
            SyncQueue.user_id == current_user.id,
            SyncQueue.status == SyncStatus.FAILED,
            SyncQueue.retry_count < SyncQueue.max_retries
        )
    )
    
    failed_items = failed_items_result.scalars().all()
    retry_count = 0
    
    for item in failed_items:
        try:
            result = await process_sync_item(item, current_user, db)
            
            if result["status"] == "success":
                item.status = SyncStatus.SYNCED
                item.server_processed_at = datetime.utcnow()
                item.error_message = None
                retry_count += 1
            else:
                item.retry_count += 1
                item.error_message = result.get("message", "Retry failed")
        
        except Exception as e:
            item.retry_count += 1
            item.error_message = str(e)
    
    await db.commit()
    
    return {
        "message": f"Retried {len(failed_items)} failed items",
        "successful_retries": retry_count,
        "failed_retries": len(failed_items) - retry_count
    }


# Export router
__all__ = ["router"]
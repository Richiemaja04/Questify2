"""
AI-Powered Smart Class & Timetable Scheduler
Custom exception classes for structured error handling
"""

from typing import Optional, Dict, Any, List
from fastapi import status


class AppException(Exception):
    """Base application exception with structured error information"""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }


# Authentication Exceptions
class AuthenticationException(AppException):
    """Raised when authentication fails"""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_FAILED",
            details=details
        )


class InvalidCredentialsException(AuthenticationException):
    """Raised when login credentials are invalid"""
    
    def __init__(self, message: str = "Invalid username or password"):
        super().__init__(
            message=message,
            details={"field": "credentials"}
        )


class TokenExpiredException(AuthenticationException):
    """Raised when JWT token has expired"""
    
    def __init__(self, message: str = "Access token has expired"):
        super().__init__(
            message=message,
            details={"action": "refresh_token"}
        )


class TokenInvalidException(AuthenticationException):
    """Raised when JWT token is invalid"""
    
    def __init__(self, message: str = "Invalid access token"):
        super().__init__(
            message=message,
            details={"action": "login_required"}
        )


class AccountDisabledException(AuthenticationException):
    """Raised when user account is disabled"""
    
    def __init__(self, message: str = "User account is disabled"):
        super().__init__(
            message=message,
            details={"contact": "administrator"}
        )


class TwoFactorRequiredException(AuthenticationException):
    """Raised when two-factor authentication is required"""
    
    def __init__(self, message: str = "Two-factor authentication required"):
        super().__init__(
            message=message,
            details={"action": "provide_2fa_code"}
        )


# Authorization Exceptions
class AuthorizationException(AppException):
    """Raised when user lacks permission for an action"""
    
    def __init__(
        self,
        message: str = "Access denied",
        required_role: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if required_role:
            details["required_role"] = required_role
        
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="ACCESS_DENIED",
            details=details
        )


class InsufficientPermissionException(AuthorizationException):
    """Raised when user has insufficient permissions"""
    
    def __init__(
        self,
        resource: str,
        action: str,
        required_role: Optional[str] = None
    ):
        message = f"Insufficient permissions to {action} {resource}"
        super().__init__(
            message=message,
            required_role=required_role,
            details={
                "resource": resource,
                "action": action
            }
        )


class ResourceOwnershipException(AuthorizationException):
    """Raised when user doesn't own the requested resource"""
    
    def __init__(self, resource_type: str, resource_id: str):
        message = f"You don't have access to this {resource_type}"
        super().__init__(
            message=message,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id
            }
        )


# Validation Exceptions
class ValidationException(AppException):
    """Raised when input validation fails"""
    
    def __init__(
        self,
        message: str = "Validation failed",
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if field:
            details["field"] = field
        if value is not None:
            details["provided_value"] = str(value)
        
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details
        )


class InvalidInputException(ValidationException):
    """Raised when input data is invalid"""
    
    def __init__(
        self,
        field: str,
        message: str,
        value: Optional[Any] = None
    ):
        super().__init__(
            message=f"Invalid {field}: {message}",
            field=field,
            value=value
        )


class MissingFieldException(ValidationException):
    """Raised when required field is missing"""
    
    def __init__(self, field: str):
        super().__init__(
            message=f"Missing required field: {field}",
            field=field,
            details={"validation_rule": "required"}
        )


class InvalidFormatException(ValidationException):
    """Raised when field format is invalid"""
    
    def __init__(
        self,
        field: str,
        expected_format: str,
        value: Optional[Any] = None
    ):
        super().__init__(
            message=f"Invalid format for {field}. Expected: {expected_format}",
            field=field,
            value=value,
            details={
                "expected_format": expected_format,
                "validation_rule": "format"
            }
        )


class ValueRangeException(ValidationException):
    """Raised when value is out of allowed range"""
    
    def __init__(
        self,
        field: str,
        value: Any,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None
    ):
        range_str = ""
        if min_value is not None and max_value is not None:
            range_str = f" (allowed range: {min_value} - {max_value})"
        elif min_value is not None:
            range_str = f" (minimum: {min_value})"
        elif max_value is not None:
            range_str = f" (maximum: {max_value})"
        
        super().__init__(
            message=f"Value for {field} is out of range{range_str}",
            field=field,
            value=value,
            details={
                "min_value": min_value,
                "max_value": max_value,
                "validation_rule": "range"
            }
        )


# Resource Exceptions
class NotFoundException(AppException):
    """Raised when requested resource is not found"""
    
    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            details=details
        )


class ResourceNotFoundByIdException(NotFoundException):
    """Raised when resource with specific ID is not found"""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type.title()} with ID '{resource_id}' not found",
            resource_type=resource_type,
            resource_id=resource_id
        )


class UserNotFoundException(NotFoundException):
    """Raised when user is not found"""
    
    def __init__(self, identifier: str, identifier_type: str = "id"):
        super().__init__(
            message=f"User not found",
            resource_type="user",
            resource_id=identifier
        )
        self.details["identifier_type"] = identifier_type


class QuizNotFoundException(NotFoundException):
    """Raised when quiz is not found"""
    
    def __init__(self, quiz_id: str):
        super().__init__(
            message="Quiz not found",
            resource_type="quiz",
            resource_id=quiz_id
        )


class ClassNotFoundException(NotFoundException):
    """Raised when class is not found"""
    
    def __init__(self, class_id: str):
        super().__init__(
            message="Class not found",
            resource_type="class",
            resource_id=class_id
        )


# Conflict Exceptions
class ConflictException(AppException):
    """Raised when operation conflicts with current state"""
    
    def __init__(
        self,
        message: str = "Conflict with current state",
        conflict_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if conflict_type:
            details["conflict_type"] = conflict_type
        
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT",
            details=details
        )


class DuplicateResourceException(ConflictException):
    """Raised when trying to create duplicate resource"""
    
    def __init__(
        self,
        resource_type: str,
        field: str,
        value: str
    ):
        super().__init__(
            message=f"{resource_type.title()} with {field} '{value}' already exists",
            conflict_type="duplicate",
            details={
                "resource_type": resource_type,
                "duplicate_field": field,
                "duplicate_value": value
            }
        )


class ScheduleConflictException(ConflictException):
    """Raised when schedule conflicts occur"""
    
    def __init__(
        self,
        message: str = "Schedule conflict detected",
        conflicting_schedules: Optional[List[str]] = None
    ):
        details = {"conflict_type": "schedule"}
        if conflicting_schedules:
            details["conflicting_schedules"] = conflicting_schedules
        
        super().__init__(
            message=message,
            details=details
        )


class QuizSubmissionConflictException(ConflictException):
    """Raised when quiz submission conflicts occur"""
    
    def __init__(self, reason: str, quiz_id: str):
        super().__init__(
            message=f"Cannot submit quiz: {reason}",
            conflict_type="quiz_submission",
            details={"quiz_id": quiz_id, "reason": reason}
        )


# Business Logic Exceptions
class BusinessLogicException(AppException):
    """Raised when business rules are violated"""
    
    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if rule_name:
            details["violated_rule"] = rule_name
        
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="BUSINESS_RULE_VIOLATION",
            details=details
        )


class QuizNotAvailableException(BusinessLogicException):
    """Raised when quiz is not available for submission"""
    
    def __init__(self, reason: str, quiz_id: str):
        super().__init__(
            message=f"Quiz not available: {reason}",
            rule_name="quiz_availability",
            details={"quiz_id": quiz_id, "reason": reason}
        )


class MaxAttemptsExceededException(BusinessLogicException):
    """Raised when maximum quiz attempts are exceeded"""
    
    def __init__(self, quiz_id: str, max_attempts: int, current_attempts: int):
        super().__init__(
            message=f"Maximum attempts exceeded ({current_attempts}/{max_attempts})",
            rule_name="max_attempts",
            details={
                "quiz_id": quiz_id,
                "max_attempts": max_attempts,
                "current_attempts": current_attempts
            }
        )


class EnrollmentRequiredException(BusinessLogicException):
    """Raised when user must be enrolled to access resource"""
    
    def __init__(self, class_id: str):
        super().__init__(
            message="You must be enrolled in this class to access this resource",
            rule_name="enrollment_required",
            details={"class_id": class_id}
        )


class GradeSubmissionException(BusinessLogicException):
    """Raised when grade cannot be submitted"""
    
    def __init__(self, reason: str, submission_id: str):
        super().__init__(
            message=f"Cannot submit grade: {reason}",
            rule_name="grade_submission",
            details={"submission_id": submission_id, "reason": reason}
        )


# External Service Exceptions
class ExternalServiceException(AppException):
    """Raised when external service calls fail"""
    
    def __init__(
        self,
        service_name: str,
        message: str = "External service error",
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        details["service"] = service_name
        
        super().__init__(
            message=f"{service_name} error: {message}",
            status_code=status_code,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details
        )


class AIServiceException(ExternalServiceException):
    """Raised when AI service calls fail"""
    
    def __init__(
        self,
        message: str = "AI service unavailable",
        model_name: Optional[str] = None
    ):
        details = {}
        if model_name:
            details["model"] = model_name
        
        super().__init__(
            service_name="AI Service",
            message=message,
            details=details
        )


class EmailServiceException(ExternalServiceException):
    """Raised when email service fails"""
    
    def __init__(self, message: str = "Email service unavailable"):
        super().__init__(
            service_name="Email Service",
            message=message
        )


class DatabaseConnectionException(ExternalServiceException):
    """Raised when database connection fails"""
    
    def __init__(self, message: str = "Database connection failed"):
        super().__init__(
            service_name="Database",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class CacheServiceException(ExternalServiceException):
    """Raised when cache service fails"""
    
    def __init__(self, message: str = "Cache service unavailable"):
        super().__init__(
            service_name="Cache Service",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# Rate Limiting Exceptions
class RateLimitException(AppException):
    """Raised when rate limit is exceeded"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None
    ):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


# File Upload Exceptions
class FileUploadException(AppException):
    """Raised when file upload fails"""
    
    def __init__(
        self,
        message: str = "File upload failed",
        filename: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if filename:
            details["filename"] = filename
        
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="FILE_UPLOAD_ERROR",
            details=details
        )


class FileSizeExceedException(FileUploadException):
    """Raised when uploaded file is too large"""
    
    def __init__(
        self,
        filename: str,
        size: int,
        max_size: int
    ):
        super().__init__(
            message=f"File '{filename}' is too large ({size} bytes). Maximum allowed: {max_size} bytes",
            filename=filename,
            details={
                "file_size": size,
                "max_allowed_size": max_size
            }
        )


class InvalidFileTypeException(FileUploadException):
    """Raised when uploaded file type is not allowed"""
    
    def __init__(
        self,
        filename: str,
        file_type: str,
        allowed_types: List[str]
    ):
        super().__init__(
            message=f"File type '{file_type}' not allowed. Allowed types: {', '.join(allowed_types)}",
            filename=filename,
            details={
                "file_type": file_type,
                "allowed_types": allowed_types
            }
        )


# Sync and Offline Exceptions
class SyncException(AppException):
    """Raised when synchronization fails"""
    
    def __init__(
        self,
        message: str = "Synchronization failed",
        sync_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if sync_type:
            details["sync_type"] = sync_type
        
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="SYNC_ERROR",
            details=details
        )


class SyncConflictException(SyncException):
    """Raised when sync conflicts occur"""
    
    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        conflict_reason: str
    ):
        super().__init__(
            message=f"Sync conflict for {entity_type} '{entity_id}': {conflict_reason}",
            sync_type="conflict",
            details={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "conflict_reason": conflict_reason
            }
        )


# Maintenance and System Exceptions
class MaintenanceModeException(AppException):
    """Raised when system is in maintenance mode"""
    
    def __init__(
        self,
        message: str = "System is currently under maintenance",
        estimated_duration: Optional[str] = None
    ):
        details = {}
        if estimated_duration:
            details["estimated_duration"] = estimated_duration
        
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MAINTENANCE_MODE",
            details=details
        )


class FeatureDisabledException(AppException):
    """Raised when requested feature is disabled"""
    
    def __init__(
        self,
        feature_name: str,
        reason: Optional[str] = None
    ):
        message = f"Feature '{feature_name}' is currently disabled"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            error_code="FEATURE_DISABLED",
            details={
                "feature": feature_name,
                "reason": reason
            }
        )


# Exception mapping for common HTTP status codes
EXCEPTION_STATUS_MAP = {
    400: BusinessLogicException,
    401: AuthenticationException,
    403: AuthorizationException,
    404: NotFoundException,
    409: ConflictException,
    422: ValidationException,
    429: RateLimitException,
    500: AppException,
    503: ExternalServiceException
}


def create_http_exception(
    status_code: int,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> AppException:
    """Create appropriate exception based on HTTP status code"""
    
    exception_class = EXCEPTION_STATUS_MAP.get(status_code, AppException)
    
    if exception_class == AppException:
        return AppException(
            message=message,
            status_code=status_code,
            details=details
        )
    
    # For specific exceptions, use their constructors
    if status_code == 404:
        return NotFoundException(message)
    elif status_code == 401:
        return AuthenticationException(message, details)
    elif status_code == 403:
        return AuthorizationException(message, details=details)
    elif status_code == 409:
        return ConflictException(message, details=details)
    elif status_code == 422:
        return ValidationException(message, details=details)
    elif status_code == 429:
        return RateLimitException(message)
    else:
        return exception_class(message)


# Export all exceptions
__all__ = [
    # Base
    "AppException",
    
    # Authentication
    "AuthenticationException",
    "InvalidCredentialsException",
    "TokenExpiredException", 
    "TokenInvalidException",
    "AccountDisabledException",
    "TwoFactorRequiredException",
    
    # Authorization
    "AuthorizationException",
    "InsufficientPermissionException",
    "ResourceOwnershipException",
    
    # Validation
    "ValidationException",
    "InvalidInputException",
    "MissingFieldException",
    "InvalidFormatException",
    "ValueRangeException",
    
    # Resources
    "NotFoundException",
    "ResourceNotFoundByIdException",
    "UserNotFoundException",
    "QuizNotFoundException",
    "ClassNotFoundException",
    
    # Conflicts
    "ConflictException",
    "DuplicateResourceException",
    "ScheduleConflictException",
    "QuizSubmissionConflictException",
    
    # Business Logic
    "BusinessLogicException",
    "QuizNotAvailableException",
    "MaxAttemptsExceededException",
    "EnrollmentRequiredException",
    "GradeSubmissionException",
    
    # External Services
    "ExternalServiceException",
    "AIServiceException",
    "EmailServiceException",
    "DatabaseConnectionException",
    "CacheServiceException",
    
    # Rate Limiting
    "RateLimitException",
    
    # File Upload
    "FileUploadException",
    "FileSizeExceedException",
    "InvalidFileTypeException",
    
    # Sync
    "SyncException",
    "SyncConflictException",
    
    # System
    "MaintenanceModeException",
    "FeatureDisabledException",
    
    # Utilities
    "create_http_exception",
    "EXCEPTION_STATUS_MAP"
]
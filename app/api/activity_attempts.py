from typing import Never
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.activity_attempts import (
    ActivityAttempt,
    ActivityAttemptList,
    ActivitySuggestionList,
    SaveActivityAttemptRequest,
)
from app.services.actions import ActionNotFoundError, ActionStorageError
from app.services.activity_attempts import (
    ActivityAttemptConflict,
    ActivityAttemptInvalid,
    ActivityAttemptNotFound,
    ActivityAttemptService,
    ActivityAttemptStorageError,
)

router = APIRouter(tags=["activity attempts"])
logger = get_logger("api.activity_attempts")


def _service(request: Request) -> ActivityAttemptService:
    return request.app.state.activity_attempt_service


def _raise(error: Exception, event: str) -> Never:
    if isinstance(error, (ActivityAttemptNotFound, ActionNotFoundError)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    if isinstance(error, ActivityAttemptConflict):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    if isinstance(error, ActivityAttemptInvalid):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    if isinstance(error, (ActivityAttemptStorageError, ActionStorageError)):
        logger.error("%s error_type=%s", event, type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not access activity attempts",
        ) from error
    raise error


@router.get("/actions/{action_id}/activity-suggestions", response_model=ActivitySuggestionList)
def list_activity_suggestions(action_id: UUID, request: Request) -> ActivitySuggestionList:
    try:
        suggestions = _service(request).suggestions(action_id)
    except Exception as error:
        _raise(error, "activity_suggestions_failed")
    return ActivitySuggestionList(suggestions=suggestions)


@router.get("/actions/{action_id}/activity-attempts", response_model=ActivityAttemptList)
def list_activity_attempts(action_id: UUID, request: Request) -> ActivityAttemptList:
    try:
        attempts = _service(request).list_attempts(action_id)
    except Exception as error:
        _raise(error, "activity_attempts_list_failed")
    return ActivityAttemptList(attempts=attempts)


@router.get("/actions/{action_id}/activity-attempts/{attempt_id}", response_model=ActivityAttempt)
def get_activity_attempt(action_id: UUID, attempt_id: UUID, request: Request) -> ActivityAttempt:
    try:
        return _service(request).get_attempt(action_id, attempt_id)
    except Exception as error:
        _raise(error, "activity_attempt_load_failed")


@router.put("/actions/{action_id}/activity-attempts/{attempt_id}", response_model=ActivityAttempt)
def save_activity_attempt(
    action_id: UUID,
    attempt_id: UUID,
    payload: SaveActivityAttemptRequest,
    request: Request,
) -> ActivityAttempt:
    try:
        attempt = _service(request).save_attempt(action_id, attempt_id, payload)
    except Exception as error:
        _raise(error, "activity_attempt_save_failed")
    logger.info(
        "activity_attempt_saved action_id=%s attempt_id=%s status=%s revision=%s",
        action_id,
        attempt_id,
        attempt.status,
        attempt.revision,
    )
    return attempt

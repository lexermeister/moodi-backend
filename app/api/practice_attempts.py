from typing import Never
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.practice_attempts import (
    PracticeAttempt,
    PracticeAttemptList,
    SavePracticeAttemptRequest,
)
from app.services.actions import ActionNotFoundError, ActionStorageError
from app.services.practice_attempts import (
    PracticeAttemptConflict,
    PracticeAttemptInvalid,
    PracticeAttemptNotFound,
    PracticeAttemptService,
    PracticeAttemptStorageError,
)

router = APIRouter(tags=["practice attempts"])
logger = get_logger("api.practice_attempts")


def _service(request: Request) -> PracticeAttemptService:
    return request.app.state.practice_attempt_service


def _raise(error: Exception, event: str) -> Never:
    if isinstance(error, (PracticeAttemptNotFound, ActionNotFoundError)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    if isinstance(error, PracticeAttemptConflict):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    if isinstance(error, PracticeAttemptInvalid):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    if isinstance(error, (PracticeAttemptStorageError, ActionStorageError)):
        logger.error("%s error_type=%s", event, type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not access practice attempts",
        ) from error
    raise error


@router.get("/actions/{action_id}/practice-attempts", response_model=PracticeAttemptList)
def list_practice_attempts(action_id: UUID, request: Request) -> PracticeAttemptList:
    try:
        attempts = _service(request).list_attempts(action_id)
    except Exception as error:
        _raise(error, "practice_attempts_list_failed")
    return PracticeAttemptList(attempts=attempts)


@router.get("/actions/{action_id}/practice-attempts/{attempt_id}", response_model=PracticeAttempt)
def get_practice_attempt(action_id: UUID, attempt_id: UUID, request: Request) -> PracticeAttempt:
    try:
        return _service(request).get_attempt(action_id, attempt_id)
    except Exception as error:
        _raise(error, "practice_attempt_load_failed")


@router.put("/actions/{action_id}/practice-attempts/{attempt_id}", response_model=PracticeAttempt)
def save_practice_attempt(
    action_id: UUID,
    attempt_id: UUID,
    payload: SavePracticeAttemptRequest,
    request: Request,
) -> PracticeAttempt:
    try:
        attempt = _service(request).save_attempt(action_id, attempt_id, payload)
    except Exception as error:
        _raise(error, "practice_attempt_save_failed")
    logger.info(
        "practice_attempt_saved action_id=%s attempt_id=%s status=%s revision=%s",
        action_id,
        attempt_id,
        attempt.status,
        attempt.revision,
    )
    return attempt

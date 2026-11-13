from datetime import UTC, date, datetime
from typing import Annotated, Never
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.core.logging import get_logger
from app.schemas.actions import (
    AmendRatingEntryRequest,
    ConfirmRatingEntryRequest,
    RatingAction,
    RatingActionListResponse,
    RatingCandidateResolutionResponse,
    RatingCandidateReviewRequest,
    RatingCandidateReviewResponse,
    RatingEntry,
    RatingEntryExportRequest,
    RatingEntryExportResponse,
    RatingEntryListResponse,
    SaveRatingEntryDraftRequest,
)
from app.services.actions import (
    ActionConflictError,
    ActionNotFoundError,
    ActionService,
    ActionStorageError,
    ActionValidationError,
)
from app.services.recording_storage import RecordingNotFoundError, RecordingStorage

router = APIRouter(tags=["actions"])
logger = get_logger("api.actions")


def get_action_service(request: Request) -> ActionService:
    return request.app.state.action_service


def get_recording_storage(request: Request) -> RecordingStorage:
    return request.app.state.recording_storage


def _raise_action_error(error: Exception, *, event: str) -> Never:
    if isinstance(error, ActionNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    if isinstance(error, ActionConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    if isinstance(error, ActionValidationError):
        logger.warning("%s reason=%s", event, error.reason)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    if isinstance(error, ActionStorageError):
        logger.error(
            "%s error_type=%s cause_type=%s",
            event,
            type(error).__name__,
            type(error.__cause__).__name__ if error.__cause__ else None,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not access Action data",
        ) from error
    raise error


@router.put(
    "/recordings/{recording_id}/action-candidates/{candidate_id}/review",
    response_model=RatingCandidateReviewResponse,
)
def review_record_rating_candidate(
    recording_id: UUID,
    candidate_id: str,
    payload: RatingCandidateReviewRequest,
    request: Request,
) -> RatingCandidateReviewResponse:
    try:
        get_recording_storage(request).find(recording_id)
        review = get_action_service(request).review_candidate(
            recording_id,
            candidate_id,
            payload,
        )
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except (
        ActionNotFoundError,
        ActionConflictError,
        ActionValidationError,
        ActionStorageError,
    ) as error:
        _raise_action_error(error, event="action_candidate_review_failed")

    logger.info(
        "action_candidate_reviewed recording_id=%s disposition=%s",
        recording_id,
        review.disposition,
    )
    return review


@router.get(
    "/recordings/{recording_id}/action-candidates/{candidate_id}/review",
    response_model=RatingCandidateReviewResponse,
)
def get_record_rating_candidate_review(
    recording_id: UUID,
    candidate_id: str,
    request: Request,
) -> RatingCandidateReviewResponse:
    try:
        get_recording_storage(request).find(recording_id)
        return get_action_service(request).load_candidate_review(recording_id, candidate_id)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except (ActionNotFoundError, ActionConflictError, ActionStorageError) as error:
        _raise_action_error(error, event="action_candidate_review_load_failed")


@router.get(
    "/recordings/{recording_id}/action-candidates/{candidate_id}/resolution",
    response_model=RatingCandidateResolutionResponse,
    summary="Resolve a Record rating candidate against current Actions",
    description=(
        "Reports whether the candidate needs setup, has already been reviewed, or can use an "
        "existing active ongoing Record rating Action for the same metric. An incompatible "
        "ongoing Action is returned as a conflict resolution and is never changed by this "
        "operation."
    ),
)
def resolve_record_rating_candidate(
    recording_id: UUID,
    candidate_id: str,
    request: Request,
) -> RatingCandidateResolutionResponse:
    try:
        get_recording_storage(request).find(recording_id)
        return get_action_service(request).resolve_candidate(recording_id, candidate_id)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except (ActionNotFoundError, ActionConflictError, ActionStorageError) as error:
        _raise_action_error(error, event="action_candidate_resolution_failed")


@router.get("/actions", response_model=RatingActionListResponse)
def list_rating_actions(
    request: Request,
    recording_id: Annotated[UUID | None, Query()] = None,
) -> RatingActionListResponse:
    try:
        actions = get_action_service(request).list_actions(recording_id)
    except ActionStorageError as error:
        _raise_action_error(error, event="actions_list_failed")
    return RatingActionListResponse(actions=actions)


@router.get("/actions/{action_id}", response_model=RatingAction)
def get_rating_action(action_id: UUID, request: Request) -> RatingAction:
    try:
        return get_action_service(request).load_action(action_id)
    except (ActionNotFoundError, ActionStorageError) as error:
        _raise_action_error(error, event="action_load_failed")


@router.delete(
    "/actions/{action_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Permanently delete a rating Action",
    description=(
        "Deletes the Action, all of its rating entries and drafts, their revision history, "
        "and the candidate review that activated it. The source recording, transcript, and "
        "session insights are preserved, so the candidate can be reviewed and activated again."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "The Action does not exist."},
        status.HTTP_409_CONFLICT: {
            "description": "The stored candidate review no longer matches the Action."
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "The Action data could not be deleted safely."
        },
    },
)
def delete_rating_action(action_id: UUID, request: Request) -> Response:
    try:
        action, entry_count = get_action_service(request).delete_action(action_id)
    except (ActionNotFoundError, ActionConflictError, ActionStorageError) as error:
        _raise_action_error(error, event="action_delete_failed")
    logger.info(
        "action_deleted action_id=%s recording_id=%s entry_count=%s",
        action_id,
        action.recording_id,
        entry_count,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/actions/{action_id}/entries/{entry_id}/draft", response_model=RatingEntry)
def save_rating_entry_draft(
    action_id: UUID,
    entry_id: UUID,
    payload: SaveRatingEntryDraftRequest,
    request: Request,
) -> RatingEntry:
    try:
        return get_action_service(request).save_draft(action_id, entry_id, payload)
    except (
        ActionNotFoundError,
        ActionConflictError,
        ActionValidationError,
        ActionStorageError,
    ) as error:
        _raise_action_error(error, event="rating_entry_draft_failed")


@router.post("/actions/{action_id}/entries/{entry_id}/confirm", response_model=RatingEntry)
def confirm_rating_entry(
    action_id: UUID,
    entry_id: UUID,
    payload: ConfirmRatingEntryRequest,
    request: Request,
) -> RatingEntry:
    try:
        return get_action_service(request).confirm_entry(action_id, entry_id, payload)
    except (
        ActionNotFoundError,
        ActionConflictError,
        ActionValidationError,
        ActionStorageError,
    ) as error:
        _raise_action_error(error, event="rating_entry_confirm_failed")


@router.patch("/actions/{action_id}/entries/{entry_id}", response_model=RatingEntry)
def amend_rating_entry(
    action_id: UUID,
    entry_id: UUID,
    payload: AmendRatingEntryRequest,
    request: Request,
) -> RatingEntry:
    try:
        return get_action_service(request).amend_entry(action_id, entry_id, payload)
    except (
        ActionNotFoundError,
        ActionConflictError,
        ActionValidationError,
        ActionStorageError,
    ) as error:
        _raise_action_error(error, event="rating_entry_amend_failed")


@router.delete(
    "/actions/{action_id}/entries/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_rating_entry(action_id: UUID, entry_id: UUID, request: Request) -> Response:
    try:
        get_action_service(request).delete_entry(action_id, entry_id)
    except (ActionNotFoundError, ActionStorageError) as error:
        _raise_action_error(error, event="rating_entry_delete_failed")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/actions/{action_id}/entries", response_model=RatingEntryListResponse)
def list_rating_entries(
    action_id: UUID,
    request: Request,
    date_from: Annotated[date | None, Query(alias="from")] = None,
    date_to: Annotated[date | None, Query(alias="to")] = None,
) -> RatingEntryListResponse:
    if date_from and date_to and date_to < date_from:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Entry date range is invalid",
        )
    try:
        entries = get_action_service(request).list_entries(action_id, date_from, date_to)
    except (ActionNotFoundError, ActionStorageError) as error:
        _raise_action_error(error, event="rating_entries_list_failed")
    return RatingEntryListResponse(entries=entries)


@router.post("/actions/{action_id}/entries/export", response_model=RatingEntryExportResponse)
def export_rating_entries(
    action_id: UUID,
    payload: RatingEntryExportRequest,
    request: Request,
) -> RatingEntryExportResponse:
    try:
        service = get_action_service(request)
        action = service.load_action(action_id)
        entries = service.export_entries(action_id, payload.entry_ids)
    except (ActionNotFoundError, ActionValidationError, ActionStorageError) as error:
        _raise_action_error(error, event="rating_entries_export_failed")
    return RatingEntryExportResponse(
        action=action,
        entries=entries,
        exported_at=datetime.now(UTC),
    )

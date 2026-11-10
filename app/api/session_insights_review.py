from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.session_insights_review import (
    SessionInsightsReviewRequest,
    SessionInsightsReviewResponse,
)
from app.services.recording_storage import RecordingNotFoundError, RecordingStorage
from app.services.session_insights_review import (
    InsightsReviewConflictError,
    InsightsReviewDraftNotFoundError,
    InsightsReviewNotFoundError,
    InsightsReviewStorageError,
    InsightsReviewValidationError,
    SessionInsightsReviewService,
)

router = APIRouter(prefix="/recordings", tags=["session insights review"])
logger = get_logger("api.session_insights_review")


def get_recording_storage(request: Request) -> RecordingStorage:
    return request.app.state.recording_storage


def get_review_service(request: Request) -> SessionInsightsReviewService:
    return request.app.state.session_insights_review_service


@router.get(
    "/{recording_id}/insights/review",
    response_model=SessionInsightsReviewResponse,
)
def get_session_insights_review(
    recording_id: UUID,
    request: Request,
) -> SessionInsightsReviewResponse:
    try:
        get_recording_storage(request).find(recording_id)
        review = get_review_service(request).load(recording_id)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except InsightsReviewNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except InsightsReviewDraftNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except InsightsReviewConflictError as error:
        logger.warning("insights_review_stale recording_id=%s", recording_id)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except InsightsReviewStorageError as error:
        logger.error(
            "insights_review_load_failed recording_id=%s error_type=%s cause_type=%s",
            recording_id,
            type(error).__name__,
            type(error.__cause__).__name__ if error.__cause__ else None,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not load reviewed session insights",
        ) from error

    logger.info(
        "insights_review_loaded recording_id=%s draft_id=%s",
        recording_id,
        review.draft_id,
    )
    return review


@router.put(
    "/{recording_id}/insights/review",
    response_model=SessionInsightsReviewResponse,
)
def save_session_insights_review(
    recording_id: UUID,
    payload: SessionInsightsReviewRequest,
    request: Request,
) -> SessionInsightsReviewResponse:
    try:
        get_recording_storage(request).find(recording_id)
        review = get_review_service(request).save(recording_id, payload)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except InsightsReviewDraftNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except InsightsReviewConflictError as error:
        logger.warning("insights_review_conflict recording_id=%s", recording_id)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except InsightsReviewValidationError as error:
        logger.warning(
            "insights_review_rejected recording_id=%s reason=%s",
            recording_id,
            error.reason,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except InsightsReviewStorageError as error:
        logger.error(
            "insights_review_save_failed recording_id=%s error_type=%s cause_type=%s",
            recording_id,
            type(error).__name__,
            type(error.__cause__).__name__ if error.__cause__ else None,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not save reviewed session insights",
        ) from error

    items = [item for section in review.sections for item in section.items]
    logger.info(
        "insights_review_saved recording_id=%s draft_id=%s retained_count=%s "
        "edited_count=%s removed_count=%s",
        recording_id,
        review.draft_id,
        sum(item.disposition == "retained" for item in items),
        sum(item.disposition == "edited" for item in items),
        sum(item.disposition == "removed" for item in items),
    )
    return review

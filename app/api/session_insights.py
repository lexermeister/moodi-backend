from time import monotonic
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.session_insights import SessionInsightsRequest, SessionInsightsResponse
from app.services.recording_storage import RecordingNotFoundError, RecordingStorage
from app.services.session_insights import (
    InsightsInProgressError,
    InsightsNotConfiguredError,
    InsightsProviderError,
    InsightsStorageError,
    InsightsTimeoutError,
    InsightsValidationError,
    SessionInsightsService,
    TranscriptNotFoundForInsightsError,
)

router = APIRouter(prefix="/recordings", tags=["session insights"])
logger = get_logger("api.session_insights")


def get_recording_storage(request: Request) -> RecordingStorage:
    return request.app.state.recording_storage


def get_session_insights_service(request: Request) -> SessionInsightsService:
    return request.app.state.session_insights_service


@router.post(
    "/{recording_id}/insights",
    response_model=SessionInsightsResponse,
    responses={
        409: {"description": "Transcript missing or insights already being generated"},
        504: {"description": "Insights provider timed out; safe to retry"},
    },
)
def create_session_insights(
    recording_id: UUID,
    payload: SessionInsightsRequest,
    request: Request,
) -> SessionInsightsResponse:
    started_at = monotonic()
    logger.info(
        "Processing a request for session insights. | event=insights_request_started "
        "recording_id=%s",
        recording_id,
    )

    try:
        get_recording_storage(request).find(recording_id)
        completed = get_session_insights_service(request).generate(
            recording_id,
            payload.session_date,
        )
    except RecordingNotFoundError as error:
        logger.warning(
            "Cannot generate session insights because the recording was not found. | "
            "event=insights_rejected recording_id=%s reason=recording_not_found",
            recording_id,
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except TranscriptNotFoundForInsightsError as error:
        logger.warning(
            "Cannot generate session insights until a transcript has been saved. | "
            "event=insights_rejected recording_id=%s reason=transcript_not_found",
            recording_id,
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except InsightsNotConfiguredError as error:
        logger.error(
            "Session insights are unavailable because OpenAI is not configured. | "
            "event=insights_unavailable recording_id=%s reason=not_configured",
            recording_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except InsightsInProgressError as error:
        logger.info(
            "Session insights are already being generated; rejected the duplicate request. | "
            "event=insights_in_progress recording_id=%s",
            recording_id,
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except InsightsTimeoutError as error:
        logger.error(
            "Session insights request timed out; the client can retry. | "
            "event=insights_request_timeout recording_id=%s duration_ms=%s",
            recording_id,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Session insights took too long. Please try again.",
        ) from error
    except InsightsProviderError as error:
        logger.error(
            "Could not generate session insights because the OpenAI request failed. | "
            "event=insights_provider_failed recording_id=%s provider_error_type=%s status_code=%s "
            "request_id=%s duration_ms=%s",
            recording_id,
            error.provider_error_type,
            error.status_code,
            error.request_id,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate session insights",
        ) from error
    except InsightsValidationError as error:
        logger.error(
            "Session insights generation failed validation; no new insights were saved. | "
            "event=insights_generation_invalid recording_id=%s reason=%s duration_ms=%s",
            recording_id,
            error.reason,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate session insights",
        ) from error
    except InsightsStorageError as error:
        logger.error(
            "Could not read or save session insights in local storage. | "
            "event=insights_storage_failed recording_id=%s error_type=%s cause_type=%s",
            recording_id,
            type(error).__name__,
            type(error.__cause__).__name__ if error.__cause__ else None,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate session insights",
        ) from error

    logger.info(
        "Session insights request completed; returning the saved insights. | "
        "event=insights_request_completed recording_id=%s source_version=%s prompt_version=%s "
        "model=%s duration_ms=%s",
        recording_id,
        completed.source_version,
        completed.prompt_version,
        completed.model,
        round((monotonic() - started_at) * 1000),
    )
    return completed.response()

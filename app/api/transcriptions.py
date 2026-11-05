from time import monotonic
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.database import get_database_session
from app.core.logging import get_logger
from app.schemas.transcriptions import TranscriptionLanguage, TranscriptionResponse
from app.services.recording_service import RecordingNotFoundError, RecordingService
from app.services.recording_storage import RecordingStorage
from app.services.transcription import (
    NoSpeechDetectedError,
    RecordingTooLargeForTranscriptionError,
    TranscriptionNotConfiguredError,
    TranscriptionPersistenceError,
    TranscriptionProviderError,
    TranscriptionService,
    UnsupportedTranscriptionFormatError,
)

router = APIRouter(prefix="/recordings", tags=["transcriptions"])
logger = get_logger("api.transcriptions")


def get_recording_storage(request: Request) -> RecordingStorage:
    return request.app.state.recording_storage


def get_transcription_service(request: Request) -> TranscriptionService:
    return request.app.state.transcription_service


@router.post("/{recording_id}/transcription", response_model=TranscriptionResponse)
def create_transcription(
    recording_id: UUID,
    request: Request,
    session: Annotated[Session, Depends(get_database_session)],
) -> TranscriptionResponse:
    started_at = monotonic()
    logger.info("transcription_started recording_id=%s", recording_id)

    try:
        recording_service = RecordingService(get_recording_storage(request), session)
        _, recording_path = recording_service.find(recording_id)
        completed = get_transcription_service(request).transcribe(
            recording_id,
            recording_path,
            session,
        )
    except RecordingNotFoundError as error:
        logger.warning("transcription_rejected recording_id=%s reason=not_found", recording_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except RecordingTooLargeForTranscriptionError as error:
        logger.warning("transcription_rejected recording_id=%s reason=too_large", recording_id)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error
    except UnsupportedTranscriptionFormatError as error:
        logger.warning(
            "transcription_rejected recording_id=%s reason=unsupported_media_type",
            recording_id,
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error
    except TranscriptionNotConfiguredError as error:
        logger.error(
            "transcription_unavailable recording_id=%s reason=not_configured", recording_id
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except NoSpeechDetectedError as error:
        logger.warning(
            "transcription_rejected recording_id=%s transcription_id=%s "
            "provider_request_id=%s reason=no_speech duration_ms=%s",
            recording_id,
            error.transcription_id,
            error.provider_request_id,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except TranscriptionProviderError as error:
        logger.error(
            "transcription_provider_failed recording_id=%s transcription_id=%s "
            "provider_error_type=%s status_code=%s provider_request_id=%s duration_ms=%s",
            recording_id,
            error.transcription_id,
            error.provider_error_type,
            error.status_code,
            error.request_id,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not transcribe recording",
        ) from error
    except TranscriptionPersistenceError as error:
        logger.error(
            "transcription_save_failed recording_id=%s transcription_id=%s "
            "error_type=%s cause_type=%s",
            recording_id,
            error.transcription_id,
            type(error).__name__,
            type(error.__cause__).__name__ if error.__cause__ else None,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not transcribe recording",
        ) from error

    logger.info(
        "transcription_completed recording_id=%s transcription_id=%s "
        "provider_request_id=%s model=%s language_count=%s duration_ms=%s",
        recording_id,
        completed.transcription_id,
        completed.provider_request_id,
        completed.model,
        len(completed.languages),
        round((monotonic() - started_at) * 1000),
    )

    return TranscriptionResponse(
        transcription_id=completed.transcription_id,
        recording_id=completed.recording_id,
        text=completed.text,
        languages=[TranscriptionLanguage(code=code) for code in completed.languages],
        model=completed.model,
        status="completed",
        completed_at=completed.completed_at,
    )

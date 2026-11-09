from time import monotonic
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.transcriptions import TranscriptionLanguage, TranscriptionResponse, TranscriptTurn
from app.services.recording_storage import RecordingNotFoundError, RecordingStorage
from app.services.transcript_store import TranscriptStorageError
from app.services.transcription import (
    AmbiguousRoleAttributionError,
    DiarizationValidationError,
    NoSpeechDetectedError,
    RecordingTooLargeForTranscriptionError,
    RoleAttributionProviderError,
    TranscriptionNotConfiguredError,
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
def create_transcription(recording_id: UUID, request: Request) -> TranscriptionResponse:
    started_at = monotonic()
    logger.info("transcription_started recording_id=%s", recording_id)

    try:
        recording_path = get_recording_storage(request).find(recording_id)
        completed = get_transcription_service(request).transcribe(recording_id, recording_path)
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
            "transcription_rejected recording_id=%s reason=no_speech duration_ms=%s",
            recording_id,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except DiarizationValidationError as error:
        logger.warning(
            "transcription_rejected recording_id=%s reason=%s duration_ms=%s",
            recording_id,
            error.reason,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except AmbiguousRoleAttributionError as error:
        logger.warning(
            "role_attribution_ambiguous recording_id=%s duration_ms=%s",
            recording_id,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except RoleAttributionProviderError as error:
        logger.error(
            "role_attribution_provider_failed recording_id=%s provider_error_type=%s "
            "status_code=%s request_id=%s duration_ms=%s",
            recording_id,
            error.provider_error_type,
            error.status_code,
            error.request_id,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not assign transcript roles",
        ) from error
    except TranscriptionProviderError as error:
        logger.error(
            "transcription_provider_failed recording_id=%s provider_error_type=%s "
            "status_code=%s request_id=%s duration_ms=%s",
            recording_id,
            error.provider_error_type,
            error.status_code,
            error.request_id,
            round((monotonic() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not transcribe recording",
        ) from error
    except TranscriptStorageError as error:
        logger.error(
            "transcript_save_failed recording_id=%s error_type=%s cause_type=%s",
            recording_id,
            type(error).__name__,
            type(error.__cause__).__name__ if error.__cause__ else None,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not transcribe recording",
        ) from error

    logger.info(
        "transcription_completed recording_id=%s model=%s role_model=%s "
        "speaker_count=%s turn_count=%s language_count=%s duration_ms=%s",
        recording_id,
        completed.model,
        completed.role_model,
        len({turn.source_speaker for turn in completed.turns}),
        len(completed.turns),
        len(completed.languages),
        round((monotonic() - started_at) * 1000),
    )

    return TranscriptionResponse(
        recording_id=completed.recording_id,
        text=completed.text,
        turns=[TranscriptTurn(role=turn.role, text=turn.text) for turn in completed.turns],
        languages=[TranscriptionLanguage(code=code) for code in completed.languages],
        model=completed.model,
        role_model=completed.role_model,
        role_prompt_version=completed.role_prompt_version,
        status="completed",
        completed_at=completed.completed_at,
    )

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.transcriptions import TranscriptionLanguage, TranscriptionResponse
from app.services.recording_storage import RecordingNotFoundError, RecordingStorage
from app.services.transcription import (
    RecordingTooLargeForTranscriptionError,
    TranscriptionNotConfiguredError,
    TranscriptionProviderError,
    TranscriptionService,
    TranscriptStorageError,
    UnsupportedTranscriptionFormatError,
)

router = APIRouter(prefix="/recordings", tags=["transcriptions"])


def get_recording_storage(request: Request) -> RecordingStorage:
    return request.app.state.recording_storage


def get_transcription_service(request: Request) -> TranscriptionService:
    return request.app.state.transcription_service


@router.post("/{recording_id}/transcription", response_model=TranscriptionResponse)
def create_transcription(recording_id: UUID, request: Request) -> TranscriptionResponse:
    try:
        recording_path = get_recording_storage(request).find(recording_id)
        completed = get_transcription_service(request).transcribe(recording_id, recording_path)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except RecordingTooLargeForTranscriptionError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error
    except UnsupportedTranscriptionFormatError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error
    except TranscriptionNotConfiguredError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except (TranscriptionProviderError, TranscriptStorageError) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not transcribe recording",
        ) from error

    return TranscriptionResponse(
        recording_id=completed.recording_id,
        text=completed.text,
        languages=[TranscriptionLanguage(code=code) for code in completed.languages],
        model=completed.model,
        status="completed",
        completed_at=completed.completed_at,
    )

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.core.logging import get_logger
from app.schemas.recordings import RecordingResponse
from app.services.recording_storage import (
    EmptyRecordingError,
    InvalidRecordingTypeError,
    RecordingStorage,
    RecordingStorageError,
    RecordingTooLargeError,
)

router = APIRouter(prefix="/recordings", tags=["recordings"])
logger = get_logger("api.recordings")


def get_recording_storage(request: Request) -> RecordingStorage:
    return request.app.state.recording_storage


@router.post("", response_model=RecordingResponse, status_code=status.HTTP_201_CREATED)
def create_recording(
    request: Request,
    file: Annotated[UploadFile, File(description="M4A, AAC, or WAV audio file")],
) -> RecordingResponse:
    storage = get_recording_storage(request)

    try:
        saved = storage.save(file)
    except EmptyRecordingError as error:
        logger.warning("recording_upload_rejected reason=empty")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except RecordingTooLargeError as error:
        logger.warning("recording_upload_rejected reason=too_large")
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error
    except InvalidRecordingTypeError as error:
        logger.warning("recording_upload_rejected reason=unsupported_media_type")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error
    except RecordingStorageError as error:
        logger.error(
            "recording_save_failed error_type=%s cause_type=%s",
            type(error).__name__,
            type(error.__cause__).__name__ if error.__cause__ else None,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save recording",
        ) from error
    finally:
        file.file.close()

    logger.info(
        "recording_saved recording_id=%s extension=%s size_bytes=%s",
        saved.recording_id,
        saved.filename.rsplit(".", maxsplit=1)[-1],
        saved.size_bytes,
    )

    return RecordingResponse(
        id=saved.recording_id,
        filename=saved.filename,
        size_bytes=saved.size_bytes,
        status="saved",
    )

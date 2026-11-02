from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.recordings import RecordingResponse
from app.services.recording_storage import (
    EmptyRecordingError,
    InvalidRecordingTypeError,
    RecordingStorage,
    RecordingStorageError,
    RecordingTooLargeError,
)

router = APIRouter(prefix="/recordings", tags=["recordings"])


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except RecordingTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error
    except InvalidRecordingTypeError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error
    except RecordingStorageError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save recording",
        ) from error
    finally:
        file.file.close()

    return RecordingResponse(
        id=saved.recording_id,
        filename=saved.filename,
        size_bytes=saved.size_bytes,
        status="saved",
    )

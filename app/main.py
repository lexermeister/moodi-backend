from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, recordings
from app.core.config import Settings, get_settings
from app.services.recording_storage import RecordingStorage


def create_app(settings: Settings | None = None) -> FastAPI:
    application_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        storage = RecordingStorage(
            directory=application_settings.recordings_path,
            max_size_bytes=application_settings.max_upload_bytes,
        )
        storage.prepare()
        application.state.recording_storage = storage
        yield

    application = FastAPI(title="Moodi API", lifespan=lifespan)
    application.include_router(health.router)
    application.include_router(recordings.router)
    return application


app = create_app()

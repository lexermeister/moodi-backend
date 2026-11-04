from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, recordings, transcriptions
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.services.recording_storage import RecordingStorage
from app.services.transcription import TranscriptionClient, TranscriptionService

logger = get_logger("lifecycle")


def create_app(
    settings: Settings | None = None,
    transcription_client: TranscriptionClient | None = None,
) -> FastAPI:
    application_settings = settings or get_settings()
    configure_logging(application_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        storage = RecordingStorage(
            directory=application_settings.recordings_path,
            max_size_bytes=application_settings.max_upload_bytes,
        )
        storage.prepare()
        application.state.recording_storage = storage
        application.state.transcription_service = TranscriptionService(
            directory=application_settings.recordings_path,
            api_key=application_settings.openai_api_key,
            model=application_settings.openai_transcription_model,
            client=transcription_client,
        )
        logger.info(
            "application_started app_env=%s recordings_dir=%s max_upload_mb=%s "
            "transcription_model=%s openai_configured=%s",
            application_settings.app_env,
            application_settings.recordings_path,
            application_settings.max_upload_mb,
            application_settings.openai_transcription_model,
            bool(application_settings.openai_api_key),
        )
        try:
            yield
        finally:
            logger.info("application_stopped")

    application = FastAPI(title="Moodi API", lifespan=lifespan)
    application.include_router(health.router)
    application.include_router(recordings.router)
    application.include_router(transcriptions.router)
    return application


app = create_app()

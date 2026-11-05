from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import Engine, text

from app.api import health, recordings, transcriptions
from app.core.config import Settings, get_settings
from app.core.database import create_database_engine, create_session_factory
from app.core.logging import configure_logging, get_logger
from app.services.recording_storage import RecordingStorage
from app.services.transcription import TranscriptionClient, TranscriptionService

logger = get_logger("lifecycle")


def create_app(
    settings: Settings | None = None,
    transcription_client: TranscriptionClient | None = None,
    database_engine: Engine | None = None,
) -> FastAPI:
    application_settings = settings or get_settings()
    configure_logging(application_settings.log_level)
    engine = database_engine or create_database_engine(application_settings.database_url)
    owns_engine = database_engine is None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        storage = RecordingStorage(
            directory=application_settings.recordings_path,
            max_size_bytes=application_settings.max_upload_bytes,
        )
        storage.prepare()
        application.state.recording_storage = storage
        application.state.session_factory = create_session_factory(engine)
        application.state.transcription_service = TranscriptionService(
            api_key=application_settings.openai_api_key,
            model=application_settings.openai_transcription_model,
            client=transcription_client,
        )
        logger.info(
            "application_started app_env=%s recordings_dir=%s max_upload_mb=%s "
            "database_backend=%s transcription_model=%s openai_configured=%s",
            application_settings.app_env,
            application_settings.recordings_path,
            application_settings.max_upload_mb,
            engine.url.get_backend_name(),
            application_settings.openai_transcription_model,
            bool(application_settings.openai_api_key),
        )
        try:
            yield
        finally:
            logger.info("application_stopped")
            if owns_engine:
                engine.dispose()

    application = FastAPI(title="Moodi API", lifespan=lifespan)
    application.include_router(health.router)
    application.include_router(recordings.router)
    application.include_router(transcriptions.router)
    return application


app = create_app()

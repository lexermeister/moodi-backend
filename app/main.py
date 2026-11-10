from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api import (
    actions,
    health,
    recordings,
    session_insights,
    session_insights_review,
    transcriptions,
)
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.services.actions import ActionService
from app.services.recording_storage import RecordingStorage
from app.services.session_insights import SessionInsightsService
from app.services.session_insights_review import SessionInsightsReviewService
from app.services.transcript_store import TranscriptStore
from app.services.transcription import (
    OpenAISessionInsightsGenerator,
    TranscriptionClient,
    TranscriptionService,
)

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
        transcript_store = TranscriptStore(application_settings.recordings_path)
        application.state.transcription_service = TranscriptionService(
            transcript_store=transcript_store,
            api_key=application_settings.openai_api_key,
            model=application_settings.openai_transcription_model,
            role_model=application_settings.openai_role_attribution_model,
            client=transcription_client,
        )
        insights_generator = OpenAISessionInsightsGenerator(
            api_key=application_settings.openai_api_key,
            model=application_settings.openai_insights_model,
            reasoning_effort=application_settings.openai_insights_reasoning_effort,
            client=transcription_client,
        )
        application.state.session_insights_service = SessionInsightsService(
            directory=application_settings.recordings_path,
            transcript_store=transcript_store,
            generator=insights_generator,
            prompt_path=Path(__file__).resolve().parent / "prompts" / "session_insights_v0_4.txt",
        )
        application.state.session_insights_review_service = SessionInsightsReviewService(
            directory=application_settings.recordings_path,
            insights_service=application.state.session_insights_service,
        )
        action_service = ActionService(
            directory=application_settings.recordings_path,
            insights_service=application.state.session_insights_service,
        )
        application.state.action_service = action_service
        logger.info(
            "application_started app_env=%s recordings_dir=%s max_upload_mb=%s "
            "transcription_model=%s role_attribution_model=%s insights_model=%s "
            "insights_reasoning_effort=%s "
            "openai_configured=%s",
            application_settings.app_env,
            application_settings.recordings_path,
            application_settings.max_upload_mb,
            application_settings.openai_transcription_model,
            application_settings.openai_role_attribution_model,
            application_settings.openai_insights_model,
            application_settings.openai_insights_reasoning_effort,
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
    application.include_router(session_insights.router)
    application.include_router(session_insights_review.router)
    application.include_router(actions.router)
    return application


app = create_app()

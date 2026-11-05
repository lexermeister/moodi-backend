from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Transcription(Base):
    __tablename__ = "transcriptions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('started', 'completed', 'failed')",
            name="ck_transcriptions_status",
        ),
        CheckConstraint(
            "(status = 'started' AND completed_at IS NULL) OR "
            "(status IN ('completed', 'failed') AND completed_at IS NOT NULL)",
            name="ck_transcriptions_completion_time",
        ),
        CheckConstraint(
            "status != 'completed' OR (text IS NOT NULL AND error_code IS NULL)",
            name="ck_transcriptions_completed_result",
        ),
        CheckConstraint(
            "status != 'failed' OR (text IS NULL AND error_code IS NOT NULL)",
            name="ck_transcriptions_failed_result",
        ),
        Index("ix_transcriptions_recording_started", "recording_id", "started_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    recording_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    provider_request_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    languages: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

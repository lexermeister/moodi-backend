"""Create recordings and transcriptions tables.

Revision ID: 0001
Revises:
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "recordings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("format", sa.String(length=10), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "format IN ('aac', 'm4a', 'wav')",
            name="ck_recordings_format",
        ),
        sa.CheckConstraint("size_bytes > 0", name="ck_recordings_size_bytes_positive"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_table(
        "transcriptions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("recording_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("provider_request_id", sa.String(length=255), nullable=True),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("languages", sa.JSON(), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('started', 'completed', 'failed')",
            name="ck_transcriptions_status",
        ),
        sa.CheckConstraint(
            "(status = 'started' AND completed_at IS NULL) OR "
            "(status IN ('completed', 'failed') AND completed_at IS NOT NULL)",
            name="ck_transcriptions_completion_time",
        ),
        sa.CheckConstraint(
            "status != 'completed' OR (text IS NOT NULL AND error_code IS NULL)",
            name="ck_transcriptions_completed_result",
        ),
        sa.CheckConstraint(
            "status != 'failed' OR (text IS NULL AND error_code IS NOT NULL)",
            name="ck_transcriptions_failed_result",
        ),
        sa.ForeignKeyConstraint(["recording_id"], ["recordings.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_transcriptions_provider_request_id",
        "transcriptions",
        ["provider_request_id"],
        unique=False,
    )
    op.create_index(
        "ix_transcriptions_recording_started",
        "transcriptions",
        ["recording_id", "started_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_transcriptions_recording_started", table_name="transcriptions")
    op.drop_index("ix_transcriptions_provider_request_id", table_name="transcriptions")
    op.drop_table("transcriptions")
    op.drop_table("recordings")

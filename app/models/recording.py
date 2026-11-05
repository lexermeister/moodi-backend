from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, DateTime, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Recording(Base):
    __tablename__ = "recordings"
    __table_args__ = (
        CheckConstraint("format IN ('aac', 'm4a', 'wav')", name="ck_recordings_format"),
        CheckConstraint("size_bytes > 0", name="ck_recordings_size_bytes_positive"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    format: Mapped[str] = mapped_column(String(10), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

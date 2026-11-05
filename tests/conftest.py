from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from app.core.config import Settings
from app.core.database import Base, create_database_engine
from app.main import create_app
from app.models import Recording, Transcription  # noqa: F401


@pytest.fixture
def recordings_dir(tmp_path: Path) -> Path:
    return tmp_path / "recordings"


@pytest.fixture
def database_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_database_engine(f"sqlite+pysqlite:///{tmp_path / 'moodi.db'}")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def client(recordings_dir: Path, database_engine: Engine) -> Iterator[TestClient]:
    settings = Settings(
        app_env="testing",
        database_url="sqlite+pysqlite://",
        recordings_dir=recordings_dir,
        max_upload_mb=1,
    )
    with TestClient(create_app(settings, database_engine=database_engine)) as test_client:
        yield test_client

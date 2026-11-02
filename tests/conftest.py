from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def recordings_dir(tmp_path: Path) -> Path:
    return tmp_path / "recordings"


@pytest.fixture
def client(recordings_dir: Path) -> Iterator[TestClient]:
    settings = Settings(recordings_dir=recordings_dir, max_upload_mb=1)
    with TestClient(create_app(settings)) as test_client:
        yield test_client

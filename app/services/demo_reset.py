import re
import shutil
from pathlib import Path
from uuid import UUID


class DemoResetError(Exception):
    """Raised when demo data cannot be removed completely."""


_RECORDING_FILE = re.compile(
    r"^(?P<id>[0-9a-f-]{36})(?:\.m4a|\.aac|\.wav|\.transcript\.json|"
    r"\.insights\.json|\.insights-review\.json)$"
)
_TEMPORARY_FILE = re.compile(
    r"^\.(?P<id>[0-9a-f-]{36})(?:\.tmp|\.transcript\.tmp|"
    r"\.insights\.tmp|\.insights-review\.tmp)$"
)


class DemoResetService:
    def __init__(self, directory: Path, fixture_path: Path | None) -> None:
        self.directory = directory
        self.fixture_path = fixture_path.resolve() if fixture_path else None

    @staticmethod
    def _is_recording_file(name: str) -> bool:
        match = _RECORDING_FILE.fullmatch(name) or _TEMPORARY_FILE.fullmatch(name)
        if match is None:
            return False
        try:
            return str(UUID(match.group("id"))) == match.group("id")
        except ValueError:
            return False

    def reset(self) -> None:
        actions = self.directory / "actions"
        try:
            if self.fixture_path and self.fixture_path.is_relative_to(actions.resolve()):
                raise DemoResetError("Transcript fixture is inside the Action storage directory")
            if actions.is_symlink():
                raise DemoResetError("Action storage is a symbolic link")
            if actions.exists():
                shutil.rmtree(actions)
            if not self.directory.exists():
                return
            for path in self.directory.iterdir():
                if not self._is_recording_file(path.name):
                    continue
                if self.fixture_path and path.resolve() == self.fixture_path:
                    continue
                path.unlink()
        except OSError as error:
            raise DemoResetError("Could not clear demo data") from error

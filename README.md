# Moodi backend

Small FastAPI service that accepts audio recordings from the Moodi mobile app and saves them
locally. This first milestone supports M4A, AAC, and WAV uploads and does not perform any
downstream audio processing.

## Setup

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
uv sync --locked
```

Relative `RECORDINGS_DIR` values are resolved from the repository root. The directory is created
when the application starts.

## Start the API

```bash
uv run uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Check it with:

```bash
curl http://127.0.0.1:8000/health
```

## Upload a recording

```bash
curl -X POST http://127.0.0.1:8000/recordings \
  -F "file=@/path/to/recording.m4a;type=audio/mp4"
```

The request returns only after the recording has been written and finalized in the configured
storage directory.

## Verify changes

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```


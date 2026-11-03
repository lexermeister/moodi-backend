# Moodi backend agent guide

## Project layout

- `app/main.py`: FastAPI application factory and startup lifecycle.
- `app/api/`: HTTP route definitions.
- `app/core/`: environment-backed application configuration.
- `app/schemas/`: API response models.
- `app/services/`: filesystem recording storage and validation.
- `tests/`: pytest API and storage behavior tests.
- `data/recordings/`: runtime upload storage; intentionally ignored by Git.

Keep route handlers small and put filesystem behavior in `app/services/`. Never use a client
filename as a path. Upload endpoints must not report success before durable local writing and
atomic finalization complete.

OpenAI integration belongs in `app/services/transcription.py`. Keep provider errors out of API
responses, never log or commit `OPENAI_API_KEY`, and mock the OpenAI client in automated tests.
Transcription is synchronous for now and supports one-speaker M4A and WAV recordings up to 25 MB.

## Verification

Run all checks before handing off changes:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```


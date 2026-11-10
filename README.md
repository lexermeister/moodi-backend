# Moodi backend

Moodi is a small FastAPI service that receives audio recordings from the Moodi mobile app, saves
them to local storage, and transcribes one- or two-speaker M4A or WAV recordings with OpenAI
`gpt-4o-transcribe-diarize`. A second structured model call maps the diarized speakers to
`client` and `therapist` roles. A third structured call produces evidence-linked, client-owned
session insights from the saved transcript.
Upload, transcription, and insight requests are synchronous: their responses are returned only
after the requested artifact has been saved locally.

This milestone intentionally has no authentication, database, Supabase integration, queue,
worker, reminders, or clinical-record integration.

## What is implemented

- `GET /health` for service health checks.
- `POST /recordings` for multipart audio uploads.
- M4A, AAC, and WAV extension and MIME-type screening.
- UUID-based server filenames; client filenames are never used as filesystem paths.
- Chunked copying so the complete upload is not loaded into application memory.
- Configurable upload-size enforcement with HTTP `413` responses.
- Empty-upload rejection.
- Temporary-file writing and atomic finalization.
- Partial-file cleanup after failed uploads.
- Environment-based settings with `.env` support.
- Pytest coverage for the health endpoint and upload behavior.
- Ruff linting and formatting.
- Docker and Docker Compose support with persistent local recording storage.
- GitHub Actions checks for tests, linting, and formatting.
- `POST /recordings/{recording_id}/transcription` for one- or two-speaker transcription.
- OpenAI `gpt-4o-transcribe-diarize` integration with normalized speaker turns.
- Structured role attribution that returns `client` and `therapist` turns or fails as ambiguous.
- Atomic local transcript JSON storage.
- `POST /recordings/{recording_id}/insights` for evidence-linked session insights.
- `GET` and `PUT /recordings/{recording_id}/insights/review` for a separate client-reviewed layer.
- Idempotent transcript and insight generation backed by versioned local artifacts.
- Application-side validation of model-generated evidence and cross references.
- Provider-error handling that does not expose OpenAI error details.
- Structured application logs for lifecycle, uploads, and transcription outcomes.

File extensions and MIME types are only basic screening. They do not prove that an uploaded file
contains valid audio.

## Project layout

```text
app/
  main.py                  FastAPI application and startup lifecycle
  api/
    health.py              Health endpoint
    recordings.py          Recording upload endpoint
    transcriptions.py      Saved-recording transcription endpoint
    session_insights.py    Saved-transcript insight endpoint
    session_insights_review.py  Reviewed-insight endpoints
  core/
    config.py              Environment-backed settings
    logging.py             Application logger setup
  schemas/
    recordings.py          Recording response model
    transcriptions.py      Transcription response models
    session_insights.py    Structured insight request and response models
    session_insights_review.py  Reviewed-insight request and response models
  services/
    recording_storage.py   Upload validation and filesystem storage
    transcript_store.py    Atomic completed-transcript persistence
    transcription.py       OpenAI transcription, role, and insight provider calls
    session_insights.py    Insight validation, caching, and persistence
    session_insights_review.py  Reviewed-insight validation and persistence
  prompts/
    session_insights_v0_3.txt  Versioned insight-generation instructions
tests/                     Automated tests
Dockerfile                 Production container image
compose.yaml               Local container setup and recordings volume
```

## Configuration

Copy the example settings for local development:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Identifies the running environment. |
| `LOG_LEVEL` | `INFO` | Minimum application log level. |
| `RECORDINGS_DIR` | `./data/recordings` | Directory where finalized recordings are stored. |
| `MAX_UPLOAD_MB` | `100` | Maximum accepted upload size in MiB. |
| `OPENAI_API_KEY` | none | OpenAI API key used only by the backend. |
| `OPENAI_TRANSCRIPTION_MODEL` | `gpt-4o-transcribe-diarize` | OpenAI diarization model. |
| `OPENAI_ROLE_ATTRIBUTION_MODEL` | `gpt-4o-mini` | Model that maps diarized speakers to client and therapist roles. |
| `OPENAI_INSIGHTS_MODEL` | `gpt-6-luna` | Model that produces structured session insights. |
| `OPENAI_INSIGHTS_REASONING_EFFORT` | `none` | Reasoning effort for session insights; `none` favors low latency. |
| `PORT` | `8000` in Docker | HTTP port used by the container. Hosting platforms can override it. |

Relative recording paths are resolved against the repository root. The application creates the
recordings directory during startup.

Never commit `.env`; it is intentionally ignored by Git and Docker.

Create an API key in the OpenAI API platform and place it in your local `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
```

Audio sent for transcription leaves the local server and is processed by OpenAI. Do not use real
therapy recordings until the project's consent, privacy, retention, and production security
requirements have been decided.

## Run locally with Python

Requirements:

- Python 3.13
- [uv](https://docs.astral.sh/uv/)

Install the locked dependencies:

```bash
uv sync --locked
```

Start the development server:

```bash
uv run uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`.

- Health check: `http://127.0.0.1:8000/health`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Stop the server with `Ctrl+C`.

## API

### Health check

```http
GET /health
```

Successful response (`200`):

```json
{
  "status": "ok"
}
```

### Upload a recording

```http
POST /recordings
Content-Type: multipart/form-data
```

The multipart request must contain a field named `file`. Supported extensions are `.m4a`, `.aac`,
and `.wav`, with a matching audio MIME type.

Example:

```bash
curl -X POST http://127.0.0.1:8000/recordings \
  -F "file=@/path/to/recording.m4a;type=audio/mp4"
```

Successful response (`201`):

```json
{
  "id": "8fb3c678-5b52-4a57-b51d-a8d0b77ae193",
  "filename": "8fb3c678-5b52-4a57-b51d-a8d0b77ae193.m4a",
  "size_bytes": 12345,
  "status": "saved"
}
```

Important error responses:

| Status | Meaning |
| --- | --- |
| `400` | The uploaded file is empty. |
| `413` | The upload exceeds `MAX_UPLOAD_MB`. |
| `415` | The extension or MIME type is unsupported or mismatched. |
| `422` | The required multipart `file` field is missing. |
| `500` | The recording could not be saved. |

### How an upload is stored

The upload request is handled synchronously from the mobile client's perspective:

1. FastAPI receives the multipart `file` field as an `UploadFile`.
2. The extension and MIME type are checked against the supported M4A, AAC, and WAV values.
3. The server generates a UUID and uses it for both the recording ID and stored filename.
4. The client filename is used only to read its extension; it is never used as a path.
5. The upload is copied in 1 MiB chunks to a hidden temporary file.
6. The configured size limit is checked while chunks are copied.
7. Empty and oversized uploads are rejected and their temporary files are removed.
8. A successful write is flushed and synchronized to disk.
9. The temporary file is atomically renamed to its final UUID filename.
10. Only then does the endpoint return HTTP `201` with `status: saved`.

The request flow stays intentionally simple:

```text
HTTP request -> API route -> recording storage service -> response schema
```

### Transcribe a saved recording

```http
POST /recordings/{recording_id}/transcription
```

Upload a recording first and copy the returned `id`:

```bash
curl -X POST http://127.0.0.1:8000/recordings \
  -F "file=@/path/to/recording.m4a;type=audio/mp4"
```

Then request its transcription:

```bash
curl -X POST \
  http://127.0.0.1:8000/recordings/8fb3c678-5b52-4a57-b51d-a8d0b77ae193/transcription
```

Successful response (`200`):

```json
{
  "recording_id": "8fb3c678-5b52-4a57-b51d-a8d0b77ae193",
  "text": "Kako se osećate? Osećam se anksiozno.",
  "turns": [
    {
      "role": "therapist",
      "text": "Kako se osećate?"
    },
    {
      "role": "client",
      "text": "Osećam se anksiozno."
    }
  ],
  "languages": [],
  "model": "gpt-4o-transcribe-diarize",
  "role_model": "gpt-4o-mini",
  "role_prompt_version": "v1",
  "status": "completed",
  "completed_at": "2026-11-03T12:00:00Z"
}
```

The endpoint:

1. Accepts only a UUID recording ID, never a client-provided filesystem path.
2. Locates the previously saved recording inside `RECORDINGS_DIR`.
3. Rejects files larger than OpenAI's 25 MB file-transcription limit before making an API call.
4. Requests diarized JSON from `gpt-4o-transcribe-diarize` with automatic audio chunking.
5. Normalizes the first two distinct provider speakers to `speaker_1` and `speaker_2`, then merges
   adjacent segments from the same speaker.
6. Treats a single labeled speaker as the `client`. For two labeled speakers, it uses a structured
   LLM response to assign exactly one `therapist` and one `client` role without changing the text.
7. Returns only after saving
   `<uuid>.transcript.json` atomically beside the audio file.

Raw AAC remains accepted by the upload endpoint, but this transcription milestone rejects `.aac`
with HTTP `415` because it is not in OpenAI's documented file-transcription input list. No audio
conversion or file splitting is performed yet.

Important transcription error responses:

| Status | Meaning |
| --- | --- |
| `404` | The recording UUID does not exist. |
| `413` | The saved recording exceeds the 25 MB transcription limit. |
| `415` | The saved recording format is not supported for transcription. |
| `422` | No speech was detected, speaker labels are missing, more than two speakers were detected, or two-speaker roles are ambiguous. |
| `502` | OpenAI transcription or role attribution failed, or the completed transcript could not be saved. |
| `503` | `OPENAI_API_KEY` is not configured. |

The API returns `client` and `therapist` turns. Normalized `speaker_1` and `speaker_2` labels remain
only in the stored transcript for traceability. The role model may return an ambiguous decision;
the endpoint then rejects the result rather than guessing. It does not summarize, diagnose, or
rewrite the recording.

### Generate session insights

```http
POST /recordings/{recording_id}/insights
Content-Type: application/json

{
  "session_date": "2026-11-06"
}
```

The endpoint reads the server-side completed transcript, assigns stable segment IDs, calculates a
source hash, and requests a structured insight result. It validates the structured facts, actions,
sections, review flags, and transcript references before assigning an application-owned `draft_id`
and stable bullet `item_id` values. The immutable draft is atomically saved as
`<uuid>.insights.json`. Repeating the same request returns the cached artifact without another model
call. A schema-valid `insufficient_source` result is returned as HTTP `200`.

Important insight error responses:

| Status | Meaning |
| --- | --- |
| `404` | The recording UUID does not exist. |
| `409` | No completed transcript exists yet. |
| `502` | Generation, evidence validation, or insight storage failed. |
| `503` | `OPENAI_API_KEY` is not configured. |

### Review and edit session insights

```http
GET /recordings/{recording_id}/insights/review
PUT /recordings/{recording_id}/insights/review
Content-Type: application/json

{
  "draft_id": "sha256:...",
  "source_version": "sha256:...",
  "items": [
    {"item_id": "B1", "disposition": "edited", "displayed_text": "Updated wording"}
  ]
}
```

The review is a complete overlay on the immutable generated draft. Each generated bullet must be
retained, edited, or removed. Edited text is stored as `client_edited` provenance and is not treated
as transcript-backed wording. The reviewed layer is atomically saved as
`<uuid>.insights-review.json`. A stale draft or source version returns `409` instead of silently
applying edits to regenerated insights.

## Logging

Application logs are written to the backend process's standard error stream, next to Uvicorn's
access logs. When running locally, watch the terminal where `uvicorn` is running. With Docker
Compose, use:

```bash
docker compose logs --follow api
```

Logs use stable event names such as `recording_saved`, `transcription_started`,
`role_attribution_started`, `role_attribution_completed`, `transcription_completed`, and provider
failure events. A recording with no detected speech or ambiguous roles is logged as a warning
rather than an error. Logs include useful operational fields such as the server-generated recording
ID, file size, duration, speaker and turn counts, safe provider status code, and provider request ID.

Set `LOG_LEVEL` in `.env` to `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. The default is
`INFO`:

```dotenv
LOG_LEVEL=INFO
```

For privacy and security, logs never include API keys, authorization headers, transcript text,
audio content, prompts, or client filenames.

## Run with Docker Compose

Build and start the API:

```bash
docker compose up --build
```

Check it from another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Compose stores recordings in the named volume `moodi-backend_moodi_recordings`, mounted at
`/app/data/recordings`. The recordings survive container replacement and normal shutdowns.

Stop the service while preserving its recordings:

```bash
docker compose down
```

To deliberately remove the containers **and all recordings in the named volume**:

```bash
docker compose down --volumes
```

## Build and run the Docker image directly

```bash
docker build -t moodi-backend .
docker run --rm \
  --name moodi-backend \
  -p 8000:8000 \
  -e APP_ENV=development \
  -e RECORDINGS_DIR=/app/data/recordings \
  -e MAX_UPLOAD_MB=100 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e OPENAI_TRANSCRIPTION_MODEL=gpt-4o-transcribe-diarize \
  -e OPENAI_ROLE_ATTRIBUTION_MODEL=gpt-4o-mini \
  -v moodi_recordings:/app/data/recordings \
  moodi-backend
```

The image:

- Uses Python 3.13 slim.
- Installs production dependencies from the committed `uv.lock`.
- Excludes tests, caches, recordings, IDE files, Git metadata, and `.env` from its build context.
- Listens on the platform-provided `PORT`, falling back to `8000`.
- Includes a Docker health check for `/health`.

## Deployment options for testing

The current milestone stores files on a local filesystem. A hosted deployment therefore needs a
persistent volume mounted at `/app/data/recordings`. Without a volume, recordings disappear when
the container restarts or is redeployed.

### Railway — recommended for the first shared test environment

Railway automatically detects a root `Dockerfile` and supports persistent volumes on its trial,
Free, and paid plans.

1. Create a Railway project from the GitHub repository.
2. Let Railway build the detected `Dockerfile`.
3. Add a volume to the service and mount it at `/app/data/recordings`.
4. Set `APP_ENV=testing`, `RECORDINGS_DIR=/app/data/recordings`, and `MAX_UPLOAD_MB=100`.
5. Add `OPENAI_API_KEY` as a secret and set
   `OPENAI_TRANSCRIPTION_MODEL=gpt-4o-transcribe-diarize` and
   `OPENAI_ROLE_ATTRIBUTION_MODEL=gpt-4o-mini`.
6. Generate a public domain for the service.
7. Use `/health` as the health-check path.

Railway currently offers a 30-day trial with a one-time $5 credit, followed by a Free plan with
limited monthly credit and 0.5 GB of volume storage. See the official
[Railway pricing](https://docs.railway.com/pricing/plans),
[Dockerfile](https://docs.railway.com/builds/dockerfiles), and
[volume](https://docs.railway.com/volumes) documentation.

### Render — simple, but persistent uploads require a paid service

Render can deploy the Dockerfile as a web service. Its free web services are useful for checking
the API, but do not support persistent disks, so uploaded recordings can disappear after a restart
or deploy. For recording persistence, use a paid service with a disk mounted at
`/app/data/recordings` and set `RECORDINGS_DIR` to the same path.

See Render's official [Docker](https://render.com/docs/docker),
[free service](https://render.com/docs/free), and
[persistent disk](https://render.com/docs/disks) documentation.

### Fly.io — inexpensive persistent VM and volume

Fly.io is a good option when more infrastructure control is useful. Deploy the Docker image with a
small Fly Machine and mount a Fly Volume at `/app/data/recordings`. It requires a little more CLI
configuration than Railway. Fly currently lists shared compute starting around a few dollars per
month and persistent volumes at $0.15 per GB per month.

See the official [Fly.io pricing](https://fly.io/pricing/) and
[volume pricing](https://fly.io/docs/about/pricing/) documentation.

### Google Cloud Run — use later with object storage

Cloud Run is attractive for stateless, scale-to-zero APIs and includes a usage-based free tier.
It is not the simplest match for this milestone's local-file storage. Use it after recordings are
moved to object storage such as Google Cloud Storage, or after explicitly configuring a supported
volume mount.

See the official [Cloud Run pricing](https://cloud.google.com/run/pricing) and
[container runtime contract](https://docs.cloud.google.com/run/docs/container-contract).

### Storage and scaling limitation

A single local volume is appropriate for development and a small test environment. It ties the
service to one persistent filesystem and is not a good foundation for horizontal scaling. Before
running multiple replicas or moving to production, recordings should be stored in object storage
such as S3, Cloudflare R2, Google Cloud Storage, or Supabase Storage.

## Development guidelines

Keep this backend easy to understand and maintain, especially for contributors who are new to
Python.

- Prefer straightforward code over clever abstractions.
- Keep API routes small: validate HTTP input, call a service, map known errors, and return a
  response model.
- Put filesystem and business behavior in `app/services/`.
- Keep OpenAI-specific behavior inside `app/services/transcription.py`.
- Keep environment-backed configuration in `app/core/config.py`.
- Put Pydantic request and response models in `app/schemas/`.
- Use type hints for function parameters and return values.
- Use `pathlib.Path` for filesystem paths.
- Prefer a function over a class when no state or clear object boundary is needed.
- Use synchronous code unless a measured requirement needs background processing.
- Do not add a database, queue, worker, authentication system, or cloud service prematurely.
- Avoid new dependencies when the standard library or an existing dependency is clear enough.
- Never commit, log, or return `OPENAI_API_KEY` or raw provider error details.
- Mock the OpenAI client in tests; automated tests must not spend API credits.
- Follow nearby code patterns instead of introducing a second way to solve the same problem.
- Let Ruff handle formatting rather than formatting code manually.

When adding or changing an endpoint:

1. Add or update the route in `app/api/`.
2. Add schemas in `app/schemas/` when needed.
3. Put non-HTTP behavior in `app/services/`.
4. Register a new router in `app/main.py` if necessary.
5. Test the successful path and important failures.
6. Update this README when behavior, configuration, setup, or deployment changes.

### Dependency commands

Use uv for dependency management:

```bash
uv add package-name
uv add --dev package-name
uv remove package-name
uv sync --locked
```

Commit both `pyproject.toml` and `uv.lock` whenever dependencies change.

## Test coverage

The current test suite verifies:

- The health endpoint returns HTTP `200` and `{"status": "ok"}`.
- A successful upload returns HTTP `201` and saves the exact uploaded bytes.
- The multipart `file` field is required.
- Empty uploads are rejected and leave no file behind.
- Oversized uploads return HTTP `413` and their partial files are removed.
- A malicious client filename cannot escape the recordings directory.
- A saved M4A recording can be diarized and returned as `client` and `therapist` turns.
- Adjacent segments from the same speaker are merged.
- Missing labels and results with more than two speakers are rejected; one speaker is accepted as the client.
- Ambiguous roles and role-attribution provider failures do not leave a completed transcript.
- Unknown and invalid recording IDs are rejected.
- Oversized and raw AAC recordings are rejected before contacting OpenAI.
- OpenAI failures preserve the original recording and do not leave a transcript.
- Missing OpenAI configuration returns a service-unavailable response.
- Completed transcripts and matching insights are reused without repeated provider calls.
- Insight fact, action, and review-flag source references must resolve to real transcript segments.
- Invalid insight evidence gets one bounded regeneration attempt and is never persisted.
- Insight provider failures do not expose provider details or leave partial artifacts.
- Reviewed insights persist separately and never mutate the generated draft.

Tests use a temporary recordings directory and do not write into the development storage folder.

## Verification

Run all checks before committing:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
docker compose config
docker build -t moodi-backend:test .
```

The GitHub Actions workflow runs pytest, Ruff linting, and Ruff formatting checks for pushes and
pull requests.

If Ruff reports formatting changes, apply them with:

```bash
uv run ruff format .
```

Before committing, confirm that:

- The change solves one clear problem.
- Tests cover its behavior and important errors.
- All verification commands pass.
- No `.env`, recording, cache, IDE, or secret file is staged.
- `git status` contains only files intended for the commit.
- This README reflects any user-visible or operational changes.

## Current milestone boundaries

This repository currently provides durable local recording, transcription, role attribution, and
session insight generation. These features are deliberately deferred:

- User authentication and authorization.
- Database metadata and recording ownership.
- Supabase or other object storage.
- Audio-content validation beyond extension and MIME screening.
- Queues, workers, retries, and background jobs.
- Reminders, action execution, scheduling, or writes to clinical records.
- Horizontal scaling across multiple application replicas.

The next storage milestone should replace the local volume with object storage before introducing
multiple replicas or production traffic.


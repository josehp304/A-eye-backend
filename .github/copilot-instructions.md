## Repo purpose

This repository hosts a simple real-time object detection web service powered by YOLO (Ultralytics) and OpenCV. The server can either read a local webcam or accept JPEG frames posted by a remote frontend and serves a processed MJPEG stream. The system includes AI-powered analysis via Groq API and email alert functionality when persons are detected.

These instructions teach an automated coding assistant how to safely and effectively modify this project.

## Primary requirements (checklist for any change)

- Preserve user privacy: do not enable or open the server's local webcam by default.
- Prefer stream mode (frontend POST -> `/stream_input`) for public-facing deployments.
- Never commit secrets (API keys) into the repo. Use `.env` and the environment.
- Keep edits small and focused. Run the project's smoke tests and a quick manual run before finishing.
- Update `requirements.txt` when you add or remove Python dependencies.

## Quick contract for changes

- Inputs: Python source files, templates in `templates/`, static assets in `static/`, environment variables from `.env`.
- Outputs: runnable Flask app providing `/video_feed` (MJPEG), `/stream_input` (POST JPEG), and status/config endpoints under `/api/*`.
- Success criteria: app starts, `/api/status` returns 200 JSON, `/video_feed` yields multipart frames, `/stream_input` accepts a JPEG and sets `stream_active`.
- Error modes: missing dependencies, camera device unavailable, malformed POSTed frames, missing API key for optional AI analysis.

## Key files and where to look

- `web_app.py` — main Flask application and `WebObjectDetector` class (frame ingest, YOLO inference, MJPEG generator, email alerts).
- `templates/index.html` — simplified viewer page (display-only). `templates/index_complex.html` contains the capture-enabled page (backup).
- `static/` — JS/CSS used by frontend (may contain capture logic in older revisions).
- `requirements.txt` — pinned dependencies (keep updated).
- `.env` — contains GROQ_API_KEY, email credentials (FROM_EMAIL, EMAIL_PASSWORD, TO_EMAIL) and other secrets (not checked in).

## How to run locally (developer / CI)

1. Create/activate virtualenv (the repo contains a working `myenv` for local dev):

   - python -m venv myenv
   - source myenv/bin/activate

2. Install deps:

   - pip install -r requirements.txt

3. Start server (dev):

   - python web_app.py

4. Smoke checks (manual or in CI):

   - curl -s http://localhost:5000/api/status | jq .
   - curl -s http://localhost:5000/video_feed  # should stream multipart JPEGs
   - curl -s http://localhost:5000/api/test_email  # test email functionality if configured

If you need to test stream ingest, POST a JPEG to `/stream_input` and check `/api/status` for `stream_active: true`.

## Behavioural constraints for an AI editing agent

- Do not change default operation to open the local webcam. If you need to add a feature that uses the server webcam, gate it behind an explicit configuration variable and document it.
- Prefer adding code paths that default to 'stream' mode and only enable `use_webcam=True` when an operator explicitly requests it via `/api/switch_source/webcam` or config.
- Avoid long-running blocking calls on the Flask main thread. Use background threads or async patterns for non-blocking work (as the repository already does for AI analysis).
- Keep any new heavy computation optional and document resource requirements (CPU/GPU, memory).

## Quality gates

- Before committing any change:
  - Run a quick lint (flake8/black if present) and correct obvious style or syntax problems.
  - Start the server and verify `/api/status` returns 200.
  - If you altered dependencies, update `requirements.txt` (pip freeze) and include that change.

- In CI (recommended):
  - Install deps, run unit tests (if present), run the server in the background, and perform the smoke checks above.

## Suggested small improvements (safe, low-risk)

- Add a minimal unit/integration test that posts a sample JPEG to `/stream_input` and asserts `stream_active` becomes true.
- Add a Dockerfile and minimal `docker-compose.yml` for local reproducible runs. Document resource expectations (CPU vs GPU).
- Add a small health-check endpoint (e.g., `/healthz`) that returns OK if server running and model loaded.

## Notes on hosting and resources

- This application loads a YOLO model and does non-trivial CPU/GPU work. For production or public hosting prefer a host with adequate RAM/CPU (or GPU) and use HTTPS and authentication for the `/stream_input` endpoint.
- For demos: consider ngrok, Replit, or a small container on Fly.io. For sustained use, choose a cloud VM with a GPU or a managed inference service.

## When you are blocked

- If tests fail or a behavior is unclear, open a short issue describing the exact failing command and error output.
- If secrets or keys are needed to reproduce a feature (e.g., Groq API), mock or gate the feature behind a config flag so CI and local dev do not require a key.

## Commit message guidance

- Use concise imperative messages. Example: "web_app: add stream ingest endpoint and protect local webcam by default"

## Final rule

When in doubt make the smallest change that achieves the goal, prefer safe defaults (no webcam, no keys in repo), and verify the application starts and basic endpoints return healthy responses before finishing.

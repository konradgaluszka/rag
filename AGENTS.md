# Repository Guidelines

## Project Structure & Module Organization
- `backend/` contains the ingestion pipeline (`backend/main.py`), FastAPI server (`backend/server.py`), tests, and runtime artifacts.
- `frontend/` contains the Next.js + Tailwind UI (`frontend/app/`).
- `backend/tests/integration_test/` holds integration coverage, including `integration_test.py` and sample HTML inputs in `backend/tests/integration_test/ships_input_data/`.
- `backend/docker-compose.yml` defines the local Qdrant service; persistent storage is mounted at `./backend/qdrant_data/`.
- `backend/index/` and `backend/qdrant_data/` are runtime artifacts; avoid committing generated contents.

## Build, Test, and Development Commands
- `python -m venv backend/.venv && source backend/.venv/bin/activate` to create a local environment.
- `pip install -r backend/requirements.txt` installs backend dependencies.
- `python -m backend.main --input ./data/raw --out ./backend/index` runs the ingestion pipeline (local index).
- `python -m backend.main --query "your question" --out ./backend/index` runs a query against an existing index.
- `uvicorn backend.server:app --host 127.0.0.1 --port 8000` starts the API server.
- `cd frontend && npm install && npm run dev` starts the Next.js UI (configure `NEXT_PUBLIC_API_BASE_URL` as needed).
- `docker compose -f backend/docker-compose.yml up -d` starts Qdrant; use `--use_qdrant` and `--qdrant_url http://localhost:6333` to ingest/query via Qdrant.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and standard PEP 8 naming: `snake_case` for functions/vars, `CamelCase` for classes, and `UPPER_CASE` constants.
- Keep helper functions in `main.py` cohesive; prefer small utilities over large monoliths.
- Use descriptive CLI flag names that match existing patterns (`--use_qdrant`, `--allow_fallback`).

## Testing Guidelines
- Tests use the stdlib `unittest` framework.
- Integration tests live in `backend/tests/integration_test/` and should be named `*_test.py` or similar (see `integration_test.py`).
- Run with the project venv: `backend/.venv/bin/python -m unittest backend/tests/integration_test/integration_test.py`.
- Integration tests start Qdrant via Docker Compose; ensure Docker is running.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative messages (e.g., “Add integration tests”).
- Keep commits focused and avoid mixing refactors with behavior changes.
- PRs should include a concise description, test results (or reason for skipping), and any required setup changes (e.g., Qdrant config).

## Configuration & Data Notes
- Qdrant URL defaults to `http://localhost:6333`; API key can be passed via `--qdrant_api_key` or `QDRANT_API_KEY`.
- Embedding backend selection is explicit; use `--prefer_neural` and opt-in to fallback via `--allow_fallback`.

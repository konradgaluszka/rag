# Repository Guidelines

## Project Structure & Module Organization
- `main.py` contains the ingestion, chunking, embedding, and retrieval pipeline plus CLI entrypoints.
- `tests/integration_test/` holds integration coverage, including `integration_test.py` and sample HTML inputs in `tests/integration_test/ships_input_data/`.
- `docker-compose.yml` defines the local Qdrant service; persistent storage is mounted at `./qdrant_data/`.
- `index/` and `qdrant_data/` are runtime artifacts; avoid committing generated contents.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` to create a local environment.
- `pip install -r requirements.txt` installs runtime dependencies.
- `python main.py --input ./data/raw --out ./index` runs the ingestion pipeline (local index).
- `python main.py --query "your question" --out ./index` runs a query against an existing index.
- `docker compose up -d` starts Qdrant; use `--use_qdrant` and `--qdrant_url http://localhost:6333` to ingest/query via Qdrant.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and standard PEP 8 naming: `snake_case` for functions/vars, `CamelCase` for classes, and `UPPER_CASE` constants.
- Keep helper functions in `main.py` cohesive; prefer small utilities over large monoliths.
- Use descriptive CLI flag names that match existing patterns (`--use_qdrant`, `--allow_fallback`).

## Testing Guidelines
- Tests use the stdlib `unittest` framework.
- Integration tests live in `tests/integration_test/` and should be named `*_test.py` or similar (see `integration_test.py`).
- Run with the project venv: `.venv/bin/python -m unittest tests/integration_test/integration_test.py`.
- Integration tests start Qdrant via Docker Compose; ensure Docker is running.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative messages (e.g., “Add integration tests”).
- Keep commits focused and avoid mixing refactors with behavior changes.
- PRs should include a concise description, test results (or reason for skipping), and any required setup changes (e.g., Qdrant config).

## Configuration & Data Notes
- Qdrant URL defaults to `http://localhost:6333`; API key can be passed via `--qdrant_api_key` or `QDRANT_API_KEY`.
- Embedding backend selection is explicit; use `--prefer_neural` and opt-in to fallback via `--allow_fallback`.

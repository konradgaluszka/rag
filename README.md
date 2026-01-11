# RAG Offline Pipeline with Qdrant

This project ingests documents, chunks them, embeds the chunks, and stores vectors for retrieval. It supports storing vectors in **Qdrant** as a vector database and exposes a FastAPI service with a Next.js frontend.

## Start Qdrant

Use Docker Compose to run a local Qdrant instance:

```bash
docker compose -f backend/docker-compose.yml up -d
```

Qdrant will be available at `http://localhost:6333`.

## Backend Setup

```bash
python -m venv backend/.venv && source backend/.venv/bin/activate
pip install -r backend/requirements.txt
```

Run the pipeline locally:

```bash
python -m backend.main \
  --input ./data/raw \
  --out ./backend/index
```

## Ingest and Store Vectors in Qdrant

Run the ingestion pipeline and store vectors in Qdrant:

```bash
python -m backend.main \
  --input ./data/raw \
  --out ./backend/index \
  --prefer_neural \
  --use_qdrant \
  --qdrant_url http://localhost:6333 \
  --qdrant_collection rag_chunks
```

Notes:
- The output folder still stores metadata (like `chunks.jsonl` and `embedder.json`).
- For Qdrant Cloud, set the API key via `--qdrant_api_key` or the `QDRANT_API_KEY` environment variable (the key is not persisted to disk).

## Query After Ingestion

You can query the stored vectors by pointing to the same `--out` directory used during ingestion:

```bash
python -m backend.main --out ./backend/index --query "your question" --top_k 5
```

The retrieval will fetch results from Qdrant when the index metadata indicates it was used during ingestion.

## Backend API

Run the API server:

```bash
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Endpoints:
- `POST /api/upload` (multipart form, field name `files`)
- `POST /api/query` (JSON body: `{"query": "your question", "top_k": 5}`)

## Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

By default, the UI expects the API at `http://localhost:8000`. Override with:
`NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`.

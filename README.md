# RAG Offline Pipeline with Qdrant

This project ingests documents, chunks them, embeds the chunks, and stores vectors for retrieval. It now supports storing vectors in **Qdrant** as a vector database.

## Start Qdrant

Use Docker to run a local Qdrant instance:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:latest
```

Qdrant will be available at `http://localhost:6333`.

## Install with Poetry

```bash
poetry install
```

Run the pipeline via Poetry:

```bash
poetry run python main.py \
  --input ./data/raw \
  --out ./index
```

## Ingest and Store Vectors in Qdrant

Run the ingestion pipeline and store vectors in Qdrant:

```bash
poetry run python main.py \
  --input ./data/raw \
  --out ./index \
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
python main.py --out ./index --query "your question" --top_k 5
```

The retrieval will fetch results from Qdrant when the index metadata indicates it was used during ingestion.

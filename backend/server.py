import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import main

app = FastAPI(title="RAG API")

_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [origin.strip() for origin in _cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


def _get_config() -> dict:
    return {
        "chunk_words": int(os.environ.get("CHUNK_WORDS", "900")),
        "overlap_words": int(os.environ.get("OVERLAP_WORDS", "120")),
        "model_name": os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2"),
        "qdrant_url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
        "qdrant_collection": os.environ.get("QDRANT_COLLECTION", "rag_chunks"),
        "qdrant_api_key": os.environ.get("QDRANT_API_KEY"),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    config = _get_config()
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir)
        for idx, upload_file in enumerate(files):
            filename = Path(upload_file.filename or f"upload_{idx}").name
            dest = input_dir / filename
            content = await upload_file.read()
            dest.write_bytes(content)

        summary = main.build_offline_pipeline(
            input_dir=input_dir,
            chunk_size_words=config["chunk_words"],
            overlap_words=config["overlap_words"],
            model_name=config["model_name"],
            qdrant_url=config["qdrant_url"],
            qdrant_collection=config["qdrant_collection"],
            qdrant_api_key=config["qdrant_api_key"],
        )
    return {"status": "ok", "summary": summary}


@app.post("/api/query")
def query(request: QueryRequest) -> dict:
    config = _get_config()
    embedder = main.build_embedder(model_name=config["model_name"])
    index = main.build_index(
        qdrant_url=config["qdrant_url"],
        qdrant_collection=config["qdrant_collection"],
        qdrant_api_key=config["qdrant_api_key"],
    )

    qv = embedder.embed([request.query])[0]
    payloads, scores = index.search(qv, top_k=request.top_k)
    results = []
    for payload, score in zip(payloads, scores):
        results.append(
            {
                "score": float(score),
                "chunk_id": payload.get("chunk_id"),
                "title": payload.get("title"),
                "chunk_index": payload.get("chunk_index"),
                "source_path": payload.get("source_path"),
                "text": payload.get("text"),
            }
        )

    return {"query": request.query, "top_k": request.top_k, "results": results}

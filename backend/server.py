import json
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


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_config() -> dict:
    return {
        "out_dir": Path(os.environ.get("RAG_OUT_DIR", "./index")),
        "chunk_words": int(os.environ.get("CHUNK_WORDS", "900")),
        "overlap_words": int(os.environ.get("OVERLAP_WORDS", "120")),
        "prefer_neural": _bool_env("PREFER_NEURAL", True),
        "model_name": os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2"),
        "allow_fallback": _bool_env("ALLOW_FALLBACK", False),
        "use_qdrant": True,
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
    out_dir: Path = config["out_dir"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir)
        for idx, upload_file in enumerate(files):
            filename = Path(upload_file.filename or f"upload_{idx}").name
            dest = input_dir / filename
            content = await upload_file.read()
            dest.write_bytes(content)

        main.build_offline_pipeline(
            input_dir=input_dir,
            out_dir=out_dir,
            chunk_size_words=config["chunk_words"],
            overlap_words=config["overlap_words"],
            prefer_neural=config["prefer_neural"],
            model_name=config["model_name"],
            prefer_faiss=False,
            allow_fallback=config["allow_fallback"],
            use_qdrant=config["use_qdrant"],
            qdrant_url=config["qdrant_url"],
            qdrant_collection=config["qdrant_collection"],
            qdrant_api_key=config["qdrant_api_key"],
        )

    summary_path = out_dir / "build_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {"status": "ok", "summary": summary}


@app.post("/api/query")
def query(request: QueryRequest) -> dict:
    config = _get_config()
    out_dir: Path = config["out_dir"]
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="Index directory not found.")

    chunks = main.load_chunks(out_dir)
    embedder = main.load_embedder(out_dir)
    index = main.load_index(out_dir)

    qv = embedder.embed([request.query])[0]
    results = []
    if isinstance(index, main.QdrantCosineIndex):
        payloads, scores = index.search(qv, top_k=request.top_k)
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
    else:
        idxs, scores = index.search(qv, top_k=request.top_k)
        for idx, score in zip(idxs, scores):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[int(idx)]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "source_path": chunk.source_path,
                    "text": chunk.text,
                }
            )

    return {"query": request.query, "top_k": request.top_k, "results": results}

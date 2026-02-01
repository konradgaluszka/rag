"""
RAG offline pipeline (ingestion → chunking → embeddings → vector index)

What this script does:
1) Loads documents from a folder (PDF / Markdown / HTML / TXT)
2) Cleans text
3) Chunks text with overlap
4) Embeds chunks (SentenceTransformers)
5) Stores vectors in Qdrant
6) Writes a local build summary for debugging

Run:
  python -m backend.main --input ./data/raw --chunk_words 900 --overlap_words 120

Optional installs (recommended):
  pip install pypdf beautifulsoup4 markdown sentence-transformers

Then you can test retrieval:
  python -m backend.main --query "your question" --qdrant_url http://localhost:6333
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# -----------------------------
# Data models
# -----------------------------

@dataclass
class Document:
    doc_id: str
    source_path: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    chunk_index: int
    text: str


# -----------------------------
# Ingestion (PDF / MD / HTML / TXT)
# -----------------------------

def _stable_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def clean_text(text: str) -> str:
    # Minimal cleanup that works reasonably across PDFs/HTML/MD
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("Missing dependency for PDFs. Install: pip install pypdf") from e

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        # Note: PDF extraction quality varies; for learning RAG this is OK.
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except Exception as e:
        raise RuntimeError("Missing dependency for HTML. Install: pip install beautifulsoup4") from e

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Drop script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return text


def load_markdown(path: Path) -> str:
    # Prefer rendering MD → HTML → text so we keep headings reasonably
    try:
        import markdown as md
        from bs4 import BeautifulSoup
    except Exception:
        # Fallback: raw markdown text
        return path.read_text(encoding="utf-8", errors="ignore")

    raw = path.read_text(encoding="utf-8", errors="ignore")
    html = md.markdown(raw, extensions=["fenced_code", "tables"])
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    return text


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def ingest_folder(input_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext not in {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}:
            continue

        if ext == ".pdf":
            text = load_pdf(path)
        elif ext in {".html", ".htm"}:
            text = load_html(path)
        elif ext in {".md", ".markdown"}:
            text = load_markdown(path)
        else:
            text = load_text(path)

        text = clean_text(text)
        if not text:
            continue

        source_path = str(path.resolve())
        doc_id = _stable_id(source_path)
        title = path.stem

        docs.append(Document(doc_id=doc_id, source_path=source_path, title=title, text=text))

    return docs


# -----------------------------
# Chunking (word-based, overlap)
# -----------------------------

def chunk_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def make_chunks(docs: List[Document], chunk_size_words: int, overlap_words: int) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in docs:
        parts = chunk_words(doc.text, chunk_size_words, overlap_words)
        for idx, part in enumerate(parts):
            chunk_id = _stable_id(f"{doc.doc_id}:{idx}:{part[:64]}")
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    title=doc.title,
                    chunk_index=idx,
                    text=part,
                )
            )
    return all_chunks


# -----------------------------
# Embeddings (SentenceTransformers only)
# -----------------------------

class Embedder:
    def fit(self, texts: List[str]) -> None:
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(vecs, dtype=np.float32)
def build_embedder(model_name: str) -> Embedder:
    try:
        return SentenceTransformerEmbedder(model_name=model_name)
    except Exception as e:
        raise RuntimeError("SentenceTransformers unavailable; install sentence-transformers") from e


# -----------------------------
# Vector index (Qdrant)
# -----------------------------

class VectorIndex:
    def build(self, vectors: np.ndarray, chunks: Optional[List[Chunk]] = None) -> None:
        raise NotImplementedError

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[Iterable, Iterable]:
        """Returns results and scores where scores are cosine similarity (higher is better)."""
        raise NotImplementedError


class QdrantCosineIndex(VectorIndex):
    def __init__(self, url: str, collection: str, api_key: Optional[str] = None):
        from qdrant_client import QdrantClient  # type: ignore

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.url = url
        self.api_key = api_key

    def build(self, vectors: np.ndarray, chunks: Optional[List[Chunk]] = None) -> None:
        if chunks is None:
            raise RuntimeError("Qdrant index build requires chunks metadata.")
        if len(chunks) != vectors.shape[0]:
            raise RuntimeError("Chunks and vectors must have matching lengths.")

        from qdrant_client.http import models as qmodels  # type: ignore

        dim = vectors.shape[1]
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = asdict(chunk)
            # Qdrant point IDs must be unsigned int or UUID.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[List[Dict], List[float]]:
        q = query_vec.astype(np.float32)
        if q.ndim != 1:
            raise RuntimeError("Qdrant search expects a 1D query vector.")

        response = self.client.query_points(
            collection_name=self.collection,
            query=q.tolist(),
            limit=top_k,
            with_payload=True,
        )
        points = response.points
        payloads = [p.payload or {} for p in points]
        scores = [float(p.score) for p in points]
        return payloads, scores


def build_index(
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_api_key: Optional[str],
) -> VectorIndex:
    return QdrantCosineIndex(
        url=qdrant_url,
        collection=qdrant_collection,
        api_key=qdrant_api_key,
    )


# -----------------------------
# Build pipeline
# -----------------------------

def build_offline_pipeline(
    input_dir: Path,
    chunk_size_words: int,
    overlap_words: int,
    model_name: str,
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_api_key: Optional[str],
) -> Dict[str, int]:

    docs = ingest_folder(input_dir)
    if not docs:
        raise RuntimeError(f"No supported documents found in {input_dir}")

    chunks = make_chunks(docs, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
    if not chunks:
        raise RuntimeError("No chunks created (documents empty after cleaning?)")

    texts = [c.text for c in chunks]

    embedder = build_embedder(model_name=model_name)
    print(f"Using embedder: {embedder.__class__.__name__}")
    embedder.fit(texts)

    vectors = embedder.embed(texts)
    if vectors.ndim != 2:
        raise RuntimeError("Embeddings must be a 2D array (n_chunks, dim)")

    index = build_index(
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        qdrant_api_key=qdrant_api_key,
    )
    index.build(vectors, chunks=chunks)

    # Useful build summary
    summary = {
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "embedding_dim": int(vectors.shape[1]),
        "chunk_size_words": chunk_size_words,
        "overlap_words": overlap_words,
    }
    print("Built index:", json.dumps(summary, indent=2))
    return summary


# -----------------------------
# Quick retrieval test (no generation yet)
# -----------------------------

def retrieve(
    query: str,
    top_k: int,
    model_name: str,
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_api_key: Optional[str],
) -> None:
    embedder = build_embedder(model_name=model_name)
    index = build_index(
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        qdrant_api_key=qdrant_api_key,
    )

    qv = embedder.embed([query])[0]
    print(f"\nQuery: {query}\nTop {top_k} results:\n")
    payloads, scores = index.search(qv, top_k=top_k)
    for rank, (payload, score) in enumerate(zip(payloads, scores), start=1):
        preview = str(payload.get("text", ""))[:280].replace("\n", " ")
        title = payload.get("title", "unknown")
        chunk_index = payload.get("chunk_index", "n/a")
        chunk_id = payload.get("chunk_id", "n/a")
        source_path = payload.get("source_path", "n/a")
        print(f"{rank:2d}. score={float(score):.4f}  doc={title}  chunk={chunk_index}  id={chunk_id}")
        print(f"    {preview}")
        print(f"    source={source_path}\n")


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None, help="Folder with raw docs (PDF/MD/HTML/TXT)")
    p.add_argument("--chunk_words", type=int, default=900)
    p.add_argument("--overlap_words", type=int, default=120)
    p.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--qdrant_url", type=str, default="http://localhost:6333")
    p.add_argument("--qdrant_collection", type=str, default="rag_chunks")
    p.add_argument("--qdrant_api_key", type=str, default=None)
    p.add_argument("--query", type=str, default=None, help="If set, run a retrieval test using Qdrant")
    p.add_argument("--top_k", type=int, default=5)

    args = p.parse_args()
    if args.query is not None:
        retrieve(
            query=args.query,
            top_k=args.top_k,
            model_name=args.model_name,
            qdrant_url=args.qdrant_url,
            qdrant_collection=args.qdrant_collection,
            qdrant_api_key=args.qdrant_api_key,
        )
        return

    if args.input is None:
        raise SystemExit("Provide --input to build the index, or --query to test Qdrant retrieval.")

    build_offline_pipeline(
        input_dir=Path(args.input),
        chunk_size_words=args.chunk_words,
        overlap_words=args.overlap_words,
        model_name=args.model_name,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
        qdrant_api_key=args.qdrant_api_key,
    )


if __name__ == "__main__":
    main()

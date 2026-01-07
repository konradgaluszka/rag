"""
RAG offline pipeline (ingestion → chunking → embeddings → vector index)

What this script does:
1) Loads documents from a folder (PDF / Markdown / HTML / TXT)
2) Cleans text
3) Chunks text with overlap
4) Embeds chunks (SentenceTransformers if available; otherwise TF-IDF fallback)
5) Builds a vector index (FAISS if available; otherwise sklearn NearestNeighbors)
6) Saves index + metadata to an output folder

Run:
  python rag_offline_pipeline.py --input ./data/raw --out ./index --chunk_words 900 --overlap_words 120

Optional installs (recommended):
  pip install pypdf beautifulsoup4 markdown sentence-transformers faiss-cpu

Fallback-only installs (if you want zero neural deps):
  pip install pypdf beautifulsoup4 markdown scikit-learn

Then you can test retrieval:
  python rag_offline_pipeline.py --query "your question" --out ./index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
# Embeddings (SentenceTransformers preferred; TF-IDF fallback)
# -----------------------------

class Embedder:
    def fit(self, texts: List[str]) -> None:
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, out_dir: Path) -> None:
        pass

    @classmethod
    def load(cls, out_dir: Path) -> "Embedder":
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(vecs, dtype=np.float32)

    def save(self, out_dir: Path) -> None:
        meta = {"type": "sentence_transformers", "model_name": self.model_name}
        (out_dir / "embedder.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path) -> "SentenceTransformerEmbedder":
        meta = json.loads((out_dir / "embedder.json").read_text(encoding="utf-8"))
        return cls(model_name=meta["model_name"])


class TfidfEmbedder(Embedder):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=200_000,
            ngram_range=(1, 2),
        )

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)

    def embed(self, texts: List[str]) -> np.ndarray:
        # Returns dense float32 matrix for simplicity (OK for learning projects)
        mat = self.vectorizer.transform(texts)
        dense = mat.toarray().astype(np.float32)
        # Normalize so cosine similarity works nicely
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
        return dense / norms

    def save(self, out_dir: Path) -> None:
        import joblib  # type: ignore
        meta = {"type": "tfidf"}
        (out_dir / "embedder.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        joblib.dump(self.vectorizer, out_dir / "tfidf_vectorizer.joblib")

    @classmethod
    def load(cls, out_dir: Path) -> "TfidfEmbedder":
        import joblib  # type: ignore
        inst = cls()
        inst.vectorizer = joblib.load(out_dir / "tfidf_vectorizer.joblib")
        return inst


def build_embedder(prefer_neural: bool, model_name: str) -> Embedder:
    if prefer_neural:
        try:
            return SentenceTransformerEmbedder(model_name=model_name)
        except Exception:
            pass
    return TfidfEmbedder()


def load_embedder(out_dir: Path) -> Embedder:
    meta = json.loads((out_dir / "embedder.json").read_text(encoding="utf-8"))
    if meta["type"] == "sentence_transformers":
        return SentenceTransformerEmbedder.load(out_dir)
    if meta["type"] == "tfidf":
        return TfidfEmbedder.load(out_dir)
    raise RuntimeError(f"Unknown embedder type: {meta}")


# -----------------------------
# Vector index (FAISS preferred; sklearn fallback)
# -----------------------------

class VectorIndex:
    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (indices, scores) where scores are cosine similarity (higher is better)."""
        raise NotImplementedError

    def save(self, out_dir: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, out_dir: Path) -> "VectorIndex":
        raise NotImplementedError


class FaissCosineIndex(VectorIndex):
    def __init__(self):
        import faiss  # type: ignore
        self.faiss = faiss
        self.index = None
        self.vectors = None  # we may keep for debugging; not required

    def build(self, vectors: np.ndarray) -> None:
        # Assume vectors are L2-normalized. Then inner product == cosine similarity.
        d = vectors.shape[1]
        index = self.faiss.IndexFlatIP(d)
        index.add(vectors)
        self.index = index

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        q = query_vec.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        scores, idx = self.index.search(q, top_k)
        return idx[0], scores[0]

    def save(self, out_dir: Path) -> None:
        if self.index is None:
            raise RuntimeError("Index not built.")
        meta = {"type": "faiss_ip"}
        (out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.faiss.write_index(self.index, str(out_dir / "faiss.index"))

    @classmethod
    def load(cls, out_dir: Path) -> "FaissCosineIndex":
        import faiss  # type: ignore
        inst = cls()
        inst.index = faiss.read_index(str(out_dir / "faiss.index"))
        return inst


class SklearnCosineIndex(VectorIndex):
    def __init__(self):
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        self.nn = NearestNeighbors(metric="cosine", algorithm="auto")
        self.vectors: Optional[np.ndarray] = None

    def build(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype(np.float32)
        self.nn.fit(self.vectors)

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            raise RuntimeError("Index not built.")
        q = query_vec.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        distances, indices = self.nn.kneighbors(q, n_neighbors=top_k)
        # cosine distance = 1 - cosine similarity
        scores = 1.0 - distances[0]
        return indices[0], scores

    def save(self, out_dir: Path) -> None:
        if self.vectors is None:
            raise RuntimeError("Index not built.")
        meta = {"type": "sklearn_cosine"}
        (out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        np.save(out_dir / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, out_dir: Path) -> "SklearnCosineIndex":
        inst = cls()
        inst.vectors = np.load(out_dir / "vectors.npy").astype(np.float32)
        inst.nn.fit(inst.vectors)
        return inst


def build_index(prefer_faiss: bool) -> VectorIndex:
    if prefer_faiss:
        try:
            return FaissCosineIndex()
        except Exception:
            pass
    return SklearnCosineIndex()


def load_index(out_dir: Path) -> VectorIndex:
    meta = json.loads((out_dir / "index_meta.json").read_text(encoding="utf-8"))
    if meta["type"] == "faiss_ip":
        return FaissCosineIndex.load(out_dir)
    if meta["type"] == "sklearn_cosine":
        return SklearnCosineIndex.load(out_dir)
    raise RuntimeError(f"Unknown index type: {meta}")


# -----------------------------
# Persistence: chunks metadata
# -----------------------------

def save_chunks(chunks: List[Chunk], out_dir: Path) -> None:
    out_path = out_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(out_dir: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with (out_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(Chunk(**obj))
    return chunks


# -----------------------------
# Build pipeline
# -----------------------------

def build_offline_pipeline(
    input_dir: Path,
    out_dir: Path,
    chunk_size_words: int,
    overlap_words: int,
    prefer_neural: bool,
    model_name: str,
    prefer_faiss: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = ingest_folder(input_dir)
    if not docs:
        raise RuntimeError(f"No supported documents found in {input_dir}")

    chunks = make_chunks(docs, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
    if not chunks:
        raise RuntimeError("No chunks created (documents empty after cleaning?)")

    texts = [c.text for c in chunks]

    embedder = build_embedder(prefer_neural=prefer_neural, model_name=model_name)
    # TF-IDF needs fit; SentenceTransformers doesn't
    try:
        embedder.fit(texts)
    except Exception:
        pass

    vectors = embedder.embed(texts)
    if vectors.ndim != 2:
        raise RuntimeError("Embeddings must be a 2D array (n_chunks, dim)")

    index = build_index(prefer_faiss=prefer_faiss)
    index.build(vectors)

    # Save artifacts
    save_chunks(chunks, out_dir)
    embedder.save(out_dir)
    index.save(out_dir)

    # Useful build summary
    summary = {
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "embedding_dim": int(vectors.shape[1]),
        "chunk_size_words": chunk_size_words,
        "overlap_words": overlap_words,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Built index:", json.dumps(summary, indent=2))


# -----------------------------
# Quick retrieval test (no generation yet)
# -----------------------------

def retrieve(out_dir: Path, query: str, top_k: int) -> None:
    chunks = load_chunks(out_dir)
    embedder = load_embedder(out_dir)
    index = load_index(out_dir)

    # Embed query
    qv = embedder.embed([query])[0]
    idx, scores = index.search(qv, top_k=top_k)

    print(f"\nQuery: {query}\nTop {top_k} results:\n")
    for rank, (i, s) in enumerate(zip(idx, scores), start=1):
        if i < 0 or i >= len(chunks):
            continue
        c = chunks[int(i)]
        preview = c.text[:280].replace("\n", " ")
        print(f"{rank:2d}. score={float(s):.4f}  doc={c.title}  chunk={c.chunk_index}  id={c.chunk_id}")
        print(f"    {preview}")
        print(f"    source={c.source_path}\n")


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None, help="Folder with raw docs (PDF/MD/HTML/TXT)")
    p.add_argument("--out", type=str, required=True, help="Output folder for index artifacts")
    p.add_argument("--chunk_words", type=int, default=900)
    p.add_argument("--overlap_words", type=int, default=120)
    p.add_argument("--prefer_neural", action="store_true", help="Prefer SentenceTransformers embeddings")
    p.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--prefer_faiss", action="store_true", help="Prefer FAISS index")
    p.add_argument("--query", type=str, default=None, help="If set, run a retrieval test using existing index")
    p.add_argument("--top_k", type=int, default=5)

    args = p.parse_args()
    out_dir = Path(args.out)

    if args.query is not None:
        retrieve(out_dir=out_dir, query=args.query, top_k=args.top_k)
        return

    if args.input is None:
        raise SystemExit("Provide --input to build the index, or --query to test an existing index.")

    build_offline_pipeline(
        input_dir=Path(args.input),
        out_dir=out_dir,
        chunk_size_words=args.chunk_words,
        overlap_words=args.overlap_words,
        prefer_neural=args.prefer_neural,
        model_name=args.model_name,
        prefer_faiss=args.prefer_faiss,
    )


if __name__ == "__main__":
    main()

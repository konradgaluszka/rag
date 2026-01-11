"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function UploadPage() {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState("idle");
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  const onFileChange = (event) => {
    setFiles(Array.from(event.target.files || []));
    setSummary(null);
    setError(null);
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!files.length) {
      setError("Select at least one file to upload.");
      return;
    }

    setStatus("uploading");
    setError(null);
    setSummary(null);

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file, file.name);
    });

    try {
      const response = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Upload failed.");
      }
      const data = await response.json();
      setSummary(data.summary || {});
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  };

  return (
    <section className="grid gap-8">
      <div className="rounded-3xl border border-emerald-200 bg-white/80 p-8 shadow-lg shadow-emerald-100">
        <h1 className="brand-title text-3xl font-semibold">Upload documents</h1>
        <p className="mt-2 text-sm text-slate-600">
          Drop multiple files to ingest them into Qdrant. The backend uses
          SentenceTransformers by default.
        </p>
        <form className="mt-6 grid gap-4" onSubmit={onSubmit}>
          <input
            type="file"
            multiple
            onChange={onFileChange}
            className="rounded-xl border border-slate-200 bg-white p-3 text-sm"
          />
          <button
            type="submit"
            disabled={status === "uploading"}
            className="w-fit rounded-full bg-emerald-600 px-6 py-3 text-sm font-semibold text-white shadow-md shadow-emerald-300 transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-emerald-300"
          >
            {status === "uploading" ? "Uploading..." : "Upload & ingest"}
          </button>
        </form>
        {error && <p className="mt-4 text-sm text-amber-700">{error}</p>}
      </div>

      {summary && (
        <div className="rounded-2xl border border-slate-200 bg-white/80 p-6">
          <h2 className="text-lg font-semibold">Ingestion summary</h2>
          <div className="mt-4 grid gap-2 text-sm text-slate-700">
            <p>
              <span className="font-semibold">Documents:</span>{" "}
              {summary.n_docs}
            </p>
            <p>
              <span className="font-semibold">Chunks:</span>{" "}
              {summary.n_chunks}
            </p>
            <p>
              <span className="font-semibold">Embedding dim:</span>{" "}
              {summary.embedding_dim}
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

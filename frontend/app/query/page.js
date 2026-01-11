"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function QueryPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim()) {
      setError("Enter a question to search.");
      return;
    }

    setStatus("searching");
    setError(null);
    setResults([]);

    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Query failed.");
      }
      const data = await response.json();
      setResults(data.results || []);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  };

  return (
    <section className="grid gap-8">
      <div className="rounded-3xl border border-amber-200 bg-white/80 p-8 shadow-lg shadow-amber-100">
        <h1 className="brand-title text-3xl font-semibold">Query the index</h1>
        <p className="mt-2 text-sm text-slate-600">
          Ask a question and inspect the most relevant chunks stored in Qdrant.
        </p>
        <form className="mt-6 grid gap-4" onSubmit={onSubmit}>
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="e.g. What happened to the Titanic?"
            className="rounded-xl border border-slate-200 bg-white p-3 text-sm"
          />
          <button
            type="submit"
            disabled={status === "searching"}
            className="w-fit rounded-full bg-amber-500 px-6 py-3 text-sm font-semibold text-white shadow-md shadow-amber-300 transition hover:bg-amber-600 disabled:cursor-not-allowed disabled:bg-amber-300"
          >
            {status === "searching" ? "Searching..." : "Run query"}
          </button>
        </form>
        {error && <p className="mt-4 text-sm text-amber-700">{error}</p>}
      </div>

      <div className="grid gap-4">
        {results.map((item, index) => (
          <article
            key={`${item.chunk_id}-${index}`}
            className="rounded-2xl border border-slate-200 bg-white/80 p-6"
          >
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>{item.title || "Untitled"}</span>
              <span>score: {item.score?.toFixed(4)}</span>
            </div>
            <p className="mt-3 text-sm text-slate-700">{item.text}</p>
          </article>
        ))}
        {status === "done" && results.length === 0 && (
          <p className="text-sm text-slate-600">No results found.</p>
        )}
      </div>
    </section>
  );
}

import Link from "next/link";

export default function Home() {
  return (
    <section className="grid gap-10">
      <div className="rounded-3xl border border-emerald-200/60 bg-white/70 p-10 shadow-lg shadow-emerald-100">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-emerald-700">
          Qdrant Studio
        </p>
        <h1 className="brand-title mt-3 text-4xl font-semibold">
          Upload fresh documents, then query them in seconds.
        </h1>
        <p className="mt-4 max-w-2xl text-base text-slate-700">
          This interface talks directly to the backend REST API. Use it to ingest
          PDFs, HTML, Markdown, or TXT files and search the vector store.
        </p>
        <div className="mt-8 flex flex-wrap gap-4">
          <Link
            href="/upload"
            className="rounded-full bg-emerald-600 px-6 py-3 text-sm font-semibold text-white shadow-md shadow-emerald-300 transition hover:bg-emerald-700"
          >
            Upload files
          </Link>
          <Link
            href="/query"
            className="rounded-full border border-emerald-600 px-6 py-3 text-sm font-semibold text-emerald-700 transition hover:bg-emerald-50"
          >
            Run a query
          </Link>
        </div>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white/80 p-6">
          <h2 className="text-lg font-semibold">Upload pipeline</h2>
          <p className="mt-2 text-sm text-slate-600">
            Files are chunked, embedded with your chosen model, and stored in
            Qdrant.
          </p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white/80 p-6">
          <h2 className="text-lg font-semibold">Query flow</h2>
          <p className="mt-2 text-sm text-slate-600">
            Questions are embedded, then matched against the vector index to
            return top-scoring chunks.
          </p>
        </div>
      </div>
    </section>
  );
}

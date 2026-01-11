import "./globals.css";
import Link from "next/link";

export const metadata = {
  title: "RAG Dock",
  description: "Upload documents and query a Qdrant-backed RAG index.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="relative isolate overflow-hidden">
          <div className="absolute inset-x-0 top-[-12rem] -z-10 blur-3xl">
            <div className="mx-auto h-64 w-96 rounded-full bg-emerald-200/40"></div>
          </div>
          <header className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 py-6">
            <Link href="/" className="brand-title text-2xl font-semibold">
              RAG Dock
            </Link>
            <nav className="flex items-center gap-4 text-sm font-medium">
              <Link className="hover:text-emerald-700" href="/upload">
                Upload
              </Link>
              <Link className="hover:text-emerald-700" href="/query">
                Query
              </Link>
            </nav>
          </header>
          <main className="mx-auto w-full max-w-5xl px-6 pb-16">{children}</main>
        </div>
      </body>
    </html>
  );
}

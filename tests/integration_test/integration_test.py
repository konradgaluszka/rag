import shutil
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import main


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("docker") is None:
            raise unittest.SkipTest("docker not available")

        cls._compose_file = Path(__file__).parents[2] / "docker-compose.yml"
        if not cls._compose_file.exists():
            raise unittest.SkipTest("docker-compose.yml not found")

        # Start Qdrant via docker compose.
        cls._compose_cmd = ["docker", "compose", "-f", str(cls._compose_file)]
        try:
            subprocess.run(cls._compose_cmd + ["up", "-d"], check=True)
        except subprocess.CalledProcessError as exc:
            raise unittest.SkipTest(f"Failed to start Qdrant via docker compose: {exc}") from exc

        # Wait for Qdrant to become ready.
        deadline = time.time() + 30
        url = "http://localhost:6333/"
        while True:
            try:
                with urllib.request.urlopen(url, timeout=2):
                    break
            except (urllib.error.URLError, ConnectionError):
                if time.time() >= deadline:
                    raise RuntimeError("Qdrant did not become ready in time")
                time.sleep(1)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_compose_cmd"):
            subprocess.run(cls._compose_cmd + ["down"], check=False)

    def test_pipeline_build_and_query(self) -> None:
        # Loads data from tests/integration_test/ships_input_data and runs a query.
        input_dir = Path(__file__).parent / "ships_input_data"
        self.assertTrue(input_dir.exists(), f"Missing test data at {input_dir}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "index"

            main.build_offline_pipeline(
                input_dir=input_dir,
                out_dir=out_dir,
                chunk_size_words=200,
                overlap_words=50,
                prefer_neural=True,
                model_name="all-MiniLM-L6-v2",
                prefer_faiss=False,
                allow_fallback=False,
                use_qdrant=True,
                qdrant_url="http://localhost:6333",
                qdrant_collection="rag_chunks_test",
                qdrant_api_key=None,
            )

            chunks = main.load_chunks(out_dir)
            self.assertTrue(chunks, "Expected chunks to be created")

            embedder = main.load_embedder(out_dir)
            index = main.load_index(out_dir)

            query = "Titanic"
            qv = embedder.embed([query])[0]
            results, _scores = index.search(qv, top_k=5)

            titanic_ids = {c.chunk_id for c in chunks if "Titanic" in c.title}
            found = False
            for payload in results:
                if payload.get("chunk_id") in titanic_ids:
                    found = True
                    break

            self.assertTrue(found, "Expected a Titanic-related chunk in top results")


if __name__ == "__main__":
    unittest.main()

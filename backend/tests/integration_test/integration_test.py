import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import requests


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

        # Start API server.
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls._api_port = 8001
        cls._api_url = f"http://127.0.0.1:{cls._api_port}"
        env = os.environ.copy()
        env["QDRANT_URL"] = "http://localhost:6333"
        env["QDRANT_COLLECTION"] = "rag_chunks_test"
        cls._api_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls._api_port),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for API to become ready.
        deadline = time.time() + 30
        while True:
            if cls._api_proc.poll() is not None:
                output = cls._api_proc.stdout.read() if cls._api_proc.stdout else ""
                raise RuntimeError(f"API server exited early:\n{output}")
            try:
                resp = requests.get(f"{cls._api_url}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() >= deadline:
                raise RuntimeError("API server did not become ready in time")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_api_proc"):
            cls._api_proc.terminate()
            try:
                cls._api_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls._api_proc.kill()
        if hasattr(cls, "_tmp_dir"):
            cls._tmp_dir.cleanup()
        if hasattr(cls, "_compose_cmd"):
            subprocess.run(cls._compose_cmd + ["down"], check=False)

    def test_pipeline_build_and_query(self) -> None:
        # Loads data from backend/tests/integration_test/ships_input_data and runs a query via REST.
        input_dir = Path(__file__).parent / "ships_input_data"
        self.assertTrue(input_dir.exists(), f"Missing test data at {input_dir}")

        files = []
        for path in sorted(input_dir.iterdir()):
            if path.is_file():
                files.append(("files", (path.name, path.read_bytes(), "text/html")))
        resp = requests.post(f"{self._api_url}/api/upload", files=files, timeout=120)
        self.assertEqual(resp.status_code, 200, resp.text)
        summary = resp.json().get("summary", {})
        self.assertGreater(summary.get("n_chunks", 0), 0)

        query_resp = requests.post(
            f"{self._api_url}/api/query",
            json={"query": "Titanic", "top_k": 5},
            timeout=30,
        )
        self.assertEqual(query_resp.status_code, 200, query_resp.text)
        results = query_resp.json().get("results", [])
        self.assertTrue(results, "Expected query results")
        found = any("Titanic" in (r.get("title") or "") for r in results)
        self.assertTrue(found, "Expected a Titanic-related chunk in top results")


if __name__ == "__main__":
    unittest.main()
